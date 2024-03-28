"""
Name:       Byn_Lake_valuation.py

Label:      Valuate an improvement in ecological status for Byn Lake by Nissum Fjord.

Summary:    ThorNoe.GitHub.io/GreenGDP explains the overall approach and methodology.

Rqmts:      None.

Usage:      Draw on Byn_Lake_variables.csv created by Byn_Lake_data.py to valuate an
            improvement in ecological status for Byn Lake by outer Nissum Fjord.
            See GitHub.com/ThorNoe/GreenGDP/tree/Byn-Lake for instructions.

License:    MIT Copyright (c) 2024
Author:     Thor Donsby Noe
"""

########################################################################################
#   0. Imports
########################################################################################
# Import Operation System (os) and ArcPy package (requires ArcGIS Pro installed)
import os

import numpy as np
import pandas as pd

########################################################################################
#   1. Setup
########################################################################################
# Specify the parent folder as the working directory of the operating system
root = r"C:\Users\au687527\GitHub\GreenGDP"
path = root + "\\gis"
os.chdir(path)

########################################################################################
#   2. Specifications and functions
########################################################################################
# Specify the name of the data file for CPI and NPV operator
data = {"shared": ["CPI_NPV.xlsx"]}


def BT(df, elast=1):
    """Apply Benefit Transfer equation from meta study (Zandersen et al., 2022)"""
    # ln MWTP for improvement from current ecological status to "Good"
    lnMWTP = (
        4.142
        + 0.551 * df["Q"]
        + elast * df["ln y"]
        + 0.496 * df["D age"]
        + 0.121 * df["ln PSL"]
        - 0.072 * df["ln PAL"]
        - 0.005 * df["SL"]
        - 0.378 * df["D lake"]
    )
    # Real MWTP per household (DKK, 2018 prices) using the meta study variance
    MWTP = np.exp(lnMWTP + (0.136 + 0.098) / 2)  #  variance components
    return MWTP


def valuation(dfBT, investment=False):
    """Valuation as either Cost of Water Pollution (CWP) or Investment Value (IV).
    If not set to return real values (2018 prices), instead returns values in the prices of both the current year and the preceding year (for year-by-year chain linking).
    """
    # Copy DataFrame with the variables needed for the benefit transfer equation
    df = dfBT.copy()

    # Define a small constant to avoid RuntimeWarning due to taking the log of 0
    epsilon = 1e-6  #  a millionth part

    if investment is False:
        # MWTP = 0 if all water bodies of type j have ≥ good ecological status
        df["nonzero"] = np.select([df["Q"] < 3 - epsilon], [1])  #  dummy

        # Distance from current to Good: transform mean Q to lnΔQ ≡ ln(good - Q)
        df["Q"] = df["Q"].mask(
            df["Q"] < 3 - epsilon,  # if some water bodies have < good status
            np.log(3 - df["Q"] + epsilon),  #  log-transform difference good - Q
        )

    else:
        # Actual change in ecological status since preceding year
        df = df.reorder_levels(["j", "v", "t"]).sort_index()  #  series by j & v
        df["Q"] = df["Q"].diff()  #  transform Q to be the change in Q since t-1
        df = df.reorder_levels(["j", "t", "v"]).sort_index()  #  series by j & t

        # Dummy used to set MWTP = 0 if actual change in water quality is zero
        df["nonzero"] = np.select([df["Q"] != 0], [1])  #  dummy

        # Mark if actual change is negative (used to switch MWTP to negative)
        df["neg"] = np.select([df["Q"] < 0], [1])  #  dummy

        # Transform Q to the log of the actual change in water quality since t-1
        df["Q"] = df["Q"].mask(
            df["Q"] != 0,  #  if actual change in water quality is nonzero
            np.log(np.abs(df["Q"]) + epsilon),  #  log-transform absolute value
        )

    # Drop year 1989 and specify integer values
    df = df.drop(df[df.index.get_level_values("t") == 1989].index)
    df[["D age", "D lake", "N"]] = df[["D age", "D lake", "N"]].astype(int)

    # Consumer Price Index by year t (1990-2020)
    CPI_NPV = pd.read_excel("data\\" + data["shared"][0], index_col=0)

    # Merge data with CPI to correct for assumption of unitary income elasticity
    kwargs = dict(how="left", left_index=True, right_index=True)
    df1 = df.merge(CPI_NPV, **kwargs)
    df1["unityMWTP"] = BT(df1)  #  MWTP assuming unitary income elasticity

    # Calculate factor that MWTP is increased by if using estimated income ε
    df2018 = df1[df1.index.get_level_values("t") == 2018].copy()
    df2018["elastMWTP"] = BT(df2018, elast=1.453)  #  meta reg income ε
    df2018["factor"] = df2018["elastMWTP"] / df2018["unityMWTP"]
    df2018 = df2018.droplevel("t")
    df2 = df1.merge(df2018[["factor"]], **kwargs)
    df2 = df2.reorder_levels(["j", "t", "v"]).sort_index()

    # Adjust with factor of actual ε over unitary ε; set MWTP to 0 for certain Q
    df2["MWTP"] = df2["unityMWTP"] * df2["factor"] * df2["nonzero"]

    # Aggregate real MWTP per hh over households in coastal catchment area
    df2["CWP"] = df2["MWTP"] * df2["N"] / 1e06  #  million DKK (2018 prices)

    if investment is True:
        # Switch MWTP to negative if actual change is negative
        cond = [df2["neg"] == 1]
        df2["CWP"] = np.select(cond, [-df2["CWP"]], default=df2["CWP"])

        # Declining r as prescribed by Ministry of Finance during 2014-2020
        df2["CWP"] = df2["CWP"] * CPI_NPV.loc[2018, "NPV"]

        # Rename CWP to IV (investment value of water quality improvements)
        df2["IV"] = df2["CWP"]  #  million DKK (2018 prices)

        return df2[["IV"]]  #  return real investment value by j, t, and v

    return df2[["CWP"]]  #  real cost of water pollution by j, t, v


########################################################################################
#   4.c Real cost of water pollution and investment in water quality for journal article
########################################################################################
# Read data created using Byn_Lake_data.py
df_BT_Byn = pd.read_csv("output\\Byn_Lake_variables.csv", index_col=[0, 1, 2])

# Costs of Water Pollution (CWP) in real terms (million DKK, 2018 prices)
CWP_v = valuation(df_BT_Byn)
msg_CWP = "The marginal willingness to pay for improving the ecological status of Byn Lake from Moderate to Good is estimated to be {0:,} DKK per household per year, amounting to a total cost of {1:,} DKK per year, using 2018 prices and demographics for the {2:,} households in the catchment area of outer Nissum Fjord.".format(
    int(
        1e6
        * CWP_v.loc[("lakes", 2018, 129), "CWP"]
        / df_BT_Byn.loc[("lakes", 2018, 129), "N"]
    ),
    int(1e6 * CWP_v.loc[("lakes", 2018, 129), "CWP"]),
    int(df_BT_Byn.loc[("lakes", 2018, 129), "N"]),
)
print(msg_CWP)

# Investment Value of water quality improvement in real terms (million DKK, 2018 prices)
df_BT_Byn_IV = df_BT_Byn.copy()  #  df for estimating investment value
df_BT_Byn_IV.iloc[-3:, 0] = 3  #  set all water bodies to good status after 2018
IV_v = valuation(df_BT_Byn_IV, investment=True)
msg_IV = "The investment value of improving the ecological status of Byn Lake from Moderate to Good is estimated to be {0:,} DKK per household, amounting to a total investment value of {1:,} DKK, using 2018 prices and demographics for the {2:,} households in the catchment area of outer Nissum Fjord and the net present value implied by the declining discount rate prescribed by the Ministry of Finance during 2014-2020.".format(
    int(
        1e6
        * IV_v.loc[("lakes", 2018, 129), "IV"]
        / df_BT_Byn.loc[("lakes", 2018, 129), "N"]
    ),
    int(1e6 * IV_v.loc[("lakes", 2018, 129), "IV"]),
    int(df_BT_Byn.loc[("lakes", 2018, 129), "N"]),
)
print(msg_IV)
