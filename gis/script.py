"""
Name:       script.py

Label:      Construct and map longitudinal data of ecological status of water bodies.

Summary:    ThorNoe.GitHub.io/GreenGDP explains the overall approach and methodology.

Rqmts:      ArcGIS Pro must be installed on the system and be up to date.

Usage:      This script supports WaterbodiesScriptTool in the gis.tbx toolbox.
            See GitHub.com/ThorNoe/GreenGDP for instructions to run or update it all.

License:    MIT Copyright (c) 2024
Author:     Thor Donsby Noe
"""

########################################################################################
#   0. Imports
########################################################################################
# Import Operation System (os) and ArcPy package (requires ArcGIS Pro installed)
import os

import arcpy
import pandas as pd

########################################################################################
#   1. Setup
########################################################################################
# Set overwrite option
arcpy.env.overwriteOutput = True

# Specify the parent folder as the working directory of the operating system
root = r"C:\Users\au687527\GitHub\GreenGDP"
path = root + "\\gis"
os.chdir(path)
# os.chdir(arcpy.GetParameterAsText(0))

# Specify whether to replace existing feature classes downloaded from WFS service
wfs_replace = 0
# wfs_replace = arcpy.GetParameterAsText(1)

# Specify whether to keep the geodatabase when the script finishes
keep_gdb = 1
# keep_gdb = arcpy.GetParameterAsText(2)

########################################################################################
#   2. Specifications
########################################################################################
# Span of natural capital account (1990 investment value depends on change from 1989)
year_first = 1989
year_last = 2020

# Specify the names of data files for each category of water body and shared statistics
data = {
    "coastal": ["coastal_chlorophyll.xlsx"],
    "lakes": ["lakes_chlorophyll.xlsx"],
    "streams": ["streams_DVFI.xlsx", "streams_1987-2020.xlsx"],
    "shared": ["CPI_NPV.xlsx", "demographics.csv", "geographical.xlsx"],
}

# Specify the names of the corresponding linkage files (and other supporting tables)
linkage = {
    "coastal": ["coastal_stations_VP3.csv", "coastal_chlorophyll_limits.csv"],
    "lakes": ["lakes_stations_VP3.csv", "lakes_stations_XY.csv"],
    "streams": ["streams_stations_VP3.csv"],
}

# WFS service URL for the current water body plan (VP2 is for 2015-2021)
wfs_service = "https://wfs2-miljoegis.mim.dk/vp3endelig2022/ows?service=WFS&request=Getcapabilities"

# For the WFS, specify the name of the feature class (fc) for each type of water body
wfs_fc = {
    "catch": "vp3e2022_kystvand_opland_afg",
    "coastal": "vp3e2022_marin_samlet_1mil",
    "lakes": "vp3e2022_soe_samlet",
    "streams": "vp3e2022_vandloeb_samlet",
}

# For the WFS, specify the names of relevant fields for each type of water body
wfs_fields = {
    "catch": ["op_id", "op_navn"],
    "coastal": ["distr_id", "ov_id", "ov_navn", "ov_typ", "til_oko_fy"],
    "lakes": ["distr_id", "ov_id", "ov_navn", "ov_typ", "til_oko_fy"],
    "streams": ["distr_id", "ov_id", "ov_navn", "ov_typ", "til_oko_bb", "na_kun_stm"],
}

########################################################################################
#   3. Import module and run the functions
########################################################################################
# Import the module with all the homemade functions
import script_module

# Initialize the class for all data processing and mapping functions
c = script_module.Water_Quality(
    year_first,
    year_last,
    data,
    linkage,
    wfs_service,
    wfs_fc,
    wfs_fields,
    wfs_replace,
    keep_gdb,
)

# Dictionaries to store DataFrame, shore length, and stats for each category j
frames_j = {}
shores_j = {}
stats_j = {}
stats_MA_j = {}  #  based on a 5-year moving average for each water body to reduce noise

# Loop over each category j ∈ {coastal, lakes, streams}
for j in ("coastal", "lakes", "streams"):
    # Get the feature class from the WFS service
    c.get_fc_from_WFS(j)

    # Create a DataFrame with observed biophysical indicator by year
    df_ind_obs, df_VP = c.observed_indicator(j)

    # Report ecological status based on observed biophysical indicator
    df_eco_obs, obs_stats, index_sorted = c.ecological_status(j, df_ind_obs, df_VP)

    # if j == 'streams':
    #     # Create a map book with yearly maps of observed ecological status
    #     c.map_book(j, df_eco_obs)

    # Impute missing values for biophysical indicator and return ecological status
    df_eco_imp, df_eco_imp_MA, stats_j[j], stats_MA_j[j] = c.impute_missing(
        j, df_eco_obs, df_VP, index_sorted
    )

    # df with variables by coastal catchment area for the Benefit Transfer equation
    frames_j[j], shores_j[j] = c.values_by_catchment_area(j, df_eco_imp_MA, df_VP)

    # Optional: Clean up after each iteration of loop
    if keep_gdb != "true":
        # Delete feature class
        if arcpy.Exists(j):
            arcpy.Delete_management(j)

# Optional: Clean up geodatabase
if keep_gdb != "true":
    # Delete all feature classes in geodatabase
    for fc in arcpy.ListFeatureClasses():
        arcpy.Delete_management(fc)

########################################################################################
#   4.a Stats for all categories j: Shore length and share of it where eco status < Good
########################################################################################
# Set up DataFrame of shore length for each category j ∈ {coastal, lakes, streams}
shores = pd.DataFrame(shores_j)
shores["shores all j"] = shores.sum(axis=1, skipna=True)
shores.to_csv("output\\all_VP_shore length.csv")  #  save to csv
shoresTotal = shores.sum()

# Mean ecological status with and without 5-year moving average (MA) for each water body
for dict, suffix in zip([stats_j, stats_MA_j], ["LessThanGood", "LessThanGood_MA"]):
    # Set up DataFrame of statistics for each category j ∈ {coastal, lakes, streams}
    stats = pd.DataFrame(dict)

    # Plot water bodies by category (mean ecological status weighted by length)
    for format in (".pdf", ".png"):
        f1 = (
            stats.drop(1989)
            .plot(ylabel="Share of category with less than good ecological status")
            .get_figure()
        )
        f1.savefig("output\\all_eco_imp_" + suffix + format, bbox_inches="tight")

    # Calculate mean ecological status across all categories j weighted by shore length
    stats["all j"] = (
        stats["coastal"] * shoresTotal["coastal"]
        + stats["lakes"] * shoresTotal["lakes"]
        + stats["streams"] * shoresTotal["streams"]
    ) / shoresTotal["shores all j"]

    # Save statistics to csv
    stats.to_csv("output\\all_eco_imp_" + suffix + ".csv")


########################################################################################
#   4.b Nominal cost of pollution and investment in water quality for national accounts
########################################################################################
# Concatenate DataFrames for each category j ∈ {coastal, lakes, streams}
df_BT = pd.concat(frames_j)
df_BT.index.names = ["j", "t", "v"]
df_BT.to_csv("output\\all_eco_imp.csv")  #  save to csv

# Marginal willingness to pay (MWTP) for improvement of water quality to "Good"
CWPn_j = c.valuation(df_BT, real=False)

# Investment in water quality (net present value of infinite stream of MWTP for change)
IVn_j = c.valuation(df_BT, real=False, investment=True)

# Merge cost of pollution and investment value of increase (decrease) in water quality
nominal = pd.concat([CWPn_j, IVn_j], axis=1)
nominal.to_excel("output\\all_nominal.xlsx")  # manually Wrap Text row 1 & delete row 3

########################################################################################
#   4.c Real cost of water pollution and investment in water quality for journal article
########################################################################################
# Add adjustment factor due to using unitary income elasticity rather than estimated ε
df_v = c.valuation(df_BT)
df_BT_factor = df_BT.copy()
df_BT_factor["factor"] = df_v["factor"]

df_BT.loc[("coastal", 2018)]
df_BT.loc[("coastal", 2019)]

(CWP_v.loc[("coastal", 2019)] - CWP_v.loc[("coastal", 2018)])


# Costs of Water Pollution (CWP) in real terms (million DKK, 2018 prices)
CWP_v = df_v[["CWP"]]
CWP_j = (
    CWP_v.groupby(["j", "t"]).sum().unstack(level=0).rename_axis(None)  #  sum over v
)
CWP_j.rename_axis([None, None], axis=1).to_csv("output\\all_cost.csv")  #  save CSV
f2 = (
    CWP_j.loc[:, "CWP"]
    .rename_axis(None, axis=1)
    .plot(ylabel="Cost of current water pollution (million DKK, 2018 prices)")
    .get_figure()
)
f2.savefig("output\\all_cost.pdf", bbox_inches="tight")  #  save figure as PDF

# Investment Value of water quality improvement in real terms (million DKK, 2018 prices)
IV_v = c.valuation(df_BT, investment=True)
IV_j = IV_v.groupby(["j", "t"]).sum().unstack(level=0).rename_axis(None)  #  sum over v
IV_j.rename_axis([None, None], axis=1).to_csv("output\\all_investment.csv")  #  save CSV
f3 = (
    IV_j.loc[:, "IV"]
    .rename_axis(None, axis=1)
    .plot(
        kind="bar",
        ylabel="Investment in water quality improvement (million DKK, 2018 prices)",
    )
    .get_figure()
)
f3.savefig("output\\all_investment.pdf", bbox_inches="tight")  #  save figure as PDF

# Overview using real prices and the same declining discount rate for all years
CWP_j.mean()  #  average yearly cost of water pollution
IV_j.mean()  #  average yearly investment value in better (or worse) water quality


########################################################################################
#   5. Decompose development by holding everything else equal at 1990 level
########################################################################################


########################################################################################
#   6. Robustness check: Treat DK as a single catchment area
########################################################################################
