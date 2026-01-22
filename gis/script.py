"""
Name:       script.py

Label:      Construct and map longitudinal data of ecological status of water bodies.

Summary:    ThorNoe.GitHub.io/GreenGDP explains the overall approach and methodology.

Rqmts:      script_module.py in working directory. ArcGIS Pro 3.2 or later installed.

Usage:      This script supports WaterbodiesScriptTool in the gis.tbx toolbox.
            See GitHub.com/ThorNoe/GreenGDP for instructions to run or update it all.

License:    MIT Copyright (c) 2024–2026
Author:     Thor Donsby Noe
"""

########################################################################################
#   0. Imports
########################################################################################
# Import Operation System (os)
import os

# Import ArcPy, NumPy, Pandas, seaborn, and Matplotlib functionality
import arcpy  # requires ArcGIS Pro installed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from cycler import cycler
from matplotlib.ticker import FuncFormatter

########################################################################################
#   1. Setup
########################################################################################
# Set overwrite option
arcpy.env.overwriteOutput = True

# Specify the parent folder as the working directory of the operating system
# path = r"C:\Users\au687527\GitHub\GreenGDP\gis"
# os.chdir(path)
os.chdir(arcpy.GetParameterAsText(0))

# Specify whether to replace existing feature classes downloaded from WFS service
# wfs_replace = 0
wfs_replace = arcpy.GetParameterAsText(1)

# Specify whether to keep the geodatabase when the script finishes
# keep_gdb = 1
keep_gdb = arcpy.GetParameterAsText(2)

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
frames_j, shores_j, stats_obs_j, stats_imp_j, stats_imp_MA_j = {}, {}, {}, {}, {}

# Loop over each category j ∈ {coastal, lakes, streams}
for j in ("coastal", "lakes", "streams"):
    # Get the feature class from the WFS service
    c.get_fc_from_WFS(j)

    # df for observed biophysical indicator and waterbody characteristics respectively
    df_ind_obs, df_VP = c.observed_indicator(j)

    # Report ecological status based on observed biophysical indicator
    df_eco_obs, stats_obs_j[j], index_sorted = c.ecological_status(j, df_ind_obs, df_VP)

    # if j == 'streams':
    #     # Create a map book with yearly maps of observed ecological status
    #     c.map_book(j, df_eco_obs)

    # Impute missing values for biophysical indicator and return ecological status
    df_eco_imp, df_eco_imp_MA, stats_imp_j[j], stats_imp_MA_j[j] = c.impute_missing(
        j, df_eco_obs, df_VP, index_sorted
    )

    # df with variables by coastal catchment area for the Benefit Transfer function
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

# Total shore length of each category j
shoresTotal = shores.sum()

# Dictionary of stats for observed, imputed, and imputed with moving average respectively
stats_method = {
    "obs_LessThanGood": stats_obs_j,
    "imp_LessThanGood": stats_imp_j,
    "imp_LessThanGood_MA": stats_imp_MA_j,
}

for key, dict in stats_method.items():
    # Set up df of share < good status for each category j ∈ {coastal, lakes, streams}
    stats = pd.DataFrame(dict)

    # Plot share of category j with less than good ecological status by year
    for format in (".pdf", ".png"):
        f1 = (
            stats.loc[list(range(year_first + 1, year_last + 1))]
            .plot(ylabel="Share of category with less than good ecological status")
            .get_figure()
        )
        f1.savefig("output\\all_eco_" + key + format, bbox_inches="tight")

    # Calculate share < eco good status across all categories j weighted by shore length
    stats["all j"] = (
        stats["coastal"] * shoresTotal["coastal"]
        + stats["lakes"] * shoresTotal["lakes"]
        + stats["streams"] * shoresTotal["streams"]
    ) / shoresTotal["shores all j"]

    # Add df including "all j" columns to dictionary of stats by method
    stats_method[key] = stats

# Concatenate stats for observed, imputed, and imputed with moving average respectively
dfStats = pd.concat(stats_method, axis=1)
dfStats.to_excel("output\\all_eco_LessThanGood.xlsx")  #  manually delete row 3 in Excel

########################################################################################
#   4.b Set up comprehensive DataFrame with variables for the Benefit Transfer function
########################################################################################
# Concatenate DataFrames for each category j ∈ {coastal, lakes, streams}
df_BT = pd.concat(frames_j)
df_BT.index.names = ["j", "t", "v"]
df_BT.to_csv("output\\all_eco_imp.csv")  #  save to csv

########################################################################################
#   4.c Real cost of water pollution and investment in water quality for journal article
########################################################################################
# Costs of Water Pollution (CWP) in real terms (million DKK, 2023 prices) by t, v, and j
CWP_vj, k = c.valuation(df_BT)  # k is factor for estimated income ε relative to unitary

# Costs of Water Pollution (CWP) in real terms (million DKK, 2023 prices) by t and j
CWP_j = CWP_vj.groupby("t").sum().rename_axis(None).rename_axis(None, axis=1)
CWP_label = "Cost of current water pollution (million DKK, 2023 prices)"
fig = CWP_j.plot(ylabel=CWP_label).get_figure()
fig.savefig("output\\all_cost.pdf", bbox_inches="tight")  #  save figure as PDF
plt.close(fig)  #  close figure to free up memory

# Investment Value of water quality improvement in real terms (million DKK, 2023 prices)
IV_vj = c.valuation(df_BT, investment=True)

# IV of water quality improvement in real terms (million DKK, 2023 prices) by t and j
IV_j = IV_vj.groupby("t").sum().rename_axis(None).rename_axis(None, axis=1)

# Totals using real prices and the same declining discount rate for all years
for a, b in zip([CWP_j, IV_j], ["cost of pollution", "investment in water quality"]):
    a["total"] = a.sum(axis=1)  #  sum of CWP or IV over all categories j (by year)
    print("Average yearly", b, "(million DKK, 2023 prices)\n")
    print(a.mean())  #  average yearly CWP or IV over all years (by category j & total)
CWP_vj["total"] = CWP_vj.sum(axis=1)  #  sum of CWP over all categories j (by year & v)
IV_vj["total"] = IV_vj.sum(axis=1)  #  sum of IV over all categories j (by year & v)

# Save tables as CSV
CWP_j.to_csv("output\\all_cost.csv")  #  cost by t and j
IV_j.to_csv("output\\all_investment.csv")  #  IV by t and j


# Function to format the y-axis ticks with thousands delimiter (for IV plots)
def y_f(x, pos):
    return f"{int(x):,}"


# IV line plot (total) and stacked bar plot (coastal, lakes, and streams)
IV_label = "Investment in water quality improvement (million DKK, 2023 prices)"
fig, ax = plt.subplots()  #  create figure and axis
IV_j.iloc[:, -1].plot(  #  line plot for total IV
    kind="line",
    ax=ax,
    color="black",
    use_index=False,  #  ignore DatetimeIndex units for x-axis
)
IV_j.iloc[:, :-1].plot(kind="bar", stacked=True, ax=ax)  #  stacked bar plot (same axis)
ax.set_ylabel(IV_label)  #  set y-axis label
ax.set_ylim(bottom=-75000)  # set y-axis to begin at -75,000 (to match figures)
ax.yaxis.set_major_formatter(FuncFormatter(y_f))  #  add thousands delimiter to y-axis
ax.legend()  #  add legend for both plots
fig.savefig("output\\all_investment.pdf", bbox_inches="tight")  # save figure as PDF
plt.close(fig)  # close figure to free up memory

########################################################################################
#   5. Decompose development by holding everything else equal at 1990 level
########################################################################################
# Cost of water pollution and investment value by each driver (other things equal)
CWP_driver_v, CWP_driver_j, CWP_driver, IV_driver_v, IV_driver = c.decompose(df_BT, k)

# Catchment area names
Geo = pd.read_excel("data\\" + data["shared"][2], sheet_name="ID_108", index_col=0)
Geo.index.name = "v"
names = Geo.iloc[:108, 0]

# Total real cost of water pollution (CWP) decomposed by driver (other things equal)
for d, df, suffix in zip([CWP_driver, CWP_driver_v], [CWP_j, CWP_vj], ["", "_v"]):
    # Add total CWP using all drivers (i.e., before decomposition)
    d.columns = ["coastal", "lakes", "streams", "income", "age", "households"]
    d["total"] = df["total"].copy()  #  total CWP (before decomposition)
    d = d.rename_axis("driver", axis=1)
    g = ["g (million DKK, 2023 prices)", "g (%)", "g rate (%)"]  #  names for d/df index

    if suffix != "_v":
        # Figure for total CWP decomposed by driver (other things equal at 1990 level)
        ax = CWP_driver.plot(ylabel=CWP_label)
        fig = ax.get_figure()
        fig.savefig("output\\all_cost_decomposed.pdf", bbox_inches="tight")  #  save PDF
        plt.close(fig)  #  close figure to free up memory

        # Growth (both in 2023 prices and in %) and growth rate of total CWP by driver
        d.loc[g[0], :] = d.loc[2020, :] - d.loc[1990, :]
        d.loc[g[1], :] = 100 * d.loc[g[0], :] / d.loc[1990, :]
        d.loc[g[2], :] = 100 * (d.loc[2020, :] / d.loc[1990, :]) ** (1 / 30) - 100
        growth = d.tail(3).T
        growth.columns = ["growth (million DKK)", "growth (\%)", "growth rate (\%)"]
        f = {
            col: "{:0,.0f}".format if col == growth.columns[0] else "{:0.2f}".format
            for col in growth.columns
        }
        col_f = "lrrr"  #  right-aligned column format; match number of columns
        print("Growth in total CWP due to driver (other things equal at 1990 level)")
        print(d.tail(3), "\n")

    d.to_csv("output\\all_cost_decomposed" + suffix + ".csv")  #  save table as CSV

    if suffix == "_v":
        for v in d.index.get_level_values("v").unique():
            d.loc[(g[0], v), :] = d.loc[(2020, v), :] - d.loc[(1990, v), :]
        for v in d.index.get_level_values("v").unique():
            d.loc[(g[1], v), :] = 100 * d.loc[(g[0], v), :] / d.loc[(1990, v), :]
        for v in d.index.get_level_values("v").unique():
            d.loc[(g[2], v), :] = (
                100 * (d.loc[(2020, v), :] / d.loc[(1990, v), :]) ** (1 / 30) - 100
            )
        growth = d[d.index.get_level_values("t") == "g (%)"].describe().drop("count").T
        growth.columns = ["mean", "std", "min", "25\%", "50\%", "75\%", "max"]
        f = {col: "{:0.2f}".format for col in growth.columns}  #  two decimals
        col_f = "lrrrrrrr"  #  right-aligned column format; match number of columns
        d = d.merge(names, left_on="v", right_index=True)  #  merge with catchment names
        d = d.rename_axis("driver", axis=1)
        print("The 15 catchment areas with largest reduction in total CWP (g in %)")
        print(d.loc["g (%)", :].nsmallest(15, ("total")), "\n")
        print("The 15 catchment areas with largest increase in total CWP (g in %)")
        print(d.loc["g (%)", :].nlargest(15, ("total")), "\n")
    with open("output\\all_cost_decomposed_growth" + suffix + ".tex", "w") as tf:
        tf.write(growth.apply(f).to_latex(column_format=col_f))  #  column alignment

# Mean real investment value (IV) decomposed by driver (other things equal at 1990 level)
IV_j_driver = pd.DataFrame()  #  df to store decomposed mean IV for each category j
for d, df, suffix in zip([IV_driver, IV_driver_v], [IV_j, IV_vj], ["", "_v"]):
    # Add IV due to demographics, i.e. residual after water quality improvements
    d["demographics"] = df["total"].copy() - d.sum(axis=1)
    d["total"] = df["total"].copy()  #  Total IV (no decomposition)
    if suffix != "_v":
        # Figure for total IV decomposed by driver (other things equal at 1990 level)
        fig, ax = plt.subplots()
        d.iloc[:, -1].plot(  #  line plot for total IV by driver
            kind="line",
            ax=ax,
            color="black",
            use_index=False,  #  ignore DatetimeIndex units for x-axis
        )
        d.iloc[:, :-1].plot(kind="bar", stacked=True, ax=ax)  # stacked bar plot
        ax.set_ylabel(IV_label)  #  set y-axis label
        ax.set_ylim(bottom=-75000)  # set y-axis to begin at -75,000 (to match figures)
        ax.yaxis.set_major_formatter(FuncFormatter(y_f))  #  y-axis thousands delimiter
        ax.legend()  #  add legend for both plots
        fig.savefig("output\\all_investment_decomposed.pdf", bbox_inches="tight")
        plt.close(fig)  #  close figure to free up memory

        # How many % of total IV is due to demographics?
        for dataFrame in d, df:
            dataFrame.loc["mean IV", :] = dataFrame.mean()  #  mean yearly IV by driver
        d["demographics (%)"] = 100 * d["demographics"] / d["total"]  #  % of total IV
        print("Yearly IV due to demographics (residual after water qual. improvements)")
        print(d.loc["mean IV", "demographics (%)"], "\n")

        # How many % of IV for category j is due to demographics?
        dict_j = {}  #  dictionary to store df of decomposed IV for each category j
        for j in ("coastal", "lakes", "streams"):
            IV_driver_j = pd.DataFrame()
            IV_driver_j["water quality"] = d[j]
            IV_driver_j["demographics"] = df[j] - d[j]
            IV_driver_j["total"] = df[j]
            IV_driver_j["demographics (\%)"] = 100 * (1 - d[j] / df[j])
            dict_j[j] = IV_driver_j
        IV_j_driver = pd.concat(dict_j, axis=1, names=["j", "driver"])
        for col in IV_j_driver.columns:
            if col[1] == "demographics (\%)":
                print(
                    "{0}% of yearly IV due to demographics for {1}\n".format(
                        round(IV_j_driver.loc["mean IV", col]), col[0]
                    )
                )
        # Mean yearly IV for each category j decomposed by water quality and demographics
        mean = IV_j_driver.loc["mean IV", :].unstack(level=0)
        f = {
            "demographics": "{:0.2f}",
            "water quality": "{:0,.0f}",
            "total": "{:0,.0f}",
            "demographics (\%)": "{:0.2f}",
        }
        for row in mean.index:
            if row in f:
                mean.loc[row] = mean.loc[row].map(lambda x: f[row].format(x))
        with open("output\\all_investment_decomposed.tex", "w") as tf:  #  save TeX file
            tf.write(mean.to_latex(column_format="lrrr"))

    d.to_csv("output\\all_investment_decomposed" + suffix + ".csv")  #  save as CSV

    if suffix == "_v":
        # Decomposition of mean yearly IV per household in v (DKK, 2023 prices)
        N = df_BT[
            (df_BT.index.get_level_values("j") == "coastal")
            & (df_BT.index.get_level_values("t") != 1989)
        ]["N"]
        d = 1e6 * d.div(N, axis=0)  #  decomposed IV per household (DKK, 2023 prices)
        df = 1e6 * df.div(N, axis=0)  #  IV per household (DKK, 2023 prices)
        for v in d.index.get_level_values("v").unique():
            d.loc[("mean IV", v), :] = d[d.index.get_level_values("v") == v].mean()
            df.loc[("mean IV", v), :] = df[df.index.get_level_values("v") == v].mean()
        mean = d[d.index.get_level_values("t") == "mean IV"].describe().drop("count").T
        mean.columns = ["mean", "std", "min", "25\%", "50\%", "75\%", "max"]
        f = {col: "{:0,.0f}".format for col in mean.columns}
        with open("output\\all_investment_decomposed_v.tex", "w") as tf:
            tf.write(mean.apply(f).to_latex(column_format="lrrrrrrr"))  #  col alignment
        d["demographics (%)"] = 100 * d["demographics"] / d["total"]  #  % of total IV
        d = d.merge(names, left_on="v", right_index=True)  #  catchment area names
        d = d.rename_axis("driver", axis=1)
        print("The 15 catchment areas with lowest mean total IV per household")
        print(d.loc["mean IV", :].nsmallest(15, ("total")), "\n")
        print("The 15 catchment areas with highest mean total IV per household")
        print(d.loc["mean IV", :].nlargest(15, ("total")), "\n")
        for j in ("coastal", "lakes", "streams"):
            m = pd.DataFrame()  #  df to store decomposed mean IV for each category j
            m["water quality"] = d.loc["mean IV", j]
            m["demographics"] = df.loc["mean IV", j] - m["water quality"]
            m["total for j"] = df.loc["mean IV", j]
            m["demographics (%)"] = 100 * (1 - m["water quality"] / m["total for j"])
            m["Catchment area name"] = names
            print("The 10 catchment areas with lowest mean IV per household for", j)
            print(m.nsmallest(10, ("total for j")), "\n")
            print("The 10 catchment areas with highest mean IV per household for", j)
            print(m.nlargest(10, ("total for j")), "\n")

# Colors and line styles by category j - matching those used for total CWP decomposition
cl1 = ["#4477AA", "#CCBB44", "#EE6677", "#AA3377", "black"]  #  5 colors for coastal
cl2 = ["#66CCEE", "#CCBB44", "#EE6677", "#AA3377", "black"]  #  5 colors for lakes
cl3 = ["#228833", "#CCBB44", "#EE6677", "#AA3377", "black"]  #  5 colors for streams
lc1 = ["-", "-", "--", ":", "-."]  #  5 line styles for coastal
lc2 = ["--", "-", "--", ":", "-."]  #  5 line styles for lakes
lc3 = [":", "-", "--", ":", "-."]  #  5 line styles for streams
categories = {
    "coastal": (cl1, ["#4477AA", "#CCBB44"], lc1),
    "lakes": (cl2, ["#66CCEE", "#CCBB44"], lc2),
    "streams": (cl3, ["#228833", "#CCBB44"], lc3),
}

# Real cost of water pollution of category j decomposed by driver (other things equal)
for j, (c1, c2, ls) in categories.items():
    # CWP by category j - matching the colors and line styles used for total CWP
    d = CWP_driver_j.loc[:, CWP_driver_j.columns.get_level_values(1) == j].copy()
    d.columns = [j, "income", "age", "households"]
    d["total"] = CWP_j.loc[:, j]
    d = d.rename_axis("driver", axis=1)
    fig, ax = plt.subplots()  # create a new figure and axes
    ax.set_prop_cycle(cycler(color=c1) + cycler(linestyle=ls))  # set the property cycle
    ax = d.plot(ax=ax, ylabel=CWP_label)
    fig.savefig(f"output\\{j}_cost_decomposed.pdf", bbox_inches="tight")
    plt.close(fig)  # close figure to free up memory

    # Figure: IV by category j - matching the colors used for total IV
    df = IV_j_driver.loc[:, IV_j_driver.columns.get_level_values(0) == j].copy()
    df = df.iloc[:-1, :2][j]  # drop "mean IV", "total", and "demographics (\%)"
    fig, ax = plt.subplots()  # create a new figure and axes
    df.plot(kind="bar", stacked=True, ax=ax, color=c2, ylabel=IV_label)  # stacked bars
    ax.yaxis.set_major_formatter(FuncFormatter(y_f))  #  y-axis thousands delimiter
    fig.savefig(f"output\\{j}_investment_decomposed.pdf", bbox_inches="tight")
    plt.close(fig)  # close figure to free up memory

    # Calculate growth (in 2023 prices and in %) and growth rate of CWP of j by driver
    d.loc["g (million DKK, 2023 prices)", :] = d.loc[2020, :] - d.loc[1990, :]
    d.loc["g (%)", :] = 100 * d.loc["g (million DKK, 2023 prices)", :] / d.loc[1990, :]
    d.loc["g yearly (%)", :] = 100 * (d.loc[2020, :] / d.loc[1990, :]) ** (1 / 30) - 100
    d.to_csv("output\\" + j + "_cost_decomposed.csv")  #  save table as CSV
    print("Growth in CWP of", j, "due to driver (other things equal at 1990 level)")
    print(d.tail(3), "\n")

########################################################################################
#   6. Descriptive statistics for geographical variables; Box plot demographics
########################################################################################
# Geographical data (assumed time-invariant)
SL = pd.read_excel(  #  total shore length by category j (regardless of water quality)
    "data\\" + data["shared"][2], sheet_name="all_VP_shore length", index_col=0
)
SL.columns = [
    "Coastline",
    "Shore length of lakes in VP3",
    "Shore length of streams in VP3",
    "Shore length of all water bodies in VP3",
]
PAL = pd.read_excel(
    "data\\" + data["shared"][2], index_col=0
)  #  proportion arable land
PAL["Proportion of arable land"] = np.exp(PAL["ln PAL"])

# Descriptive statistics for geographical variables
Geo = SL.merge(PAL[["Proportion of arable land"]], on="v").describe().T
Geo.rename(
    columns={"count": "n", "25%": "25\%", "50%": "50\%", "75%": "75\%"}, inplace=True
)
f = {col: "{:0,.0f}".format if col == "n" else "{:0.2f}".format for col in Geo.columns}
col_f = "lrrrrrrrr"  #  right-aligned column format; match number of columns
with open("output\\all_VP_shore length_stats.tex", "w") as tf:
    tf.write(Geo.apply(f).to_latex(column_format=col_f))  #  column alignment

# Demographics by catchment area v and year t (using extrapolated data for 2019-2020)
Dem = pd.read_csv("output\\all_demographics.csv")  #  reset index (no v & t multiindex)

# Box plot for number of households
Dem["N"] = Dem["N"] / 100000  #  convert number of households to 100,000
plt.figure(figsize=(10, 12))
sns.boxplot(x="t", y="N", data=Dem, palette=["#AA3377"])
# plt.title("Distribution of number of households over catchment areas")
plt.xlabel("")  #  omit x-axis label
plt.ylabel("Number of households (100,000)")
plt.ylim(bottom=0)  #  fix the bottom of the y-axis to start at exactly 0
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("output\\all_demographics_N.pdf", bbox_inches="tight")
plt.close()  #  close plot to free up memory

# Box plot for mean real income
plt.figure()
sns.boxplot(x="t", y="y", data=Dem, palette=["#CCBB44"])
# plt.title("Distribution of mean real household income over catchment areas")
plt.xlabel("")  #  omit x-axis label
plt.ylabel("Mean real household income (100,000 DKK, 2018 prices)")
plt.axhline(y=5, color="black", linestyle=":")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("output\\all_demographics_y.pdf", bbox_inches="tight")
plt.close()  #  close plot to free up memory

# Box plot for mean age
plt.figure()
sns.boxplot(x="t", y="age", data=Dem, palette=["#EE6677"])
# plt.title("Distribution of mean age over catchment areas")
plt.xlabel("")  #  omit x-axis label
plt.ylabel("Mean age")
plt.yticks([35, 40, 45, 50, 55])  # set specific tick positions on y-axis
plt.axhline(y=45, color="black", linestyle=":")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("output\\all_demographics_age.pdf", bbox_inches="tight")
plt.close()  #  close plot to free up memory
n = Dem.groupby("t")["D age"].sum()  #  number of catchment areas w. mean age > 45 years
pd.DataFrame({"n": n, "s (%)": 100 * n / 108})  #  number and share: mean age > 45 years
