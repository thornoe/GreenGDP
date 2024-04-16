"""
Name:       script_line-by-line.py

Label:      Construct and map longitudinal data of ecological status of water bodies.

Summary:    ThorNoe.GitHub.io/GreenGDP explains the overall approach and methodology.

Rqmts:      ArcGIS Pro must be installed on the system and be up to date.

Usage:      This sandbox is line-by-line implementation of the script supporting 
            WaterbodiesScriptTool in the gis.tbx toolbox.
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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from cycler import cycler
from scipy import interpolate
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer

# Color-blind-friendly color scheme for qualitative data by Tol: personal.sron.nl/~pault
colors = {
    "blue": "#4477AA",
    "cyan": "#66CCEE",
    "green": "#228833",
    "yellow": "#CCBB44",
    "red": "#EE6677",
    "purple": "#AA3377",
    "grey": "#BBBBBB",
}

# Set the default property-cycle and figure size for pyplots
color_cycler = cycler(color=list(colors.values()))  #  color cycler with 7 colors
linestyle_cycler = cycler(linestyle=["-", "--", ":", "-", "--", ":", "-."])  #  7 styles
plt.rc("axes", prop_cycle=(color_cycler + linestyle_cycler))
plt.rc("figure", figsize=[10, 6.2])  #  golden ratio

########################################################################################
#   1. Setup
########################################################################################
# Set overwrite option
arcpy.env.overwriteOutput = True

# Specify the parent folder as the working directory of the operating system
root = r"C:\Users\au687527\GitHub\GreenGDP"
path = root + "\\gis"
os.chdir(path)
arcPath = path + "\\gis.gdb"
arcpy.env.workspace = arcPath

# Specify whether to replace existing feature classes downloaded from WFS service
wfs_replace = 0

# Specify whether to keep the geodatabase when the script finishes
keep_gdb = 1

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

# Specify a single category
# j = "coastal"
# j = "lakes"
j = "streams"

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
    # df_ind_obs, df_VP = c.observed_indicator(j)
    # df_ind_obs = pd.read_csv("output\\" + j + "_ind_obs.csv", index_col="wb")
    # df_ind_obs.columns = df_ind_obs.columns.astype(int)
    df_VP = pd.read_csv("output\\" + j + "_VP.csv", index_col="wb")

    # Report ecological status based on observed biophysical indicator
    # df_eco_obs, stats_obs_j[j], index_sorted = c.ecological_status(j, df_ind_obs, df_VP)

    # if j == 'streams':
    #     # Create a map book with yearly maps of observed ecological status
    #     c.map_book(j, df_eco_obs)

    # Impute missing values for biophysical indicator and return ecological status
    # df_eco_imp, df_eco_imp_MA, stats_imp_j[j], stats_imp_MA_j[j] = c.impute_missing(
    #     j, df_eco_obs, df_VP, index_sorted
    # )
    df_eco_imp_MA = pd.read_csv("output\\" + j + "_eco_imp_MA.csv", index_col="wb")
    df_eco_imp_MA.columns = df_eco_imp_MA.columns.astype(int)

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
# shores = pd.DataFrame(shores_j)
# shores["shores all j"] = shores.sum(axis=1, skipna=True)
# shores.to_csv("output\\all_VP_shore length.csv")  #  skip if reading it instead
shores = pd.read_csv("output\\all_VP_shore length.csv", index_col=0)

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
        fig = (
            stats[list(range(year_first + 1, year_last + 1))]
            .plot(ylabel="Share of category with less than good ecological status")
            .get_figure()
        )
        fig.savefig("output\\all_eco_" + key + format, bbox_inches="tight")
        plt.close(fig)  #  close figure to free up memory

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
#   4.b Nominal cost of pollution and investment in water quality for national accounts
########################################################################################
# Concatenate DataFrames for each category j ∈ {coastal, lakes, streams}
# df_BT = pd.concat(frames_j)
# df_BT.index.names = ["j", "t", "v"]
# df_BT.to_csv("output\\all_eco_imp.csv")  #  skip if reading it instead
df_BT = pd.read_csv("output\\all_eco_imp.csv", index_col=[0, 1, 2])

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
# Costs of Water Pollution (CWP) in real terms (million DKK, 2018 prices) by t, v, and j
CWP_vj = c.valuation(df_BT)

# Costs of Water Pollution (CWP) in real terms (million DKK, 2018 prices) by t and j
CWP_j = CWP_vj.groupby("t").sum().rename_axis(None).rename_axis(None, axis=1)
CWP_j.to_csv("output\\all_cost.csv")  #  save table as CSV
CWP_label = "Cost of current water pollution (million DKK, 2018 prices)"
fig = CWP_j.plot(ylabel=CWP_label).get_figure()
fig.savefig("output\\all_cost.pdf", bbox_inches="tight")  #  save figure as PDF
plt.close(fig)  #  close figure to free up memory

# Investment Value of water quality improvement in real terms (million DKK, 2018 prices)
IV_vj = c.valuation(df_BT, investment=True)

# IV of water quality improvement in real terms (million DKK, 2018 prices) by t and j
IV_j = IV_vj.groupby("t").sum().rename_axis(None).rename_axis(None, axis=1)
IV_j.to_csv("output\\all_investment.csv")  #  save table as CSV
IV_label = "Investment in water quality improvement (million DKK, 2018 prices)"
fig = IV_j.plot(
    kind="bar",
    ylabel=IV_label,
).get_figure()
fig.savefig("output\\all_investment.pdf", bbox_inches="tight")  #  save figure as PDF
plt.close(fig)  #  close figure to free up memory

# Overview using real prices and the same declining discount rate for all years
for a, b in zip([CWP_j, IV_j], ["cost of pollution", "investment in water quality"]):
    a["total"] = a.sum(axis=1)  #  sum of CWP or IV over all categories j (by year)
    print("Average yearly", b, "(million DKK, 2018 prices)\n")
    print(a.mean())  #  average yearly CWP or IV over all years (by category j & total)

########################################################################################
#   5. Decompose development by holding everything else equal at 2018 level
########################################################################################
# Cost of water pollution and investment value by each driver (other things equal)
CWP_driver_v, CWP_driver_j, CWP_driver, IV_driver_v, IV_driver = c.decompose(df_BT)

# Total cost of water pollution (CWP) decomposed by driver (other things equal)
for d, df, suffix in zip([CWP_driver, CWP_driver_v], [CWP_j, CWP_vj], ["", "_v"]):
    # Add total CWP using all drivers (i.e., before decomposition)
    d.columns = ["coastal", "lakes", "streams", "income", "age", "households"]
    d["all"] = df.sum(axis=1)
    d = d.rename_axis("driver", axis=1)

    if suffix != "_v":
        # Figure for total CWP decomposed by driver (other things equal at 2018 level)
        ax = CWP_driver.plot(ylabel=CWP_label)
        ax.axhline(CWP_driver.loc[2018, "all"], linestyle=":", linewidth=0.5)
        fig = ax.get_figure()
        fig.savefig("output\\all_cost_decomposed.pdf", bbox_inches="tight")  #  save PDF
        plt.close(fig)  #  close figure to free up memory

    # Growth (both in 2018 prices and in %) and growth rate of total CWP by driver
    g = ["g (million DKK, 2018 prices)", "g (%)", "g rate (%)"]
    if suffix != "_v":
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
        print("Growth in total CWP due to driver (other things equal at 2018 level)")
        print(d.tail(3), "\n")
    else:
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
        print("The 15 catchment areas with largest reduction in total CWP")
        print(d.loc["g (%)", :].nsmallest(15, ("all")), "\n")
        print("The 15 catchment areas with largest increase in total CWP")
        print(d.loc["g (%)", :].nlargest(15, ("all")), "\n")
    with open("output\\all_cost_decomposed_growth" + suffix + ".tex", "w") as tf:
        tf.write(growth.apply(f).to_latex(column_format=col_f))  #  column alignment
    d.to_csv("output\\all_cost_decomposed" + suffix + ".csv")  #  save table as CSV

# Colors and line styles by category j - matching those used for total CWP decomposition
cl1 = ["#4477AA", "#CCBB44", "#EE6677", "#AA3377", "#BBBBBB"]  #  5 colors for coastal
cl2 = ["#66CCEE", "#CCBB44", "#EE6677", "#AA3377", "#BBBBBB"]  #  5 colors for lakes
cl3 = ["#228833", "#CCBB44", "#EE6677", "#AA3377", "#BBBBBB"]  #  5 colors for streams
lc1 = cycler(linestyle=["-", "-", "--", ":", "-."])  #  5 line styles for coastal
lc2 = cycler(linestyle=["--", "-", "--", ":", "-."])  #  5 line styles for lakes
lc3 = cycler(linestyle=[":", "-", "--", ":", "-."])  #  5 line styles for streams

# Cost of water pollution of category j decomposed by driver (other things equal)
for j, cl, ls in zip(["coastal", "lakes", "streams"], [cl1, cl2, cl3], [lc1, lc2, lc3]):
    d = CWP_driver_j.loc[:, CWP_driver_j.columns.get_level_values(1) == j].copy()
    d.columns = [j, "income", "age", "households"]
    d["all"] = CWP_j.loc[:, j]
    d = d.rename_axis("driver", axis=1)

    # Figure by category j - matching the colors and line styles used for total CWP
    fig, ax = plt.subplots()  #  create a new figure and axes
    ax.set_prop_cycle(cycler(color=cl) + ls)  #  set the property cycle
    ax = d.plot(ax=ax, ylabel=CWP_label)
    ax.axhline(d.loc[2018, "all"], color="black", linestyle=":", linewidth=0.5)
    fig.savefig("output\\" + j + "_cost_decomposed.pdf", bbox_inches="tight")
    plt.close(fig)  #  close figure to free up memory

    # Calculate growth (in 2018 prices and in %) and growth rate of CWP of j by driver
    d.loc["g (million DKK, 2018 prices)", :] = d.loc[2020, :] - d.loc[1990, :]
    d.loc["g (%)", :] = 100 * d.loc["g (million DKK, 2018 prices)", :] / d.loc[1990, :]
    d.loc["g yearly (%)", :] = 100 * (d.loc[2020, :] / d.loc[1990, :]) ** (1 / 30) - 100
    d.to_csv("output\\" + j + "_cost_decomposed.csv")  #  save table as CSV
    print("Growth in CWP of", j, "due to driver (other things equal at 2018 level)")
    print(d.tail(3))

# IV of water quality improvement decomposed by driver, other things equal at 2018 level
fig = IV_driver.plot(kind="bar", ylabel=IV_label).get_figure()
fig.savefig("output\\all_investment_decomposed.pdf", bbox_inches="tight")  #  save PDF
plt.close(fig)  #  close figure to free up memory

# Calculate mean real investment value by driver and total for all categories j
for d, df, suffix in zip([IV_driver, IV_driver_v], [IV_j, IV_vj], ["", "_v"]):
    d["all"] = df.sum(axis=1)  #  total IV, using demographics by year
    if suffix != "_v":
        d.loc["mean IV", :] = d.mean()
    else:
        # Calculate mean IV per household in v (DKK, 2018 prices)
        N = df_BT[
            (df_BT.index.get_level_values("j") == "coastal")
            & (df_BT.index.get_level_values("t") != 1989)
        ]["N"]
        d = 1e6 * d.div(N, axis=0)  #  IV per household (DKK, 2018 prices)
        for v in d.index.get_level_values("v").unique():
            d.loc[("mean IV", v), :] = d[d.index.get_level_values("v") == v].mean()
        mean = d[d.index.get_level_values("t") == "mean IV"].describe().drop("count").T
        mean.columns = ["mean", "std", "min", "25\%", "50\%", "75\%", "max"]
        f = {col: "{:0,.0f}".format for col in mean.columns}
        with open("output\\all_investment_decomposed_v.tex", "w") as tf:
            tf.write(mean.apply(f).to_latex(column_format="lrrrrrrr"))  #  col alignment
        print("The 15 catchment areas with lowest total IV per household")
        print(d.loc["mean IV", :].nsmallest(15, ("all")), "\n")
        print("The 15 catchment areas with highest total IV per household")
        print(d.loc["mean IV", :].nlargest(15, ("all")), "\n")
    d.to_csv("output\\all_investment_decomposed" + suffix + ".csv")  #  save as CSV

########################################################################################
#   6. Robustness check: Treat DK as a single catchment area
########################################################################################


### DUMMY FOR AGE>45 BY CATCHMENT AREA

# Average share of catchment areas with mean age > 45 years weighted by no. persons
df = pd.DataFrame(
    df.groupby(["aar"]).apply(lambda x: np.average(x["D_age"], weights=x["antal_pers"]))
)

# Extrapolate using a linear trend
df = df.append(
    pd.DataFrame([np.nan, np.nan, np.nan], index=[1989, 2019, 2020])
).sort_index()
kwargs = dict(method="index", fill_value="extrapolate", limit_direction="both")
df.interpolate(**kwargs, inplace=True)

df.to_csv("output\\" + "D_age.csv")

########################################################################################
#   7. Sandbox: Run the functions line-by-line
########################################################################################
arcpy.ListFeatureClasses()
for fc in arcpy.ListFeatureClasses():
    fc
    for field in arcpy.ListFields(fc):
        field.name, field.type, field.length
for fc in arcpy.ListFeatureClasses():
    arcpy.Delete_management(fc)
arcpy.Exists(fcStations)
for field in arcpy.ListFields(j):
    field.name, field.type, field.length
arcpy.Delete_management(j)


# def observed_indicator(self, j, radius=15):
"""Set up a longitudinal DataFrame for all water bodies of category j by year t.
Assign monitoring stations to water bodies in water body plan via linkage table.
For monitoring stations not included in the linkage table: Assign a station to a waterbody if the station's coordinates are located within said waterbody. For streams, if the station is within a radius of 15 meters of a stream where the name of the stream matches the location name attached to the monitoring station).
Finally, construct the longitudinal DataFrame of observed biophysical indicator by year for all water bodies in the current water body plan. Separately, save the water body ID, typology, district ID, and shore length of each water body in VP3 using the feature classes collected via the get_fc_from_WFS() function."""
radius = 0
if j == "streams":
    # Create longitudinal df for stations in streams by monitoring version
    kwargs = dict(
        f=c.data[j][1],
        d="Dato",
        x="Xutm_Euref89_Zone32",
        y="Yutm_Euref89_Zone32",
        valueCol="Indeks",
        parameterCol="Indekstype",
    )
    DVFI_F = c.longitudinal(j, parameter="Faunaklasse, felt", **kwargs)
    DVFI_M = c.longitudinal(j, parameter="DVFI, MIB", **kwargs)
    DVFI = c.longitudinal(j, parameter="DVFI", **kwargs)
    # Observations after 2020 (publiced after ODA database update Jan 2024)
    DVFI2 = c.longitudinal(
        j,
        f=c.data[j][0],
        d="Dato",
        x="Målested X-UTM",
        y="Målested Y-UTM)",
        valueCol="Indeks",
    )
    # Obtain some of the missing coordinates
    stations = pd.read_csv("linkage\\" + c.linkage[j][1]).astype(int)
    stations.columns = ["station", "x", "y"]
    stations.set_index("station", inplace=True)
    DVFI2[["x", "y"]] = DVFI2[["x", "y"]].combine_first(stations)
    # Group by station; keep last non-missing entry each year, DVFI>MIB>felt
    long = pd.concat([DVFI_F, DVFI_M, DVFI, DVFI2]).groupby("station").last()
else:
    # Create longitudinal df for stations in lakes and coastal waters
    long = c.longitudinal(
        j,
        f=c.data[j][0],
        d="Startdato",
        x="X_UTM32",
        y="Y_UTM32",
        valueCol="Resultat",
    )
    if j == "lakes":
        # Obtain the few missing coordinates
        stations = pd.read_csv("linkage\\" + c.linkage[j][1]).astype(int)
        stations.columns = ["station", "x", "y"]
        stations.set_index("station", inplace=True)
        long[["x", "y"]] = long[["x", "y"]].combine_first(stations)
# Read the linkage table
dfLinkage = pd.read_csv("linkage\\" + c.linkage[j][0])
# Convert station ID to integers
dfLinkage = dfLinkage.copy()  #  to avoid SettingWithCopyWarning
dfLinkage.loc[:, "station"] = dfLinkage["station_id"].str.slice(7).astype(int)
# Merge longitudinal DataFrame with linkage table for water bodies in VP3
df = long.merge(dfLinkage[["station", "ov_id"]], how="left", on="station")
# Stations covered by the linkage tabel for the third water body plan VP3
link = df.dropna(subset=["ov_id"])
# Convert water body ID (wb) to integers
link = link.copy()  #  to avoid SettingWithCopyWarning
if j == "lakes":
    link.loc[:, "wb"] = link["ov_id"].str.slice(6).astype(int)
else:
    link.loc[:, "wb"] = link["ov_id"].str.slice(7).astype(int)
# Stations not covered by the linkage table for VP3
noLink = df[df["ov_id"].isna()].drop(columns=["ov_id"])
# Create a spatial reference object with same geographical coordinate system
spatialRef = arcpy.SpatialReference("ETRS 1989 UTM Zone 32N")
# Specify name of feature class for stations (points)
fcStations = j + "_stations"
# Create new feature class shapefile (will overwrite if it already exists)
arcpy.CreateFeatureclass_management(
    c.arcPath, fcStations, "POINT", spatial_reference=spatialRef
)
# (...)


# def longitudinal(self, j, f, d, x, y, valueCol, parameterCol=0, parameter=0):
"""Set up a longitudinal DataFrame for all stations in category j by year t.
Streams: For a given year, find the DVFI index value of bottom fauna for a station with multiple observations by taking the median and rounding down
Lakes and coastal waters: For a given year, estimate the chlorophyll summer average for every station monitored at least four times during May-September by linear interpolating of daily data from May 1 to September 30 (or extrapolate by inserting the first/last observation from May/September if there exist no observations outside of said period that are no more than 6 weeks away from the first/last observation in May/September)."""
f = c.data[j][0]
d = "Startdato"
x = "X_UTM32"
y = "Y_UTM32"
valueCol = "Resultat"
parameterCol = 0
parameter = 0
# Read the data for biophysical indicator (source: ODAforalle.au.dk)
df = pd.read_excel("data\\" + f)
# Rename the station ID column and make it the index of df
df = df.set_index("ObservationsStedNr").rename_axis("station")
# Create 'Year' column from the date column
df = df.copy()  #  to avoid SettingWithCopyWarning
df.loc[:, "year"] = df[d].astype(str).str.slice(0, 4).astype(int)
if parameterCol != 0:
    # Subset the data to only contain the relevant parameter
    df = df[df[parameterCol] == parameter]
# Drop missing values and sort by year
df = df.dropna(subset=valueCol).sort_values("year")
# Column names for the final longitudinal DataFrame besides the indicator
cols = ["x", "y"]
if j == "streams":
    cols.append("location")  #  add location name for final DataFrame
    df = df[[x, y, "Lokalitetsnavn", "year", valueCol]]  #  subset columns
    df.columns = cols + ["year", "ind"]  #  shorten column names
    df["location"] = df["location"].str.upper()  # capitalize location names
    df = df[df["ind"] != "U"]  #  drop obs with unknown indicator value "U"
    df["ind"] = df["ind"].astype(int)  #  convert indicator to integer
else:  #  Lakes and coastal waters
    # Convert date column to datetime format
    df[d] = pd.to_datetime(df[d].astype(str), format="%Y%m%d")  #  convert
    df = df[[x, y, d, "year", valueCol]]  #  subset to relevant columns
    df.columns = cols + ["date", "year", "ind"]  #  shorten column names
    df.set_index("date", append=True, inplace=True)  #  add 'date' to ind
# Replace 0-values with missing in 'x' and 'y' columns
df[["x", "y"]] = df[["x", "y"]].replace(0, np.nan)
# Set up a longitudinal df with every station and its last non-null entry
long = df[cols].groupby(level="station").last()

# For each year t, add a column with observations for the indicator
for t in df["year"].unique():
    # Subset to year t
    dft = df[df["year"] == t]
    # Subset to station and indicator columns only
    dft = dft[["ind"]]
    if j == "streams":
        # Group multiple obs for a station: Take the median and round down
        dfYear = dft.groupby("station").median().apply(np.floor).astype(int)
        # Rename the indicator column to year t
        dfYear.columns = [t]
    else:
        # Generate date range 6 weeks before and after May 1 and September 30
        dates = pd.date_range(str(t) + "-03-20", str(t) + "-11-11")
        summer = pd.date_range(str(t) + "-05-01", str(t) + "-09-30")
        # Subset to dates in the date range
        dft = dft.loc[
            (dft.index.get_level_values("date") >= dates.min())
            & (dft.index.get_level_values("date") <= dates.max())
        ]
        # Take the mean of multiple obs for any station-date combination
        dft = dft.groupby(level=["station", "date"]).mean()
        # Pivot dft with dates as index and stations as columns
        dft = dft.reset_index().pivot(index="date", columns="station", values="ind")
        # Subset dft to rows that are within the summer date range
        dftSummer = dft.loc[dft.index.isin(summer), :]
        # Drop columns (stations) with less than 4 values
        dftSummer = dftSummer.dropna(axis=1, thresh=4)
        # Create empty DataFrame with dates as index and stations as columns
        dfd = pd.DataFrame(index=dates, columns=dftSummer.columns)
        # Update the empty dfd with the chlorophyll observations in dft
        dfd.update(dft)
        # Convert to numeric, errors='coerce' will set non-numeric values to NaN
        dfd = dfd.apply(pd.to_numeric, errors="coerce")
        # Linear Interpolation of missing values with consecutive gap < 6 weeks
        dfd = dfd.interpolate(limit=41, limit_direction="both")
        # Linear Interpolation for May-September without limit
        dfd = dfd.loc[dfd.index.isin(summer), :].interpolate(limit_direction="both")
        # Drop any column that might somehow still contain missing values
        dfd = dfd.dropna(axis=1)
        # Take the summer average of chlorophyll for each station in year t
        dfYear = dfd.groupby(dfd.index.year).mean().T
    # Merge into longitudinal df
    long = long.merge(dfYear, how="left", on="station")


# def impute_missing(self, dfEcoObs, dfVP, index):
"""Impute ecological status for all water bodies from the observed indicator."""
# DataFrames for observed biophysical indicator and typology
dfEcoObs, dfVP, index, stats_imp_j = df_eco_obs, df_VP, index_sorted, {}

# Merge observed ecological status each year with basis analysis for VP3
dfEco = dfEcoObs.merge(dfVP[["Basis"]], on="wb")

if j == "streams":
    # Create dummies for typology
    typ = pd.get_dummies(dfVP["ov_typ"]).astype(int)
    typ["Soft bottom"] = typ["RW4"] + typ["RW5"]
    typ.columns = [
        "Small",
        "Medium",
        "Large",
        "Small w. soft bottom",
        "Medium w. soft bottom",
        "Soft bottom",
    ]
    # Dummies for natural, artificial, and heavily modified water bodies
    natural = pd.get_dummies(dfVP["na_kun_stm"]).astype(int)
    natural.columns = ["Artificial", "Natural", "Heavily modified"]
    # Merge DataFrames for typology and natural water bodies
    typ = typ.merge(natural, on="wb")
    # Dummies used for imputation chosen via Forward Stepwise Selection (CV)
    cols = ["Soft bottom", "Natural", "Large"]
elif j == "lakes":
    # Convert typology to integers
    typ = dfVP[["ov_typ"]].copy()
    typ.loc[:, "type"] = typ["ov_typ"].str.slice(6).astype(int)
    # Create dummies for high alkalinity, brown, saline, and deep lakes
    cond1 = [(typ["type"] >= 9) & (typ["type"] <= 16), typ["type"] == 17]
    typ["Alkalinity"] = np.select(cond1, [1, np.nan], default=0)
    cond2 = [
        typ["type"].isin([5, 6, 7, 8, 13, 14, 15, 16]),
        typ["type"] == 17,
    ]
    typ["Brown"] = np.select(cond2, [1, np.nan], default=0)
    cond3 = [
        typ["type"].isin([2, 3, 7, 8, 11, 12, 15, 16]),
        typ["type"] == 17,
    ]
    typ["Saline"] = np.select(cond3, [1, np.nan], default=0)
    cond4 = [typ["type"].isin(np.arange(2, 17, 2)), typ["type"] == 17]
    typ["Deep"] = np.select(cond4, [1, np.nan], default=0)
    # Dummies used for imputation chosen via Forward Stepwise Selection (CV)
    cols = ["Saline", "Brown", "Alkalinity"]
else:  #  coastal waters
    # Get typology
    typ = dfVP[["ov_typ"]].copy()
    # Define the dictionaries
    dict1 = {
        "No": "North Sea",  # Nordsø
        "K": "Kattegat",  # Kattegat
        "B": "Belt Sea",  # Bælthav
        "Ø": "Baltic Sea",  # Østersøen
        "Fj": "Fjord",  # Fjord
        "Vf": "North Sea fjord",  # Vesterhavsfjord
    }
    dict2 = {
        "Vu": "Water exchange",  # vandudveksling
        "F": "Freshwater inflow",  # ferskvandspåvirkning
        "D": "Deep",  # vanddybde
        "L": "Stratified",  # lagdeling
        "Se": "Sediment",  # sediment
        "Sa": "Saline",  # salinitet
        "T": "Tide",  # tidevand
    }

    # Define a function to process each string
    def process_string(s):
        # Drop the hyphen and everything following it
        s = s.split("-")[0]
        # Create a en empty dictionary for relevant abbreviations as keys
        dummies = {}
        # Check for abbreviations from dict1 first
        for abbr in dict1:
            if abbr in s:
                dummies[abbr] = 1
                s = s.replace(
                    abbr, ""
                )  # Remove the matched abbreviation from the string
        # Then check for abbreviations from dict2
        for abbr in dict2:
            if abbr in s:
                dummies[abbr] = 1
        return dummies

    # Apply the function to typ["ov_typ"] to create a df with the dummies
    typ = typ["ov_typ"].apply(process_string).apply(pd.Series)
    # Replace NaN values with 0
    typ = typ.fillna(0).astype(int)
    # Rename the dummies from abbreviations to full names
    dicts = {**dict1, **dict2}  #  combine the dictionaries
    typ = typ.rename(columns=dicts)  #  rename columns to full names
    # Dummies used for imputation chosen via Forward Stepwise Selection (CV)
    cols = [
        "North Sea",
        "Kattegat",
        "Belt Sea",
        "Baltic Sea",
        "Fjord",
        "North Sea fjord",
        "Water exchange",
        "Sediment",
    ]

# Merge DataFrame for observed values with DataFrame for dummies
dfEcoSelected = dfEco.merge(typ[cols], on="wb")  #  with selected predictors

# Multivariate imputer using BayesianRidge estimator w. increased tolerance
imputer = IterativeImputer(tol=1e-1, max_iter=100, random_state=0)

# Fit imputer, transform data iteratively, and limit to years of interest
dfImp = pd.DataFrame(
    imputer.fit_transform(np.array(dfEcoSelected)),
    index=dfEcoSelected.index,
    columns=dfEcoSelected.columns,
)[dfEcoObs.columns]

# Calculate a 5-year moving average (MA) for each water body to reduce noise
dfImpMA = dfImp.T.rolling(window=5, min_periods=3, center=True).mean().T

# Convert the imputed ecological status to categorical scale {0, 1, 2, 3, 4}
impStats = c.ecological_status(j, dfImp, dfVP, "imp", index)

# Convert moving average of the imputed eco status to categorical scale
impStatsMA = c.ecological_status(j, dfImpMA, dfVP, "imp_MA", index)

df_eco_imp, df_eco_imp_MA = dfImp[c.years], dfImpMA[c.years]
stats_imp_j[j], stats_imp_MA_j[j] = impStats, impStatsMA
# return dfImp[c.years], dfImpMA[c.years], impStats, impStatsMA


# def ecological_status(self, j, dfIndicator, dfTyp, suffix="obs", index=None):
"""Call indicator_to_status() to convert the longitudinal DataFrame to the EU index of ecological status, i.e., from 0-4 for Bad, Poor, Moderate, Good, and High water quality based on the category and typology of each water body.
Also call missing_values_graph() to map missing observations by year.
Create a table of statistics and export it as an html table.
Print the shore length and share of water bodies observed at least once."""
# Report ecological status based on observed biophysical indicator
dfIndicator, dfVP, suffix, index = df_ind_obs, df_VP, "obs", None
# Convert the imputed ecological status to categorical scale {0, 1, 2, 3, 4}
dfIndicator, dfVP, suffix, index = dfImp, df_VP, "imp", index_sorted
# Convert moving average of the imputed eco status to categorical scale
dfIndicator, dfVP, suffix, index = dfImpMA, df_VP, "imp_MA", index_sorted

if suffix == "obs":
    # Convert observed biophysical indicator to ecological status
    dfEcoObs = c.indicator_to_status(j, dfIndicator, dfVP)

else:
    # Imputed ecological status using a continuous scale
    dfEcoObs = dfIndicator.copy()

# Save CSV of data on mean ecological status by water body and year
dfEcoObs.to_csv("output\\" + j + "_eco_" + suffix + ".csv")

# Merge observed ecological status each year with basis analysis for VP3
dfEco = dfEcoObs.merge(dfVP[["Basis"]], on="wb")

if suffix != "obs":
    # Prepare for statistics and missing values graph
    for t in dfEco.columns:
        # Precautionary conversion of imputed status to categorical scale
        conditions = [
            dfEco[t] < 0.5,  # Bad
            (dfEco[t] >= 0.5) & (dfEco[t] < 1.5),  #  Poor
            (dfEco[t] >= 1.5) & (dfEco[t] < 2.5),  #  Moderate
            (dfEco[t] >= 2.5) & (dfEco[t] < 3.5),  #  Good
            dfEco[t] >= 3.5,  #  High
        ]
        # Ecological status as a categorical index from Bad to High quality
        dfEco[t] = np.select(conditions, [0, 1, 2, 3, 4], default=np.nan)

if suffix != "imp_MA":
    # Create missing values graph (heatmap of missing observations by year):
    indexSorted = c.missing_values_graph(j, dfEco, suffix, index)

# Merge df for observed ecological status with df for characteristics
dfEcoLength = dfEco.merge(dfVP[["length"]], on="wb")

# Calculate total length of all water bodies in current water body plan (VP2)
totalLength = dfEcoLength["length"].sum()

# Create an empty df for statistics
stats = pd.DataFrame(
    index=c.years + ["Basis"],
    columns=["high", "good", "moderate", "poor", "bad", "not good", "known"],
)

# Calculate the above statistics for span of natural capital account & basis
for t in c.years + ["Basis"]:
    y = dfEcoLength[[t, "length"]].reset_index(drop=True)
    y["high"] = np.select([y[t] == 4], [y["length"]])
    y["good"] = np.select([y[t] == 3], [y["length"]])
    y["moderate"] = np.select([y[t] == 2], [y["length"]])
    y["poor"] = np.select([y[t] == 1], [y["length"]])
    y["bad"] = np.select([y[t] == 0], [y["length"]])
    y["not good"] = np.select([y[t] < 3], [y["length"]])
    y["known"] = np.select([y[t].notna()], [y["length"]])
    # Add shares of total length to stats
    knownLength = y["known"].sum()
    stats.loc[t] = [
        100 * y["high"].sum() / knownLength,
        100 * y["good"].sum() / knownLength,
        100 * y["moderate"].sum() / knownLength,
        100 * y["poor"].sum() / knownLength,
        100 * y["bad"].sum() / knownLength,
        100 * y["not good"].sum() / knownLength,
        100 * knownLength / totalLength,
    ]

# For imputed ecological status, convert to integers and drop 'known' column
if suffix != "obs":
    dfEco = dfEco.astype(int)
    stats = stats.drop(columns="known")

# Save statistics on mean ecological status by year weighted by shore length
stats.to_csv("output\\" + j + "_eco_" + suffix + "_stats.csv")

# Brief analysis of missing observations (not relevant for imputed data)
if suffix == "obs":
    # Create df limited to water bodies that are observed at least one year
    observed = dfEcoObs.dropna(how="all").merge(dfVP[["length"]], how="inner", on="wb")

    # df of water bodies assessed at least one year or in the basis analysis for VP3
    observedVP3 = dfEco.dropna(how="all").merge(dfVP[["length"]], how="inner", on="wb")

    # Report length and share of water bodies observed at least one year (or with basis)
    msg = "{0} km is the total shore length of {1} included in VP3, of which {2}% of {1} representing {3} km ({4}% of total shore length of {1}) have been assessed at least one year. Taking into account the basis analysis for VP3, {5}% of {1} representing {6} km ({7}% of total shore length of {1}) have been assessed. On average, {8}% of {1} representing {9} km ({10}% of total shore length of {1}) are assessed each year.\n".format(
        round(totalLength),
        j,
        round(100 * len(observed) / len(dfEco)),
        round(observed["length"].sum()),
        round(100 * observed["length"].sum() / totalLength),
        round(100 * len(observedVP3) / len(dfEco)),
        round(observedVP3["length"].sum()),
        round(100 * observedVP3["length"].sum() / totalLength),
        round(100 * np.mean(dfEco[c.years].count() / len(dfEco))),
        round(stats.drop("Basis")["known"].mean() / 100 * totalLength),
        round(stats.drop("Basis")["known"].mean()),
    )
    # print(msg)  # print statistics in Python
    arcpy.AddMessage(msg)  # return statistics in ArcGIS

    df_eco_obs = dfEco[dfEcoObs.columns]
    stats_obs_j[j] = stats["not good"]
    index_sorted = indexSorted
    #     return dfEco[dfEcoObs.columns], stats["not good"], indexSorted

    # Elaborate column names of statistics for online presentation
    stats.columns = [
        "Share of known is High (%)",
        "Share of known is Good (%)",
        "Share of known is Moderate (%)",
        "Share of known is Poor (%)",
        "Share of known is Bad (%)",
        "Share of known is not Good (%)",
        "Status known (%)",
    ]
    # Save statistics as Markdown for online presentation
    stats.astype(int).to_html("output\\" + j + "_eco_obs_stats.md")

# return stats["not good"]


# def indicator_to_status(self, j, dfIndicator, df_VP):
"""Convert biophysical indicators to ecological status."""
dfIndicator, dfVP = df_ind_obs, df_VP
cols = ["bad", "poor", "moderate", "good"]
if j == "streams":
    # Copy DataFrame for the biophysical indicator
    df = dfIndicator.copy()
    # Convert DVFI fauna index for streams to index of ecological status
    for t in df.columns:
        # Set conditions given the official guidelines for conversion
        conditions = [
            df[t] < 1.5,  # Bad
            (df[t] >= 1.5) & (df[t] < 3.5),  #  Poor
            (df[t] >= 3.5) & (df[t] < 4.5),  #  Moderate
            (df[t] >= 4.5) & (df[t] < 6.5),  #  Good
            df[t] >= 6.5,  #  High
        ]
        # Ecological status as a categorical scale from Bad to High quality
        df[t] = np.select(conditions, [0, 1, 2, 3, 4], default=np.nan)
    # return df
elif j == "lakes":
    # Merge df for biophysical indicator with df for typology
    df = dfIndicator.merge(dfVP[["ov_typ"]], on="wb")

    def SetThreshold(row):
        if row["ov_typ"] in ["LWTYPE9", "LWTYPE11", "LWTYPE13", "LWTYPE15"]:
            return pd.Series(
                {
                    "bad": 90,
                    "poor": 56,
                    "moderate": 25,
                    "good": 11.7,
                }
            )
        else:
            return pd.Series(
                {
                    "bad": 56,
                    "poor": 27,
                    "moderate": 12,
                    "good": 7,
                }
            )

    # For df, add the series of thresholds relative to High ecological status
    df[cols] = df.apply(SetThreshold, axis=1)
    df = df.drop(columns=["ov_typ"])  #  drop typology column

else:  #  coastal waters
    # Read table of thresholds of chlorophyll for each coastal water body
    thresholds = pd.read_csv("linkage\\" + c.linkage[j][1], index_col=0).astype(int)

    # Merge df for biophysical indicator with df for thresholds
    df = dfIndicator.merge(thresholds[cols], on="wb")

# Convert mean chlorophyll concentrations to index of ecological status
for t in dfIndicator.columns:
    # Set conditions given the threshold for the typology of each lake
    conditions = [
        df[t] >= df["bad"],
        (df[t] < df["bad"]) & (df[t] >= df["poor"]),
        (df[t] < df["poor"]) & (df[t] >= df["moderate"]),
        (df[t] < df["moderate"]) & (df[t] >= df["good"]),
        df[t] < df["good"],
    ]
    # Ordinal scale of ecological status: Bad, Poor, Moderate, Good, High
    df[t] = np.select(conditions, [0, 1, 2, 3, 4], default=np.nan)

# Drop columns with thresholds
df = df.drop(columns=cols)


# def missing_values_graph(self, j, frame, suffix="obs", index=None):
"""Heatmap visualizing observations of ecological status as either missing or using the EU index of ecological status, i.e., from 0-4 for Bad, Poor, Moderate, Good, and High water quality respectively.
Saves a figure of the heatmap."""
frame, suffix, index = dfEco, "obs", None
frame, suffix, index = dfEco, "imp", index_sorted

# Subset DataFrame to for span of natural capital account & basis analysis
df = frame[c.years + ["Basis"]].copy()

if suffix == "obs":
    # Sort by eco status in basis analysis then number of observed values
    df["n"] = df.count(axis=1)
    df = df.sort_values(["Basis", "n"], ascending=False).drop(columns="n")
    # Save index to reuse the order after imputing the missing values
    index = df.index
else:
    # Sort by status in basis analysis & number of observed values as above
    df = df.reindex(index)

# Check df for the presence of any missing values
if df.isna().sum().sum() > 0:
    # Replace missing values with -1
    df.fillna(-1, inplace=True)

    # Specify heatmap to show missing values as gray (xkcd spells it "grey")
    colors = ["grey", "red", "orange", "yellow", "green", "blue"]
    uniqueValues = [-1, 0, 1, 2, 3, 4]

    # Description for heatmap of observed eco status (instead of fig legend)
    description = "Bad (red), Poor (orange), Moderate (yellow), Good (green), High (blue), missing value (gray)"

else:
    # Specify heatmap without any missing values (only for imputed coastal)
    colors = ["red", "orange", "yellow", "green", "blue"]
    uniqueValues = [0, 1, 2, 3, 4]
    description = (
        "Bad (red), Poor (orange), Moderate (yellow), Good (green), High (blue)"
    )

# Plot heatmap
colorMap = sns.xkcd_palette(colors)
plt.figure(figsize=(10, 10))
ax = sns.heatmap(
    df,
    cmap=colorMap,
    cbar=False,
    cbar_kws={"ticks": uniqueValues},
)
ax.set(yticklabels=[])
plt.ylabel(str(len(df)) + " " + j + " ordered by number of missing values")
plt.title(description)
plt.tight_layout()
plt.savefig("output\\" + j + "_eco_" + suffix + ".pdf", bbox_inches="tight")

index_sorted = index
# return index


# def values_by_catchment_area(self, j, dfEcoImpMA, dfVP):
"""Assign water bodies to coastal catchment areas and calculate the weighted arithmetic mean of ecological status after truncating from above at Good status.
For each year t, set up df with variables for the Benefit Transfer equation."""
dfEcoImp, dfVP, frames_j = df_eco_imp_MA, df_VP, {}

if j == "coastal":
    dfEcoImpCatch = dfEcoImp.copy()

    # ID is shared between coastal waters and coastal catchment areas v
    dfEcoImpCatch["v"] = dfEcoImpCatch.index

else:  #  streams and lakes to coastal catchment areas
    # Specify name of joined feature class (polygons)
    jCatch = j + "_catch"

    # Join water bodies with the catchment area they have their center in
    arcpy.SpatialJoin_analysis(
        target_features=j,
        join_features="catch",
        out_feature_class=jCatch,  #  will overwrite if it already exists
        join_operation="JOIN_ONE_TO_MANY",
        match_option="HAVE_THEIR_CENTER_IN",
    )

    # Fields in fc that contain coastal catchment area ID and water body ID
    fields = ["op_id", "ov_id"]

    # Create DataFrame from jCatch of water bodies in each catchment area
    dataCatch = [row for row in arcpy.da.SearchCursor(jCatch, fields)]
    dfCatch = pd.DataFrame(dataCatch, columns=fields)

    # Convert water body ID (wb) and coastal catchment area ID to integers
    dfCatch = dfCatch.copy()  #  to avoid SettingWithCopyWarning
    if j == "lakes":
        dfCatch.loc[:, "wb"] = dfCatch["ov_id"].str.slice(6).astype(int)
    else:
        dfCatch.loc[:, "wb"] = dfCatch["ov_id"].str.slice(7).astype(int)
    dfCatch["v"] = dfCatch["op_id"]

    # Subset to columns; water body ID as index; sort by catchment area ID
    dfCatch = dfCatch[["wb", "v"]].set_index("wb").sort_values(by="v")

    # Assign unjoined water bodies to their relevant coastal catchment area
    if j == "streams":
        dfCatch.loc[3024, "v"] = "113"  #  Kruså to Inner Flensborg Fjord
        dfCatch.loc[8504, "v"] = "233"  #  outlet from Kilen to Venø Bugt
    elif j == "lakes":
        dfCatch.loc[342, "v"] = "233"  #  Nørskov Vig to Venø Bugt
        dfCatch.loc[11206, "v"] = "80"  #  Gamborg Nor to Gamborg Fjord
        dfCatch.loc[11506, "v"] = "136"  #  Lille Langesø to Indre Randers Fjord

    # Merge df for imputed ecological status w. coastal catchment area
    dfEcoImpCatch = dfEcoImp.merge(dfCatch.astype(int), on="wb")

# Merge df for imputed ecological status w. shore length
dfEco = dfEcoImpCatch.merge(dfVP[["length"]], on="wb")

# List of coastal catchment areas where category j is present
j_present = list(dfEco["v"].unique())

# Total length of water bodies of category j by coastal catchment area v
shores_v = dfEco[["v", "length"]].groupby("v").sum().iloc[:, 0]

# Demographics by coastal catchment area v and year t (1990-2018)
dem = pd.read_csv("data\\" + c.data["shared"][1], index_col=[0, 1]).sort_index()

# Years used for interpolation of demographics
t_old = np.arange(c.year_first + 1, 2018 + 1)
t_new = np.arange(c.year_first + 1, c.year_last + 1)

# For each coastal catchment area v, extrapolate demographics to 2019-2020
frames_v = {}  #  dictionary to store df for each coastal catchment area v
for v in dem.index.get_level_values("v").unique():
    df = pd.DataFrame(index=t_new)  #  empty df to store values for each year t
    for col in dem.columns:
        # Function for linear extrapolation
        f = interpolate.interp1d(t_old, dem.loc[v, col], fill_value="extrapolate")
        df[col] = f(t_new)
    frames_v[v] = df  #  store df in dictionary of DataFrames
dfDem = pd.concat(frames_v).sort_index()
dfDem.index.names = ["v", "t"]

# Consumer Price Index by year t (1990-2020)
CPI = pd.read_excel("data\\" + c.data["shared"][0], index_col=0)

# Merge CPI with demographics by v and t (households, age, and hh income)
Dem = dfDem[["N"]].merge(CPI["CPI"], "left", left_index=True, right_index=True)
Dem["D age"] = np.select([dfDem["age"] > 45], [1])  # dummy mean age > 45
# Mean gross real household income (100,000 DKK, 2018 prices) by v and t
Dem["y"] = dfDem["income"] * CPI.loc[2018, "CPI"] / Dem["CPI"] / 100000
Dem["ln y"] = np.log(Dem["y"])  #  log mean gross real household income
Dem = Dem.loc[j_present].reorder_levels([1, 0]).sort_index()

# Geographical data by coastal catchment area v (assumed time-invariant)
Geo = pd.read_excel("data\\" + c.data["shared"][2], index_col=0)
Geo.index.name = "v"
Geo = Geo.loc[j_present].sort_index()

# For each year t, create a df of variables needed for benefit transfer
frames_t = {}  #  create empty dictionary to store a df for each year t

# DataFrame for ecological status of water bodies from above at Good
Q = dfEco.copy()

# Truncate from above at Good ecological status
Q[c.years] = Q[c.years].mask(Q[c.years] > 3, 3)  #  above at Good

# DataFrames with dummy for less than Good ecological status
SL = Q.copy()
SL[c.years] = SL[c.years].mask(SL[c.years] < 3, 1).mask(SL[c.years] >= 3, 0)

# For each year t, create df by v for variables needed for benefit transfer
for t in c.years:
    df = pd.DataFrame()  #  empty df for values by coastal catchment area
    # Q is mean ecological status of water bodies weighted by shore length
    Q[t] = Q[t] * Q["length"]  #  ecological status × shore length
    df["Q"] = Q[["v", t]].groupby("v").sum()[t] / shores_v
    if t > 1989:
        df["ln y"] = Dem.loc[t, "ln y"]  #  ln mean gross real household income
        df["D age"] = Dem.loc[t, "D age"]  #  dummy for mean age > 45 years
        SL[t] = SL[t] * SL["length"]  #  shore length if status < good
        SL_not_good = SL[["v", t]].groupby("v").sum()  #  if status < good
        df["ln PSL"] = SL_not_good[t] / Geo["shores all j"]  #  proportional
        ln_PSL = np.log(df.loc[df["ln PSL"] > 0, "ln PSL"])  #  log PSL
        ln_PSL_full = pd.Series(index=df.index)  #  empty series with index
        ln_PSL_full[df["ln PSL"] != 0] = ln_PSL  #  fill with ln_PSL if > 0
        df["ln PSL"] = df["ln PSL"].mask(df["ln PSL"] > 0, ln_PSL_full)
        df["ln PAL"] = Geo["ln PAL"]  #  proportion arable land
        df["SL"] = SL_not_good / 1000  #  SL in 1,000 km
        if j == "lakes":
            df["D lakes"] = 1
        else:
            df["D lakes"] = 0
        df["N"] = Dem.loc[t, "N"]  #  number of households
    frames_t[t] = df  #  store df in dictionary of DataFrames
dfBT = pd.concat(frames_t)
dfBT.index.names = ["t", "v"]
frames_j[j] = dfBT
shores_j[j] = shores_v


def BT(df, elast=1):
    """Apply Benefit Transfer function from meta study (Zandersen et al., 2022)"""
    # ln MWTP for improvement from current ecological status to "Good"
    lnMWTP = (
        4.142
        + 0.551 * df["Q"]
        + elast * df["ln y"]
        + 0.496 * df["D age"]
        + 0.121 * df["ln PSL"]
        - 0.072 * df["ln PAL"]
        - 0.005 * df["SL"]
        - 0.378 * df["D lakes"]
    )
    # Real MWTP per household (DKK, 2018 prices) using the meta study variance
    MWTP = np.exp(lnMWTP + (0.136 + 0.098) / 2)  #  variance components
    return MWTP


def valuation(dfBT, real=True, investment=False):
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

        # lnΔQ = 0 if all water bodies of type j have ≥ good ecological status
        df["Q"] = df["Q"] * df["nonzero"]

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
    df[["D age", "D lakes", "N"]] = df[["D age", "D lakes", "N"]].astype(int)

    # Consumer Price Index by year t (1990-2020)
    CPI_NPV = pd.read_excel("data\\" + data["shared"][0], index_col=0)

    # Merge data with CPI to correct for assumption of unitary income elasticity
    df1 = df.merge(CPI_NPV, "left", left_index=True, right_index=True)
    df1["unityMWTP"] = BT(df1)  #  MWTP assuming unitary income elasticity

    # Calculate factor that MWTP is increased by if using estimated income ε
    df2018 = df1[df1.index.get_level_values("t") == 2018].copy()
    df2018["elastMWTP"] = BT(df2018, elast=1.453)  #  meta reg income ε
    df2018["factor"] = df2018["elastMWTP"] / df2018["unityMWTP"]
    df2018 = df2018.droplevel("t")
    df2 = df1.merge(df2018[["factor"]], "left", left_index=True, right_index=True)
    df2 = df2.reorder_levels(["j", "t", "v"]).sort_index()

    # Adjust with factor of actual ε over unitary ε; set MWTP to 0 for certain Q
    df2["MWTP"] = df2["unityMWTP"] * df2["factor"] * df2["nonzero"]

    # Aggregate real MWTP per hh over households in coastal catchment area
    df2["CWP"] = df2["MWTP"] * df2["N"] / 1e06  #  million DKK (2018 prices)

    if investment is True:
        # Switch MWTP to negative if actual change is negative
        cond = [df2["neg"] == 1]
        df2["CWP"] = np.select(cond, [-df2["CWP"]], default=df2["CWP"])

        # Apply net present value (NPV) factor using different discount rates r
        if real is False:
            # r as prescribed by Ministry of Finance of Denmark the given year
            df2["CWP"] = df2["CWP"] * df2["NPV"]

        else:
            # Declining r as prescribed by Ministry of Finance during 2014-2020
            df2["CWP"] = df2["CWP"] * CPI_NPV.loc[2018, "NPV"]

            # Rename CWP to IV (investment value of water quality improvements)
            df2 = df2.rename(columns={"CWP": "IV"})  #  million DKK (2018 prices)

            # Return real investment value (IV) by t, v, and j
            return df2["IV"].unstack(level=0)

    if real is True:
        #  Return real cost of water pollution (CWP) by t, v, and j
        return df2["CWP"].unstack(level=0)

    # Aggregate nominal MWTP per hh over households in coastal catchment area
    df2["CWPn"] = df2["CWP"] * df2["CPI"] / CPI_NPV.loc[2018, "CPI"]

    # CWP in prices of the preceding year (for year-by-year chain linking)
    df2["D"] = df2["CWPn"] * df2["CPI t-1"] / df2["CPI"]

    if investment is True:
        # IV in prices of the preceding year (for year-by-year chain linking)
        df2["D"] = df2["D"] * df2["NPV t-1"] / df2["NPV"]

    # Aggregate over coastal catchment areas
    grouped = (
        df2[["CWPn", "D"]]
        .groupby(["j", "t"])
        .sum()
        .unstack(level=0)
        .rename_axis(None)
        .rename_axis([None, None], axis=1)
    )

    if investment is True:
        # Rename nominal IV in prices of current year, and preceding year respectively
        grouped.columns = grouped.columns.set_levels(
            [
                "Investment value (current year's prices, million DKK)",
                "Investment value (preceding year's prices, million DKK)",
            ],
            level=0,
        )

    else:
        # Rename nominal CWP in prices of current year, and preceding year respectively
        grouped.columns = grouped.columns.set_levels(
            [
                "Cost (current year's prices, million DKK)",
                "Cost (preceding year's prices, million DKK)",
            ],
            level=0,
        )

    return grouped  #  in prices of current year and preceding year respectively


def decompose(dfBT, baseYear=2018):
    """Decompose development by holding everything else equal at 2018 level"""

    # Define a function to ready each row for decomposition analysis
    def replace_row(row):
        j, t, v = row.name

        # Fix variables "Q", "ln PSL", and "SL" at base year level for j ≠ driver
        if j != driver:
            row[["Q", "ln PSL", "SL"]] = df.loc[(j, baseYear, v), ("Q", "ln PSL", "SL")]

        # Fix "ln y", "D age", and "N" at base year level for variable ≠ driver
        cols = [col for col in ["ln y", "D age", "N"] if col != driver]
        row[cols] = df.loc[(j, baseYear, v), cols]

        return row

    # Empty dictionaries for decomposed costs of pollution and investment value
    CWP_v, CWP_j, CWP, IV_v, IV = {}, {}, {}, {}, {}

    for driver in ["coastal", "lakes", "streams", "ln y", "D age", "N"]:
        # Copy df with the variables needed for the benefit transfer equation
        df = dfBT.copy()

        # Isolate changes related to driver by holding other things equal
        df = df.apply(replace_row, axis=1)

        # Apply valuation function to decompose the development by driver
        CWP_vj = valuation(df)  #  CWP in catchment area v by category j

        # Costs of Water Pollution in real terms (million DKK, 2018 prices)
        CWP_v[driver] = CWP_vj.sum(axis=1)  #  total CWP in v
        CWP_j[driver] = CWP_vj.groupby("t").sum().rename_axis(None)  #  CWP by j
        CWP[driver] = CWP_j[driver].sum(axis=1)  #  total CWP

        categories = ["coastal", "lakes", "streams"]

        if driver in categories:
            # Investment Value (IV) of water quality improvement in real terms
            IV_v[driver] = valuation(df, investment=True)  #  IV in v by j

            # Drop categories j where the given driver has no effect
            for dict in [CWP_j, IV_v]:
                cols = [col for col in categories if col != driver]
                dict[driver] = dict[driver].drop(columns=cols)
            IV_v[driver] = IV_v[driver][driver]  #  j is redundant (= driver)

            # IV of water quality improvement in real terms by t and j
            IV[driver] = IV_v[driver].groupby("t").sum().rename_axis(None)

    # Concatenate DataFrames for each driver and name hierarchical index
    CWPdriver_v = pd.concat(CWP_v, axis=1, names=["driver"])
    CWPdriver_j = pd.concat(CWP_j, axis=1, names=["driver", "j"])
    CWPdriver = pd.concat(CWP, axis=1, names=["driver"])
    IVdriver_v = pd.concat(IV_v, axis=1, names=["driver"])
    IVdriver = pd.concat(IV, axis=1, names=["driver"])

    return CWPdriver_v, CWPdriver_j, CWPdriver, IVdriver_v, IVdriver
