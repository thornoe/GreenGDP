"""
Name:       script_streams.py

Label:      Construct and map longitudinal data of ecological status of streams.

Summary:    ThorNoe.GitHub.io/GreenGDP explains the overall approach and methodology.

Rqmts:      ArcGIS Pro must be installed on the system and be up to date.

Usage:      This sandbox is line-by-line implementation of the script supporting 
            WaterbodiesScriptTool in the gis.tbx toolbox, however, for streams only.
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
import numpy as np
import pandas as pd

# To use the experimental imputation feature, we must explicitly ask for it:
from matplotlib import pyplot as plt
from scipy import interpolate
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer

# Color-blind-friendly ordinal scale, modified from: gist.github.com/thriveth/8560036
ColorCycle = {
    "blue": "#377eb8",
    "orange": "#ff7f00",
    "green": "#4daf4a",
    "gray": "#999999",  #  moved up
    "pink": "#f781bf",
    "brown": "#a65628",
    "purple": "#984ea3",
    "yellow": "#dede00",
    "red": "#e41a1c",  #  moved down
}

# Set the default color map and figure size for pyplots
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", list(ColorCycle.values()))
plt.rcParams["figure.figsize"] = [10, 6.18]  #  golden ratio (paper with narrow margins)

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
    "streams": ["streams_DVFI.xlsx", "streams_1988-2020.xlsx"],
    "shared": ["CPI_NPV.xlsx", "demographics.csv", "geographical.xlsx"],
}

# Specify the names of the corresponding linkage files
linkage = {
    "coastal": "coastal_stations_VP3.csv",
    "lakes": "lakes_stations_VP3.csv",
    "streams": "streams_stations_VP3.csv",
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

# Specifications specific to streams
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
    wfs_replace,
    keep_gdb,
)

# Dictionaries to store DataFrame, shore length, and stats for each category j
frames_j = {}
shores_j = {}
stats_j = {}

# Get the feature class from the WFS service
c.get_fc_from_WFS(j)

# Create a DataFrame with observed biophysical indicator by year
# df_ind_obs, df_VP = c.observed_indicator(j)
df_ind_obs = pd.read_csv("output\\" + j + "_ind_obs.csv", index_col="wb")
df_ind_obs.columns = df_ind_obs.columns.astype(int)
df_VP = pd.read_csv("output\\" + j + "_VP.csv", index_col="wb")

# Report ecological status based on observed biophysical indicator
df_eco_obs, obs_stats, index_sorted = c.ecological_status(j, df_ind_obs, df_VP)

# if j == 'streams':
#     # Create a map book with yearly maps of observed ecological status
#     c.map_book(j, df_eco_obs)

# Impute missing values for biophysical indicator and return ecological status
df_eco_imp, stats_j[j] = c.impute_missing(j, df_ind_obs, df_VP, index_sorted)

# Set up df with variables by coastal catchment area for the Benefit Transfer equation
frames_j[j], shores_j[j] = c.values_by_catchment_area(j, df_eco_imp, df_VP)


########################################################################################
#   3.b Sandbox: Run the functions line-by-line
########################################################################################
arcpy.ListFeatureClasses()
arcpy.Exists(j)
for field in arcpy.ListFields("streams_catch"):
    field.name, field.type, field.length
arcpy.Delete_management("catch")

# def frame(self, j, fileName, parameterType, parameterCol, valueCol):
"""Function to set up a Pandas DataFrame for a given type of water body"""
fileName = data[j][0]
parameterType, parameterCol, valueCol = "DVFI", "Indekstype", "Indeks"

# def longitudinal(self, j, fileName, parameterType):
"""Set up a longitudinal DataFrame based on the type of water body.
Streams: For a given year, finds DVFI for a station with multiple
         observations by taking the median and rounding down."""
fileName, parameterType = data[j][0], "DVFI"
df = c.frame(
    j,
    fileName,
    parameterType,
    parameterCol="Indekstype",
    valueCol="Indeks",
)
i = 2001

# def stations_to_streams(self, j, fileName, radius=15):
"""Streams: Assign monitoring stations to water bodies via linkage table.
For unmatched stations: Assign to stream within a radius of 15 meters where the location names match the name of the stream.
For a given year, finds DVFI for a stream with multiple stations
by taking the median and rounding down.
Finally, extends to all streams in the current water body plan (VP3) and adds the ID, catchment area, and length of each stream in km using the feature classes collected via get_fc_from_WFS()"""
fileName, radius = data[j][0], 15
# Create longitudinal df for stations in streams by monitoring version
d = c.data[j][1]
x = "Xutm_Euref89_Zone32"
y = "Yutm_Euref89_Zone32"
DVFI_F = c.longitudinal(j, d, x, y, "Indeks", "Indekstype", "Faunaklasse, felt")
DVFI_M = c.longitudinal(j, d, x, y, "Indeks", "Indekstype", "DVFI, MIB")
DVFI1 = c.longitudinal(j, d, x, y, "Indeks", "Indekstype", "DVFI")
# Create longitudinal df for stations in streams after 2020
DVFI2 = c.longitudinal(j, c.data[j][0], "Målested X-UTM", "Målested Y-UTM)", "Indeks")
# Group by station and keep first non-null entry each year DVFI>MIB>felt
long = (
    pd.concat([DVFI_F, DVFI_M, DVFI1, DVFI2])
    .groupby(["station"], as_index=False)
    .last()
)
# Read the linkage table
dfLinkage = pd.read_csv("linkage\\" + c.linkage[j])
# Convert station-ID to integers
dfLinkage["station"] = dfLinkage["station_id"].str.slice(7).astype(int)
# Merge longitudinal DataFrame with linkage table for water bodies in VP3
df = long.merge(dfLinkage[["station", "ov_id"]], how="left", on="station")
# Stations covered by the linkage tabel for the third water body plan VP3
link = df.dropna(subset=["ov_id"])
# Convert water body ID (wb) to integers
link["wb"] = link["ov_id"].str.slice(7).astype(int)
# Stations not covered by the linkage table for VP3
noLink = df[df["ov_id"].isna()].drop(columns=["ov_id"])
# Create a spatial reference object with the same geographical coordinate system
spatialRef = arcpy.SpatialReference("ETRS 1989 UTM Zone 32N")
# Specify name of feature class for stations (points)
fcStations = j + "_station"
# Create new feature class shapefile (will overwrite if it already exists)
arcpy.CreateFeatureclass_management(
    c.arcPath, fcStations, "POINT", spatial_reference=spatialRef
)
# Number of stations matched to water body by radius threshold
for r in (0.5, 1, 2, 5, 10, 15, 20, 30, 50):
    r
    len(j[j.Distance <= r])
    len(j[(j.Distance <= r) & j.match == True])
    len(jClosest[jClosest.Distance <= r])

len(link), len(noLink), len(jClosest), len(allMatches), len(waterbodies), len(allVP)

# def observed_indicator(self, j):
"""Based on the type of water body, set up a longitudinal DataFrame
with the observed indicators for all water bodies."""
# Create longitudinal df and use linkage table to assign stations to water bodies
df = c.observed_indicator(j)
df = c.stations_to_streams(j, fileName=c.data[j][0])

# def indicator_to_status(self, j, df):
"""Convert biophysical indicators to ecological status."""
df = df_ind_obs
# Convert DVFI fauna index for streams to index of ecological status
for i in c.years:
    # Categorical variable for ecological status: Bad, Poor, Moderate, Good, High
    conditions = [
        df[i] < 1.5,
        (df[i] >= 1.5) & (df[i] < 3.5),
        (df[i] >= 3.5) & (df[i] < 4.5),
        (df[i] >= 4.5) & (df[i] < 6.5),
        df[i] >= 6.5,
    ]
    df[i] = np.select(conditions, [0, 1, 2, 3, 4], default=np.nan)

# def ecological_status(self, j, dfIndicator, dfVP, suffix):
"""Based on the type of water body, convert the longitudinal DataFrame to the EU index of ecological status, i.e., from 1-5 for Bad, Poor, Moderate, Good, and High water quality respectively.
Create a table of statistics and export it as an html table.
Print the length and share of water bodies observed at least once."""
dfIndicator, dfVP, suffix, index = df_ind_obs, df_VP, "obs", None
# Convert index of indicators to index of ecological status
dfEco = c.indicator_to_status(j, dfIndicator)
# Create missing values graph (heatmap of missing observations by year):
indexSorted = c.missing_values_graph(j, dfEco, suffix, index)
# Merge df for observed ecological status with df for characteristics
df = dfEco.merge(dfVP[["length"]], how="inner", on="wb")
# Calculate total length of all water bodies in current water body plan (VP2)
totalLength = df["length"].sum()
# Create an empty df for statistics
stats = pd.DataFrame(
    index=c.years,
    columns=["high", "good", "moderate", "poor", "bad", "not good", "known"],
)
# Calculate the above statistics for each year
for t in c.years:
    y = df[[t, "length"]].reset_index(drop=True)
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
stats
# For imputed ecological status, convert to integers and drop 'known' column
if suffix != "obs":
    dfEco = dfEco.astype(int)
    stats = stats.drop(columns="known")
# Save both dataset and statistics on ecological status to CSV
dfEco.to_csv("output\\" + j + "_eco_" + suffix + ".csv")
stats.to_csv("output\\" + j + "_eco_" + suffix + "_stats.csv")
# Brief analysis of missing observations (not relevant for imputed data)
if suffix == "obs":
    # Create df limited to water bodies that are observed at least once
    observed = dfVP[["length"]].merge(
        dfEco.dropna(how="all"),
        how="inner",
        on="wb",
    )
    # Report length and share of water bodies observed at least once.
    msg = "{0} km is the total shore length of {1} included in VP3, of which {2}% of {1} representing {3} km ({4}% of total shore length of {1}) have been assessed at least once. On average, {5}% of {1} representing {6} km ({7}% of total shore length of {1}) are assessed each year.\n".format(
        round(totalLength * 10 ** (-3)),
        j,
        round(100 * len(observed) / len(df)),
        round(observed["length"].sum() * 10 ** (-3)),
        round(100 * observed["length"].sum() / totalLength),
        round(100 * np.mean(dfEco.count() / len(df))),
        round(stats["known"].mean() / 100 * totalLength * 10 ** (-3)),
        round(stats["known"].mean()),
    )
    # print(msg)  # print statistics in Python
    arcpy.AddMessage(msg)  # return statistics in ArcGIS
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

# def impute_missing_values(self, dfIndicator, dfVP):
"""Impute ecological status for all water bodies from the observed indicator."""
# DataFrames for observed biophysical indicator and typology
dfIndObs, dfVP, stats_j = df_ind_obs, df_VP, {}

# Create dummies for typology
typology = pd.get_dummies(dfVP["ov_typ"]).astype(int)
typology["softBottom"] = typology["RW4"] + typology["RW5"]
typology.columns = [
    "small",
    "medium",
    "large",
    "smallSoftBottom",
    "mediumSoftBottom",
    "softBottom",
]

# Create dummy for the district DK2 (Sealand, Lolland, Falster, and Møn)
district = pd.get_dummies(dfVP["distr_id"]).astype(int)

# Merge dummies for typology and water body district DK2
col = ["small", "medium", "large", "softBottom"]
dummies = typology[col].merge(district["DK2"], how="inner", on="wb")

# Merge DataFrames for observed biophysical indicator and dummies
dfIndObsDum = dfIndObs.merge(dummies, how="inner", on="wb")

# Iterative imputer using BayesianRidge() estimator with increased tolerance
imputer = IterativeImputer(random_state=0, tol=1e-1)

# Fit imputer, transform data iteratively, and limit to years of interest
dfImp = pd.DataFrame(
    imputer.fit_transform(np.array(dfIndObsDum)),
    index=dfIndObsDum.index,
    columns=dfIndObsDum.columns,
)[c.years]

# def values_by_catchment_area(self, j, dfEcoImp, dfVP):
"""Assign water bodies to coastal catchment areas and calculate the weighted arithmetic mean of ecological status after truncating from above at Good status.
For each year t, set up df with variables for the Benefit Transfer equation."""
dfEcoImp, dfVP, frames_j = df_eco_imp, df_VP, {}

# Specify name of joined feature class (polygons)
jCatch = j + "_catch"

# Join water bodies with the coastal catchment area they have their center in
arcpy.SpatialJoin_analysis(
    target_features=j,
    join_features="catch",
    out_feature_class=jCatch,  #  will overwrite if it already exists
    join_operation="JOIN_ONE_TO_MANY",
    match_option="HAVE_THEIR_CENTER_IN",
)

# Fields in fc that contain coastal catchment area ID and water body ID
fields = ["op_id", "ov_id"]

# Create DataFrame from jCatch of water bodies in each coastal catchment area
dataCatch = [row for row in arcpy.da.SearchCursor(jCatch, fields)]
dfCatch = pd.DataFrame(dataCatch, columns=fields)

# Convert water body ID (wb) and coastal catchment area ID to integers
dfCatch["wb"] = dfCatch["ov_id"].str.slice(7).astype(int)
dfCatch["v"] = dfCatch["op_id"]

# Specify columns, water body ID as index, sort by coastal catchment area ID (ascending)
dfCatch = dfCatch[["wb", "v"]].set_index("wb").sort_values(by="v")

if j == "streams":
    # Assign unjoined water bodies to their relevant coastal catchment area
    dfCatch.loc[3024, "v"] = "113"  #  assign Kruså to Inner Flensborg Fjord
    dfCatch.loc[8504, "v"] = "233"  #  assign outlet from Kilen to Venø Bugt

# Merge df for imputed ecological status w. coastal catchment area and length
dfEcoImpCatch = dfEcoImp.merge(dfCatch, how="inner", on="wb").astype(int)
dfEco = dfEcoImpCatch.merge(dfVP[["length"]], how="inner", on="wb")

# List of coastal catchment areas where category j is present
j_present = list(dfEco["v"].unique())

# Total length of water bodies of category j by coastal catchment area v
shores_v = dfEco[["v", "length"]].groupby("v").sum().iloc[:, 0]

# Demographics by coastal catchment area v and year t (1990-2018)
dem = pd.read_csv("data\\" + c.data["shared"][1], index_col=[0, 1]).sort_index()

# Years used for interpolation of demographics
t_old = np.arange(1990, 2018 + 1)
t_new = np.arange(1990, 2020 + 1)

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

# For each year t, create a DataFrame of variables needed for benefit transfer
frames_t = {}  # create empty dictionary to store a DataFrame for each year t

# DataFrame for water quality truncated from above at Good ecological status
Q = dfEco.mask(dfEco == 4, 3)

# Create DataFrames with dummy for less than good ecological status
SL = Q.copy()
SL[t_new] = SL[t_new].mask(SL[t_new] < 3, 1).mask(SL[t_new] == 3, 0)

for t in c.years:
    df = pd.DataFrame()  #  empty DataFrame for values by coastal catchment area
    # \bar{Q}: mean ecological status of water bodies weighted by shore length
    Q[t] = Q[t] * Q["length"]  #  ecological status × shore length
    df["Q"] = Q[["v", t]].groupby("v").sum()[t] / shores_v
    if t > 1989:
        df["ln y"] = Dem.loc[t, "ln y"]  #  log mean gross real household income
        df["D age"] = Dem.loc[t, "D age"]  #  dummy for mean age > 45 years
        SL[t] = SL[t] * SL["length"]  #  shore length if ecological status <good
        SL_not_good = SL[["v", t]].groupby("v").sum()  #  SL where status < good
        df["ln PSL"] = SL_not_good[t] / Geo["shore all j"]  #  proportional SL
        df["ln PSL"] = np.select([df["ln PSL"] != 0], [np.log(df["ln PSL"])])
        df["ln PAL"] = Geo["ln PAL"]  #  proportion arable land
        df["SL"] = SL_not_good / 1000  #  SL in 1,000 km
        if j == "lakes":
            df["D lake"] = 1
        else:
            df["D lake"] = 0
        df["N"] = Dem.loc[t, "N"]  #  number of households
    frames_t[t] = df  #  store df in dictionary of DataFrames
dfBT = pd.concat(frames_t)
dfBT.index.names = ["t", "v"]
frames_j[j] = dfBT
shores_j[j] = shores_v

########################################################################################
#   4.a Stats for all categories j: Shore length and share of it where eco status < Good
########################################################################################
# Set up DataFrame of shore length for each category j ∈ {coastal, lakes, streams}
shores = pd.DataFrame(shores_j)
# shores["shore all j"] = shores["coastal"] + shores["lakes"] + shores["streams"]
shores.to_csv("output\\all_VP_shore length.csv")  #  save to csv

# Set up DataFrame of statistics for each category j ∈ {coastal, lakes, streams}
stats = pd.DataFrame(stats_j)
stats.to_csv("output\\all_eco_imp_not good.csv")  #  save to csv

# Plot water bodies by category (mean ecological status weighted by length)
f1 = (
    stats.drop(1989)
    .plot(ylabel="Share of category with less than good ecological status")
    .get_figure()
)
f1.savefig("output\\all_eco_imp_not good.pdf", bbox_inches="tight")

########################################################################################
#   4.b Marginal willingness to pay (MWTP) for improvement of water quality to "Good"
########################################################################################
# Concatenate DataFrames for each category j ∈ {coastal, lakes, streams}
df_BT = pd.concat(frames_j)
df_BT.index.names = ["j", "t", "v"]
df_BT.to_csv("output\\all_eco_imp.csv")  #  save to csv
df_BT = pd.read_csv("output\\all_eco_imp.csv", index_col=[0, 1, 2]).sort_index()

# Costs of pollution in prices of current year, and preceding year respectively
CWP = c.valuation(df_BT)
CWP.to_csv("output\\all_cost.csv")  #  save to csv for chain linking

# Costs of pollution in real values (million DKK, 2018 prices)
RWP_v = c.valuation(df_BT, real=True)
RWP = RWP_v.groupby(["j", "t"]).sum().unstack(level=0).rename_axis(None)  #  sum over v
RWP.rename_axis([None, None], axis=1).to_csv("output\\all_cost_real.csv")
f2 = (
    RWP.loc[:, "RWP"]
    .rename_axis(None, axis=1)
    .plot(ylabel="Costs of current water pollution (million DKK, 2018 prices)")
    .get_figure()
)
f2.savefig("output\\all_cost_real.pdf", bbox_inches="tight")

# Investment value of increase (decrease) in water quality
IV = c.valuation(df_BT, investment=True)
IV.to_csv("output\\all_investment.csv")  #  save to csv for chain linking

# def valuation(self, dfBT, real=False, investment=False):
"""Valuation of water quality as either current costs or investment value (IV).
If not set to return real values (2018 prices), instead returns values in the prices of both the current year and the preceding year (for year-by-year chain linking)."""
# Copy DataFrame with the variables needed for the benefit transfer equation
d = df_BT.copy()
real = False
investment = False

if investment is False:
    # MWTP = 0 if all water bodies of type j have ≥ Good ecological status
    d["nonzero"] = np.select([d["Q"] < 2.99], [1])  #  dummy

    # Distance from current to Good: convert mean Q to lnΔQ ≡ ln(Q good - Q)
    d["Q"] = np.select([d["Q"] < 2.99], [np.log(3 - d["Q"])])

else:
    # Actual change in ecological status since preceding year
    d = d.reorder_levels(["j", "v", "t"]).sort_index()
    d["Q"] = d["Q"].diff()
    d = d.reorder_levels(["j", "t", "v"]).sort_index()

    # MWTP = 0 if actual change in water quality is zero
    d["nonzero"] = np.select([d["Q"] != 0], [1])  #  dummy

    # Mark if actual change is negative (used to switch MWTP to negative)
    d["neg"] = np.select([d["Q"] < 0], [1])  #  dummy

    # Convert Q to the log of the actual change
    conditions = [d["Q"] > 0, d["Q"] < 0]
    d["Q"] = np.select(conditions, [np.log(d["Q"]), np.log(-d["Q"])])

# Drop year 1989 and specify integer values
d = d.drop(d[d.index.get_level_values("t") == 1989].index)
d[["D age", "D lake", "N"]] = d[["D age", "D lake", "N"]].astype(int)

# Consumer Price Index by year t (1990-2020)
CPI = pd.read_excel("data\\" + c.data["shared"][0], index_col=0)

# Merge data with CPI to correct for assumption of unitary income elasticity
kw = dict(how="left", left_index=True, right_index=True)
df1 = d.merge(CPI, **kw)
df1["unityMWTP"] = c.BT(df1)  #  MWTP assuming unitary income elasticity
df2018 = df1[df1.index.get_level_values("t") == 2018].copy()
df2018["elastMWTP"] = c.BT(df2018, elast=1.453)  #  meta study income ε
df2018["factor"] = df2018["elastMWTP"] / df2018["unityMWTP"]
df2018 = df2018.droplevel("t")
df2 = df1.merge(df2018[["factor"]], **kw)
df2 = df2.reorder_levels(["j", "t", "v"]).sort_index()
df2["MWTP"] = df2["unityMWTP"] * df2["factor"] * df2["nonzero"]

# Aggregate real MWTP per hh over households in coastal catchment area
df2["RWP"] = df2["MWTP"] * df2["N"] / 1e06  #  million DKK (2018 prices)

if investment is True:
    # Apply net present value (NPV) factor
    df2["RWP"] = df2["RWP"] * df2["NPV"]

    # Switch MWTP to negative if actual change is negative
    cond = [df2["neg"] == 1]
    df2["RWP"] = np.select(cond, [-df2["RWP"]], default=df2["RWP"])

# Aggregate nominal MWTP per hh over households in coastal catchment area
df2["CWP"] = df2["RWP"] * df2["CPI"] / CPI.loc[2018, "CPI"]  #  million DKK

# CWP in the prices of the preceding year (for year-by-year chain linking)
df2["D"] = df2["CWP"] * df2["CPI t-1"] / df2["CPI"]  #  million DKK

# Aggregate over coastal catchment areas
grouped = (
    df2[["CWP", "D"]]
    .groupby(["j", "t"])
    .sum()
    .unstack(level=0)
    .rename_axis(None)
    .rename_axis([None, None], axis=1)
)

if investment is True:
    # Rename CWP to IV
    grouped.columns = grouped.columns.set_levels(["IV", "D"], level=0)

########################################################################################
#   4.c Marginal willingness to pay (MWTP) for investments in water quality
########################################################################################
# Concatenate DataFrames for each category j ∈ {coastal, lakes, streams}
d = pd.concat(frames_j)
d.index.names = ["j", "t", "v"]


########################################################################################
#   5. Decompose growth by holding everything else equal
########################################################################################
# MultiIndex: see https://kanoki.org/2022/07/25/pandas-select-slice-rows-columns-multiindex-dataframe/#using-get_level_values
Dem = Dem[Dem.index.get_level_values("v").isin(shores_v)]

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
kw = dict(method="index", fill_value="extrapolate", limit_direction="both")
df.interpolate(**kw, inplace=True)

# Save to CSV
df.to_csv("output\\" + "D_age.csv")
