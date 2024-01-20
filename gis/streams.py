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
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer

########################################################################################
#   1. Setup
########################################################################################
# Set overwrite option
arcpy.env.overwriteOutput = True

# Specify the parent folder as the working directory of the operating system
# os.chdir(arcpy.GetParameterAsText(0))
root = r"C:\Users\au687527\GitHub\GreenGDP"
path = root + "\\gis"
arcpy.env.workspace = path
os.chdir(path)

# Specify whether to replace existing feature classes downloaded from WFS service
# wfs_replace = arcpy.GetParameterAsText(1)
wfs_replace = 0

# Specify whether to keep the geodatabase when the script finishes
# keep_gdb = arcpy.GetParameterAsText(1)
keep_gdb = 1

########################################################################################
#   2. Specifications
########################################################################################
# Span of natural capital account (1990 investment value depends on change from 1989)
year_first = 1989
year_last = 2020

# Specify the names of each type of water body and its data files
data = {
    "catch": ["demographics.csv", "geographical.xlsx"],
    "streams": ["streams_DVFI.xlsx", "streams_1988-2020.xlsx"],
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
fileName = data[j][0]
parameterType = "DVFI"
parameterCol = "Indekstype"
valueCol = "Indeks"
radius = 15

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

# Create empty DataFrames to store weighted ecological status by catchment area
coastal, lakes, streams = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

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
df_eco_imp, imp_stats = c.impute_missing(j, df_ind_obs, df_VP, index_sorted)
df_eco_imp

########################################################################################
#   4. Run the functions line-by-line
########################################################################################
arcpy.ListFeatureClasses()
arcpy.Exists(j)
for field in arcpy.ListFields(j):
    field.name, field.type, field.length
arcpy.Delete_management(j)

# def frame(self, j, fileName, parameterType, parameterCol, valueCol):
"""Function to set up a Pandas DataFrame for a given type of water body"""

# def longitudinal(self, j, fileName, parameterType):
"""Set up a longitudinal DataFrame based on the type of water body.
Streams: For a given year, finds DVFI for a station with multiple
         observations by taking the median and rounding down."""
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
# Convert waterbody ID (wb) to integers
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
# Number of stations matched to waterbody by radius threshold
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
dfIndicator, dfVP = df_ind_obs, df_VP
dfStatus = c.indicator_to_status(j, dfIndicator)
# Merge df for observed ecological status with df for characteristics
df = dfVP.merge(dfStatus, how="inner", on="wb")
# Calculate total length of all water bodies in current water body plan (VP2)
totalLength = df["length"].sum()
# Create an empty df for statistics
stats = pd.DataFrame(
    index=c.years,
    columns=[
        "Status known (%)",
        "Share of known is High (%)",
        "Share of known is Good (%)",
        "Share of known is Moderate (%)",
        "Share of known is Poor (%)",
        "Share of known is Bad (%)",
    ],
)
# Calculate the above statistics for each year
for i in c.years:
    y = df[[i, "length"]].reset_index(drop=True)
    y["known"] = np.select([y[i].notna()], [y["length"]])
    y["high"] = np.select([y[i] == 5], [y["length"]])
    y["good"] = np.select([y[i] == 4], [y["length"]])
    y["moderate"] = np.select([y[i] == 3], [y["length"]])
    y["poor"] = np.select([y[i] == 2], [y["length"]])
    y["bad"] = np.select([y[i] == 1], [y["length"]])
    # Add shares of total length to stats
    knownLength = y["known"].sum()
    stats.loc[i] = [
        100 * knownLength / totalLength,
        100 * y["high"].sum() / knownLength,
        100 * y["good"].sum() / knownLength,
        100 * y["moderate"].sum() / knownLength,
        100 * y["poor"].sum() / knownLength,
        100 * y["bad"].sum() / knownLength,
    ]
stats

# def impute_missing_values(self, dfIndicator, dfVP):
"""Impute ecological status for all water bodies from the observed indicator."""
# DataFrames for observed biophysical indicator and typology
dfIndObs, dfVP = df_ind_obs, df_VP

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

# Convert imputed biophysical indicator to ecological status
dfEcoImp, impStats = c.ecological_status(j, dfImp, dfVP, "imp", indexSorted)

df_eco_imp, imp_stats = dfEcoImp, impStats


### DUMMY FOR AGE>45 BY CATCHMENT AREA
df = pd.read_csv("data\\" + "demographics.csv", sep=";")  # 1990-2018

# Select relevant columns
df = df[["Kystvnd", "aar", "alder", "antal_pers"]]
df["flat"] = 1

# Dummy for mean age > 45 years for a given catchment area and year
df["D_age"] = np.select([df["alder"] > 45], [1])

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
