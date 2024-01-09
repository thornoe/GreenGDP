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
#######################################################################################
#   0. Imports
#######################################################################################
# Import Operation System (os) and ArcPy package (requires ArcGIS Pro installed)
import os

import arcpy
import numpy as np
import pandas as pd

# To use the experimental imputation feature, we must explicitly ask for it:
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge

#######################################################################################
#   1. Setup
#######################################################################################
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

#######################################################################################
#   2. Specifications
#######################################################################################
# Specify the years of interest
year_first = 1987
year_last = 2020

# Specify the names of each type of water body and its data files
data = {
    "catch": ["demographics.csv", "geographical.xlsx"],
    "streams": ["streams_DVFI.xlsx"],
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
waterbodyType = "streams"
fileName = data[waterbodyType][0]
parameterType = "DVFI"
parameterCol = "Indekstype"
valueCol = "Indeks"
radius = 15
seed = 0

#######################################################################################
#   3. Import module and run the functions
#######################################################################################
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

# Get the feature class from the WFS service
c.get_fc_from_WFS(waterbodyType)

# Create a Pandas DataFrame with observed indicator by year
# df_ind_obs, df_VP = c.observed_indicator(waterbodyType)
df_ind_obs = pd.read_csv("output\\" + waterbodyType + "_ind_obs.csv", index_col="wb")
df_ind_obs.columns = df_ind_obs.columns.astype(int)
df_VP = pd.read_csv("output\\" + waterbodyType + "_VP.csv", index_col="wb")

# Report observed ecological status by year
df_eco_obs, obs_stats = c.ecological_status(waterbodyType, df_ind_obs, df_VP)

#######################################################################################
#   4. Run the functions line-by-line
#######################################################################################
arcpy.ListFeatureClasses()
arcpy.Exists(waterbodyType)
arcpy.Exists(fcJoined)
for field in arcpy.ListFields(fc):
    field.name, field.type, field.length
arcpy.Delete_management(fcStations)

# def frame(self, waterbodyType, fileName, parameterType, parameterCol, valueCol):
"""Function to set up a Pandas DataFrame for a given type of water body"""

# def longitudinal(self, waterbodyType, fileName, parameterType):
"""Set up a longitudinal DataFrame based on the type of water body.
Streams: For a given year, finds DVFI for a station with multiple
         observations by taking the median and rounding down."""
df = c.frame(
    waterbodyType,
    fileName,
    parameterType,
    parameterCol="Indekstype",
    valueCol="Indeks",
)
i = 2001

# def stations_to_streams(self, waterbodyType, fileName, radius=15):
"""Streams: Assign monitoring stations to water bodies via linkage table.
For unmatched stations: Assign to stream within a radius of 15 meters where the location names match the name of the stream.
For a given year, finds DVFI for a stream with multiple stations
by taking the median and rounding down.
Finally, extends to all streams in the current water body plan (VP3) and adds the ID, catchment area, and length of each stream in km using the feature classes collected via get_fc_from_WFS()"""
# Create longitudinal DataFrame for stations in streams by monitoring approach
DVFI = c.longitudinal(waterbodyType, fileName, "DVFI")
DVFI_MIB = c.longitudinal(waterbodyType, fileName, "DVFI, MIB")
DVFI_F = c.longitudinal(waterbodyType, fileName, "Faunaklasse, felt")
# Group by station and keep first non-null entry each year DVFI>MIB>felt
long = pd.concat([DVFI, DVFI_MIB, DVFI_F]).groupby(["station"], as_index=False).first()
# Read the linkage table
dfLinkage = pd.read_csv("linkage\\" + c.linkage[waterbodyType])
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
fcStations = waterbodyType + "_station"
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

# def observed_indicator(self, waterbodyType):
"""Based on the type of water body, set up a longitudinal DataFrame
with the observed indicators for all water bodies."""
# Create longitudinal df and use linkage table to assign stations to water bodies
df = c.observed_indicator(waterbodyType)
df = c.stations_to_streams(waterbodyType, fileName=c.data[waterbodyType][0])

# def indicator_to_status(self, waterbodyType, df):
"""Convert biophysical indicators to ecological status."""
df = df_ind_obs
# Convert DVFI fauna index for streams to index of ecological status
for i in c.years:
    # Categorical variable for ecological status: Bad, Poor, Moderate, Good, High
    conditions = [
        df[i] == 1,
        (df[i] == 2) | (df[i] == 3),
        df[i] == 4,
        (df[i] == 5) | (df[i] == 6),
        df[i] == 7,
    ]
    df[i] = np.select(conditions, [0, 1, 2, 3, 4], default=np.nan)

# def ecological_status(self, waterbodyType, dfIndicator, dfVP, suffix):
"""Based on the type of water body, convert the longitudinal DataFrame to the EU index of ecological status, i.e., from 0-4 for Bad, Poor, Moderate, Good, and High water quality respectively.
Create a table of statistics and export it as an html table.
Print the length and share of water bodies observed at least once."""
dfIndicator, dfVP = df_ind_obs, df_VP
dfStatus = c.indicator_to_status(waterbodyType, dfIndicator)
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
    y["high"] = np.select([y[i] == 4], [y["length"]])
    y["good"] = np.select([y[i] == 3], [y["length"]])
    y["moderate"] = np.select([y[i] == 2], [y["length"]])
    y["poor"] = np.select([y[i] == 1], [y["length"]])
    y["bad"] = np.select([y[i] == 0], [y["length"]])
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

# def impute_missing_values(self, dfIndicator, dfVP, seed=0):
dfIndicator = df_ind_obs
dfVP = df_VP
# Copy DataFrame for the biophysical indicator
df = dfIndicator.copy()
df.describe()

# Simple mean as a baseline
imp_mean = SimpleImputer(strategy="mean", keep_empty_features=True)
df_mean = pd.DataFrame(imp_mean.fit_transform(np.array(df)), columns=df.columns)
df_mean.describe()

# Iterative imputer based on all other features
imp = IterativeImputer(
    estimator=BayesianRidge(),
    max_iter=25,
    n_nearest_features=len(df.columns),
    random_state=seed,
    keep_empty_features=True,
)
# Fit imputer and transform the dataset
df_imp = pd.DataFrame(
    imp.fit_transform(np.array(df)), index=df.index, columns=df.columns
)
df_imp.describe()

df_eco_imp, stats_imp = c.ecological_status(waterbodyType, df_imp, df_VP, "imp")
