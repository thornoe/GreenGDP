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
linestyle_cycler = cycler(linestyle=["-", "--", ":", "-.", "-", "--", ":"])  #  7 styles
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
    "streams": ["streams_DVFI.xlsx", "streams_1988-2020.xlsx"],
    "shared": ["CPI_NPV.xlsx", "demographics.csv", "geographical.xlsx"],
}

# Specify the names of the corresponding linkage files (and other supporting tables)
linkage = {
    "coastal": ["coastal_stations_VP3.csv", "coastal_chlorophyll_limits.csv"],
    "lakes": ["lakes_stations_VP3.csv", "lakes_stations_XY.csv"],
    "streams": ["streams_stations_VP3.csv", "streams_stations_XY.csv"],
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
j = "coastal"
# j = "lakes"
# j = "streams"

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
df_eco_imp = pd.read_csv("output\\" + j + "_ind_obs.csv", index_col="wb")
df_eco_imp.columns = df_eco_imp.columns.astype(int)

# Set up df with variables by coastal catchment area for the Benefit Transfer equation
frames_j[j], shores_j[j] = c.values_by_catchment_area(j, df_eco_imp, df_VP)


########################################################################################
#   3.b Sandbox: Run the functions line-by-line
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
# Specify shore length for each category j
if j == "streams":
    # Shore length is counted on both sides of the stream; convert to km
    dfVP[["length"]] = dfVP[["Shape_Length"]] * 2 / 1000
elif j == "lakes":
    # Shore length is the circumference, i.e. Shape_Length; convert to km
    dfVP[["length"]] = dfVP[["Shape_Length"]] / 1000
else:  #  coastal waters
    # Coastline by Zandersen et al.(2022) based on Corine Land Cover 2018
    Geo = pd.read_excel("data\\" + c.data["shared"][2], index_col=0)
    Geo.index.name = "wb"
    # Merge with df for all water bodies in VP3
    dfVP[["length"]] = Geo[["shore coastal"]]


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


# def ecological_status(self, j, dfIndicator, dfTyp, suffix="obs", index=None):
"""Call indicator_to_status() to convert the longitudinal DataFrame to the EU index of ecological status, i.e., from 0-4 for Bad, Poor, Moderate, Good, and High water quality based on the category and typology of each water body.
Also call missing_values_graph() to map missing observations by year.
Create a table of statistics and export it as an html table.
Print the shore length and share of water bodies observed at least once."""
dfIndicator, dfVP, suffix, index = df_ind_obs, df_VP, "obs", None
if suffix == "obs":
    # Convert observed biophysical indicator to ecological status
    dfEco = c.indicator_to_status(j, dfIndicator, dfVP)
elif j == "streams":
    # Convert imputed biophysical indicator to ecological status for streams
    dfEco = c.indicator_to_status(j, dfIndicator, dfVP)
else:  #  lakes and coastal waters
    # Ecological status was imputed directly but at a continuous scale
    dfEco = dfIndicator.copy()
    # Convert predicted ecological status to its ordinal scale
    for t in dfEco.columns:
        conditions = [
            dfEco[t] < 0.5,  # Bad
            (dfEco[t] >= 0.5) & (dfEco[t] < 1.5),  #  Poor
            (dfEco[t] >= 1.5) & (dfEco[t] < 2.5),  #  Moderate
            (dfEco[t] >= 2.5) & (dfEco[t] < 3.5),  #  Good
            dfEco[t] >= 3.5,  #  High
        ]
        # Ecological status as a categorical index from Bad to High quality
        dfEco[t] = np.select(conditions, [0, 1, 2, 3, 4], default=np.nan)
# Create missing values graph (heatmap of missing observations by year):
indexSorted = c.missing_values_graph(j, dfEco, suffix, index)
# Merge df for observed ecological status with df for characteristics
df = dfEco.merge(dfVP[["length"]], on="wb")
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
        round(totalLength),
        j,
        round(100 * len(observed) / len(df)),
        round(observed["length"].sum()),
        round(100 * observed["length"].sum() / totalLength),
        round(100 * np.mean(dfEco[c.years].count() / len(df))),
        round(stats["known"].mean() / 100 * totalLength),
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


# def impute_missing_values(self, dfIndicator, dfVP, index):
"""Impute ecological status for all water bodies from the observed indicator."""
# DataFrames for observed biophysical indicator and typology
dfIndObs, dfVP, stats_j, index = df_ind_obs, df_VP, {}, index_sorted

# Specify the biophysical indicator for the current category
if j == "streams":
    indicator = "til_oko_bb"  #  bottom fauna measured as DVFI index
else:  #  lakes and coastal waters
    indicator = "til_oko_fy"  #  phytoplankton measured as chlorophyll

# Include ecological status as assessed in basis analysis for VP3
basis = dfVP[[indicator]].copy()
# Define a dictionary to map the Danish strings to an ordinal scale
status_dict = {
    "Dårlig økologisk tilstand": 0,
    "Ringe økologisk tilstand": 1,
    "Moderat økologisk tilstand": 2,
    "God økologisk tilstand": 3,
    "Høj økologisk tilstand": 4,
    "Dårligt økologisk potentiale": 0,
    "Ringe økologisk potentiale": 1,
    "Moderat økologisk potentiale": 2,
    "Godt økologisk potentiale": 3,
    "Maksimalt økologisk potentiale": 4,
    "Ukendt": np.nan,
}
# Replace Danish strings in the df with the corresponding ordinal values
basis.replace(status_dict, inplace=True)
basis.columns = ["basis"]

# Convert biophysical indicator to ecological status before imputing
dfObs = c.indicator_to_status(j, dfIndObs, dfVP).merge(basis, on="wb")

# Get typology
typ = dfVP[["ov_typ"]].copy()

# Define the dictionaries
dict1 = {
    "Nordsø": "No",
    "Kattegat": "K",
    "Bælthav": "B",
    "Østersøen": "Ø",
    "Fjord": "Fj",
    "Vesterhavsfjord": "Vf",
}
dict2 = {
    "water exchange": "Vu",
    "freshwater inflow": "F",
    "water depth": "D",
    "stratification": "L",
    "sediment": "Se",
    "salinity": "Sa",
    "tide": "T",
}

# Reverse the dictionaries so the abbreviations are the keys
dict1 = {v: k for k, v in dict1.items()}
dict2 = {v: k for k, v in dict2.items()}

# Combine the dictionaries
dicts = {**dict1, **dict2}


# Define a function to process each string
def process_string(s):
    # Drop the hyphen and everything following it
    s = s.split("-")[0]

    # Create a dictionary with the relevant abbreviations as keys and 1s as values
    dummies = {}

    # Check for abbreviations from dict1 first
    for abbr in dict1:
        if abbr in s:
            dummies[abbr] = 1
            s = s.replace(abbr, "")  # Remove the matched abbreviation from the string

    # Then check for abbreviations from dict2
    for abbr in dict2:
        if abbr in s:
            dummies[abbr] = 1

    return dummies


# Apply the function to typ["ov_typ"] to create a new DataFrame with the dummy variables
dummies = typ["ov_typ"].apply(process_string).apply(pd.Series)

# Replace NaN values with 0
dummies = dummies.fillna(0).astype(int)

# Dummies for typology that improves prediction accuracy (omit fjords as reference)
cols = ["No", "K", "B", "Ø", "Vf", "Vu", "D", "L", "Se", "T"]

# Merge DataFrame for observed values with DataFrame for dummies
dfObsDum = dfObs.merge(dummies[cols], on="wb")

# Iterative imputer using BayesianRidge() estimator with increased tolerance
imputer = IterativeImputer(tol=1e-1, max_iter=50, random_state=0)

# Fit imputer, transform data iteratively, and limit to years of interest
dfImp = pd.DataFrame(
    imputer.fit_transform(np.array(dfObsDum)),
    index=dfObsDum.index,
    columns=dfObsDum.columns,
)[c.years]

# Convert imputed biophysical indicator to ecological status
dfEcoImp, impStats = c.ecological_status(j, dfImp, dfVP, "imp", index)
dfEcoImp.describe()

# def missing_values_graph(self, j, frame, suffix="obs", index=None):
"""Heatmap visualizing observations of ecological status as either missing or using the EU index of ecological status, i.e., from 0-4 for Bad, Poor, Moderate, Good, and High water quality respectively.
Saves a figure of the heatmap."""
frame, suffix, index = dfEcoImp, "imp", index_sorted
if index is None:
    # Sort water bodies by number of missing values across years of interest
    df = frame.copy()
    df["nan"] = df.shape[1] - df.count(axis=1)
    df = df.sort_values(["nan"], ascending=False)[c.years]
    # Save index to reuse the order after imputing the missing values
    index = df.index
    # Specify heatmap to show missing values as gray
    colors = ["grey", "red", "orange", "yellow", "green", "blue"]
    uniqueValues = [-1, 0, 1, 2, 3, 4]
    description = "Missing value (gray), Bad (red), Poor (orange), Moderate (yellow), Good (green), High (blue)"
else:
    # Sort water bodies by number of missing values prior to imputation
    df = frame.copy().reindex(index)

    # Specify heatmap without missing values
    colors = ["red", "orange", "yellow", "green", "blue"]
    uniqueValues = [0, 1, 2, 3, 4]
    description = (
        "Bad (red), Poor (orange), Moderate (yellow), Good (green), High (blue)"
    )
# Plot heatmap
df.fillna(-1, inplace=True)
cm = sns.xkcd_palette(colors)
plt.figure(figsize=(12, 12))
ax = sns.heatmap(
    df,
    cmap=cm,
    cbar=False,
    cbar_kws={"ticks": uniqueValues},
)
ax.set(yticklabels=[])
plt.ylabel(
    str(len(df)) + " " + j + " ordered by number of missing values",
    fontsize=14,
)
plt.xlabel("")
plt.title(
    ("Ecological status of " + j + "\n" + description),
    fontsize=14,
)
plt.tight_layout()
plt.savefig(
    "output\\" + j + "_eco_" + suffix + ".pdf",
    bbox_inches="tight",
)


# def values_by_catchment_area(self, j, dfEcoImp, dfVP):
"""Assign water bodies to coastal catchment areas and calculate the weighted arithmetic mean of ecological status after truncating from above at Good status.
For each year t, set up df with variables for the Benefit Transfer equation."""
dfEcoImp, dfVP, frames_j = df_eco_imp, df_VP, {}

if j == "coastal":  #  ID shared between coastal waters and catchment areas
    dfEcoImpCatch = dfEcoImp.copy()

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
    dfEcoImpCatch = dfEcoImp.merge(dfCatch, on="wb").astype(int)

# Merge df for imputed ecological status w. shore length
dfEco = dfEcoImpCatch.merge(dfVP[["length"]], on="wb")  #  length

if j == "coastal":  # Coastal catchment area ID is the water body ID
    dfEco["v"] = dfEco.index

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

# For each year t, create a df of variables needed for benefit transfer
frames_t = {}  #  create empty dictionary to store a df for each year t

# DataFrame for water quality truncated from above at Good ecological status
Q = dfEco.copy()
Q[c.years] = Q[c.years].where(Q < 3, 3)

# DataFrames with dummy for less than good ecological status
SL = Q.copy()
SL[c.years] = SL[c.years].mask(SL < 3, 1).mask(SL >= 3, 0)

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
        df["ln PSL"] = SL_not_good[t] / Geo["shore all j"]  #  proportional
        ln_PSL = np.log(df.loc[df["ln PSL"] > 0, "ln PSL"])  #  log PSL
        ln_PSL_full = pd.Series(index=df.index)  #  empty series with index
        ln_PSL_full[df["ln PSL"] != 0] = ln_PSL  #  fill with ln_PSL if > 0
        df["ln PSL"] = np.where(df["ln PSL"] > 0, ln_PSL_full, df["ln PSL"])
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
stats.to_csv("output\\all_eco_imp_LessThanGood.csv")  #  save to csv

# Plot water bodies by category (mean ecological status weighted by length)
f1 = (
    stats.drop(1989)
    .plot(ylabel="Share of category with less than good ecological status")
    .get_figure()
)
f1.savefig("output\\all_eco_imp_LessThanGood.pdf", bbox_inches="tight")

########################################################################################
#   4.b Marginal willingness to pay (MWTP) for improvement of water quality to "Good"
########################################################################################
# Concatenate DataFrames for each category j ∈ {coastal, lakes, streams}
df_BT = pd.concat(frames_j)
df_BT.index.names = ["j", "t", "v"]
df_BT.to_csv("output\\all_eco_imp.csv")  #  save to csv
df_BT = pd.read_csv("output\\all_eco_imp.csv", index_col=[0, 1, 2]).sort_index()

# Costs of pollution in prices of current year, and preceding year respectively
CWP = c.valuation(df_BT, real=False)
CWP.to_csv("output\\all_cost.csv")  #  save to csv for chain linking

# Costs of pollution in real values (million DKK, 2018 prices)
RWP_v = c.valuation(df_BT)
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
IV = c.valuation(df_BT, real=False, investment=True)
IV.to_csv("output\\all_investment.csv")  #  save to csv for chain linking

# def valuation(self, dfBT, real=True, investment=False):
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
    d["Q"] = np.where(d["Q"] < 2.99, np.log(3 - d["Q"]), d["Q"])

else:
    # Actual change in ecological status since preceding year
    d = d.reorder_levels(["j", "v", "t"]).sort_index()
    d["Q"] = d["Q"].diff()
    d = d.reorder_levels(["j", "t", "v"]).sort_index()

    # MWTP = 0 if actual change in water quality is zero
    d["nonzero"] = np.select([d["Q"] != 0], [1])  #  dummy

    # Mark if actual change is negative (used to switch MWTP to negative)
    d["neg"] = np.select([d["Q"] < 0], [1])  #  dummy

    # Convert Q to the log of the actual change using nested np.where()
    d["Q"] = np.where(
        d["Q"] > 0,  #  condition
        np.log(d["Q"]),  #  if condition is met
        np.where(d["Q"] < 0, np.log(-d["Q"]), d["Q"]),  #  else
    )

# Drop year 1989 and specify integer values
d = d.drop(d[d.index.get_level_values("t") == 1989].index)
d[["D age", "D lake", "N"]] = d[["D age", "D lake", "N"]].astype(int)

# Consumer Price Index by year t (1990-2020)
CPI = pd.read_excel("data\\" + c.data["shared"][0], index_col=0)

# Merge data with CPI to correct for assumption of unitary income elasticity
kwargs = dict(how="left", left_index=True, right_index=True)
df1 = d.merge(CPI, **kwargs)
df1["unityMWTP"] = c.BT(df1)  #  MWTP assuming unitary income elasticity
df2018 = df1[df1.index.get_level_values("t") == 2018].copy()
df2018["elastMWTP"] = c.BT(df2018, elast=1.453)  #  meta study income ε
df2018["factor"] = df2018["elastMWTP"] / df2018["unityMWTP"]
df2018 = df2018.droplevel("t")
df2 = df1.merge(df2018[["factor"]], **kwargs)
df2 = df2.reorder_levels(["j", "t", "v"]).sort_index()
df2["MWTP"] = df2["unityMWTP"] * df2["factor"] * df2["nonzero"]

# Aggregate real MWTP per hh over households in coastal catchment area
df2["RWP"] = df2["MWTP"] * df2["N"] / 1e06  #  million DKK (2018 prices)

if investment is True:
    # Apply net present value (NPV) factor
    df2["CWP"] = df2["CWP"] * df2["NPV"]

    # Switch MWTP to negative if actual change is negative
    cond = [df2["neg"] == 1]
    df2["CWP"] = np.select(cond, [-df2["CWP"]], default=df2["CWP"])

    if real is True:
        df2["IV"] = df2["CWP"]  #  rename CWP to IV
        return df2[["IV"]]  #  return complete dataset

if real is True:  #  Real cost of water pollution (million DKK, 2018 prices)
    return df2[["CWP"]]  #  return complete dataset

# Aggregate nominal MWTP per hh over households in coastal catchment area
df2["CWPn"] = df2["CWP"] * df2["CPI"] / CPI.loc[2018, "CPI"]  #  million DKK

# CWP in the prices of the preceding year (for year-by-year chain linking)
df2["D"] = df2["CWPn"] * df2["CPI t-1"] / df2["CPI"]  #  million DKK

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
    # Rename CWP to IV  #  rename CWP nominal to IV nominal
    grouped.columns = grouped.columns.set_levels(["IVn", "D"], level=0)

########################################################################################
#   4.c Marginal willingness to pay (MWTP) for investments in water quality
########################################################################################
# Nominal Investment value of increase (decrease) in water quality
IVn = c.valuation(df_BT, real=False, investment=True)
IVn.to_csv("output\\all_investment.csv")  #  save to csv for chain linking

# Investment value in real terms (million DKK, 2018 prices)
IV_v = c.valuation(df_BT, investment=True)
IV = IV_v.groupby(["j", "t"]).sum().unstack(level=0).rename_axis(None)  #  sum over v
IV.rename_axis([None, None], axis=1).to_csv("output\\all_investment_real.csv")
f2 = (
    IV.loc[:, "IV"]
    .rename_axis(None, axis=1)
    .plot(
        kind="bar",
        ylabel="Investment in water quality improvement (million DKK, 2018 prices)",
    )
    .get_figure()
)
f2.savefig("output\\all_investment_real.pdf", bbox_inches="tight")


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
kwargs = dict(method="index", fill_value="extrapolate", limit_direction="both")
df.interpolate(**kwargs, inplace=True)

# Save to CSV
df.to_csv("output\\" + "D_age.csv")
