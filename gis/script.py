"""
Name:       script.py

Label:      Construct and map longitudinal data of ecological status of water bodies.

Summary:    ThorNoe.GitHub.io/GreenGDP explains the overall approach and methodology.

Rqmts:      ArcGIS Pro must be installed on the system and be up to date.

Usage:      This script supports WaterbodiesScriptTool in the gis.tbx toolbox.
            See GitHub.com/ThorNoe/GreenGDP for instructions to run or update it all.

License:    MIT Copyright (c) 2020-2024
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
# keep_gdb = arcpy.GetParameterAsText(2)
keep_gdb = 1

########################################################################################
#   2. Specifications
########################################################################################
# Specify the years of interest
year_first = 1988
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

# Loop over each type of water body (to be extended with lakes and coastal waters)
for waterbodyType in ["streams"]:
    # Get the feature class from the WFS service
    c.get_fc_from_WFS(waterbodyType)

    # Create a Pandas DataFrame with observed indicator by year
    df_ind_obs, df_VP = c.observed_indicator(waterbodyType)

    # Report observed ecological status by year
    df_eco_obs, obs_stats = c.ecological_status(waterbodyType, df_ind_obs, df_VP)

    # if waterbodyType == "streams":
    #     # Create a map book with yearly maps of observed ecological status
    #     c.map_book(waterbodyType, df_eco_obs)

    # Impute missing observations


    # Save time series of total shares (weighted by length) of quality
    # df.to_csv('output\\' + waterbodyType + '_eco_imp_catch.csv')

    # Assign
    # Clean up after each iteration of loop
    if keep_gdb != "true":
        # Delete feature class
        if arcpy.Exists(waterbodyType):
            arcpy.Delete_management(waterbodyType)

# Clean up geodatabase
if keep_gdb != "true":
    # Delete all feature classes in geodatabase
    for fc in arcpy.ListFeatureClasses():
        arcpy.Delete_management(fc)

########################################################################################
#   4. Sandbox
########################################################################################


df_ind_obs.head()  # equals df? I.e. changes to df_eco_obs after the next line is run?
df_ind_obs.describe()  # equals df? I.e. changes to df_eco_obs after the next line is run?

# Drop fields not used for imputation
df_eco_obs.describe()
observed = df_eco_obs[["length"]].merge(
    df_eco_obs.drop(columns=["length"]).dropna(how="all"),
    how="inner",
    left_index=True,
    right_index=True,
)
observed.shape

# List feature classes and their respective fields
for fc in arcpy.ListFeatureClasses():
    print("\n" + fc)
    print(arcpy.GetCount_management(fc))
    for field in arcpy.ListFields(fc):
        print(field.name)

years = list(range(year_first, year_last + 1))

# Sort by number of missing values
df = df_eco_obs.copy()
df["nan"] = df.shape[1] - df.count(axis=1)
df = df.sort_values(["nan"], ascending=False)[years]
arcpy.ListFeatureClasses()


def mvg(frame, waterbodyType, suffix):
    df = frame.copy()
    df.fillna(0, inplace=True)
    cm = sns.xkcd_palette(["grey", "red", "orange", "yellow", "green", "blue"])
    plt.figure(figsize=(12, 7.4))
    ax = sns.heatmap(df, cmap=cm, cbar=False, cbar_kws={"ticks": [0, 1, 2, 3, 4, 5, 6]})
    ax.set(yticklabels=[])
    plt.ylabel(waterbodyType + " (N=" + str(len(df)) + ")", fontsize=14)
    plt.xlabel("")
    plt.title(
        (
            "Ecological status of "
            + waterbodyType
            + ":"
            + "\nmissing value (grey), bad (red), poor (orange), moderate (yellow), good (green), high (blue)"
        ),
        fontsize=14,
    )
    plt.tight_layout()
    plt.savefig(
        "output/" + waterbodyType + "_eco_" + suffix + ".png", bbox_inches="tight"
    )
    plt.show()


mvg(df, "streams", "missing")


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

# Spatial Join water bodies to coastal catchment areas
arcpy.SpatialJoin_analysis(
    "catch",
    waterbodyType,
    fcJoined,
    "JOIN_ONE_TO_MANY",
    "KEEP_ALL",
    match_option="INTERSECT",
)
