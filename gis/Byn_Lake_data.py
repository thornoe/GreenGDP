"""
Name:       Byn_Lake_data.py

Label:      Set up data for valuating ecological status of Byn Lake by Nissum Fjord.

Summary:    ThorNoe.GitHub.io/GreenGDP explains the overall approach and methodology.

Rqmts:      ArcGIS Pro must be installed on the system and be up to date.

Usage:      This script draws on script_module.py to set up data for valuation of an 
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
j = "lakes"
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

# Get the feature class from the WFS service
c.get_fc_from_WFS(j)

# Read DataFrame for waterbody characteristics
df_VP = pd.read_csv("output\\" + j + "_VP.csv", index_col="wb")

# Read ecological status based on observed biophysical indicator
df_eco_obs = pd.read_csv("output\\" + j + "_eco_obs.csv", index_col="wb")
df_eco_obs.columns = df_eco_obs.columns.astype(int)

# Set ecological status to Good globally
df_eco_obs.loc[:, :] = 3

# Set ecological status to Moderate for Byn Lake
df_eco_obs.loc[424, :] = 2

# Set up df with variables by coastal catchment area for the Benefit Transfer equation
frames_j[j], shores_j[j] = c.values_by_catchment_area(j, df_eco_obs, df_VP)

# Concatenate DataFrames for category j = lakes
df_BT = pd.concat(frames_j)

# Limit DataFrames to catchment area v = 'Nissum Fjord, ydre'
df_BT_Byn = df_BT[df_BT.index.get_level_values("v") == 129]
df_BT_Byn.index.names = ["j", "t", "v"]

# Save variables for benefit transfer
df_BT_Byn.to_csv("output\\Byn_Lake_variables.csv")
