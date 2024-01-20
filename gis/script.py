'''
Name:       script.py

Label:      Construct and map longitudinal data of ecological status of water bodies.

Summary:    ThorNoe.GitHub.io/GreenGDP explains the overall approach and methodology.

Rqmts:      ArcGIS Pro must be installed on the system and be up to date.

Usage:      This script supports WaterbodiesScriptTool in the gis.tbx toolbox.
            See GitHub.com/ThorNoe/GreenGDP for instructions to run or update it all.

License:    MIT Copyright (c) 2020-2024
Author:     Thor Donsby Noe
'''
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
# os.chdir(arcpy.GetParameterAsText(0))
root = r'C:\Users\au687527\GitHub\GreenGDP'
path = root + '\\gis'
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
# Span of natural capital account (1990 investment value depends on change from 1989)
year_first = 1989
year_last = 2020

# Specify the names of each type of water body and its data files
data = {
    'catch': ['demographics.csv', 'geographical.xlsx'],
    'streams': ['streams_DVFI.xlsx', 'streams_1988-2020.xlsx'],
}

# Specify the names of the corresponding linkage files
linkage = {
    'coastal': 'coastal_stations_VP3.csv',
    'lakes': 'lakes_stations_VP3.csv',
    'streams': 'streams_stations_VP3.csv',
}

# WFS service URL for the current water body plan (VP2 is for 2015-2021)
wfs_service = 'https://wfs2-miljoegis.mim.dk/vp3endelig2022/ows?service=WFS&request=Getcapabilities'

# For the WFS, specify the name of the feature class (fc) for each type of water body
wfs_fc = {
    'catch': 'vp3e2022_kystvand_opland_afg',
    'coastal': 'vp3e2022_marin_samlet_1mil',
    'lakes': 'vp3e2022_soe_samlet',
    'streams': 'vp3e2022_vandloeb_samlet',
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

# Create an empty DataFrame to store weighted ecological status by catchment area
df_eco_catch = pd.DataFrame()

# Loop over each category j ∈ {coastal, lakes, streams}
for j in ['streams']:
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
    df_eco_imp, imp_stats = c.impute_missing(j, df_ind_obs, df_VP, 'imp', index_sorted)

    # Save time series of total shares (weighted by length) of quality
    # df.to_csv('output\\' + j + '_eco_imp_catch.csv')

    # Assign
    # Clean up after each iteration of loop
    if keep_gdb != 'true':
        # Delete feature class
        if arcpy.Exists(j):
            arcpy.Delete_management(j)

# Clean up geodatabase
if keep_gdb != 'true':
    # Delete all feature classes in geodatabase
    for fc in arcpy.ListFeatureClasses():
        arcpy.Delete_management(fc)

########################################################################################
#   4. Run the functions line-by-line
########################################################################################
