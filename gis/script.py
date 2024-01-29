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

# Loop over each category j ∈ {coastal, lakes, streams}
for j in ["lakes", "streams"]:  # linkage.keys():
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
    df_eco_imp, stats_j[j] = c.impute_missing(j, df_ind_obs, df_VP, index_sorted)

    # df with variables by coastal catchment area for the Benefit Transfer equation
    frames_j[j], shores_j[j] = c.values_by_catchment_area(j, df_eco_imp, df_VP)

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

########################################################################################
#   4.c Marginal willingness to pay (MWTP) for investments in water quality
########################################################################################
# Investment value of increase (decrease) in water quality
IV = c.valuation(df_BT, real=False, investment=True)
IV.to_csv("output\\all_investment.csv")  #  save to csv for chain linking


########################################################################################
#   5. Decompose growth by holding everything else equal
########################################################################################


########################################################################################
#   6. Robustness check: Treat DK as a single catchment area
########################################################################################
