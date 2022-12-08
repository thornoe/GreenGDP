"""
Name:       script.py

Label:      Construct and map longitudinal data of ecological status of streams.

Summary:    ThorNoe.GitHub.io/GreenGDP explains the approach and methodology.

Rqmts:      ArcGIS Pro must be installed on the system and be up to date.

Usage:      This script supports WaterbodiesScriptTool in the gis.tbx toolbox.
            See GitHub.com/ThorNoe/GreenGDP for instructions to run or update it all.

Created:    25/03/2020
Author:     Thor Donsby Noe
"""

###############################################################################
#   0. Imports                                                                #
###############################################################################
# Import Operation System (os) and ArcPy package (requires ArcGIS Pro installed)
import os
import arcpy
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
arcpy.env.overwriteOutput = True    # set overwrite option

###############################################################################
#   1. Setup                                                                  #
###############################################################################
# Specify the parent folder as the working directory of the operating system
# os.chdir(arcpy.GetParameterAsText(0))
root = r'C:\Users\au687527\GitHub\GreenGDP'
path = root + '\\gis'
arcpy.env.workspace = path
os.chdir(path)

# Specify whether to keep the geodatabase when the script finishes
# keep_gdb = arcpy.GetParameterAsText(1)
keep_gdb = 0


###############################################################################
#   2. Specifications (update for VP3 in primo 2022)                          #
###############################################################################
# Specify the names of each type of water body and its data files
data = {'streams':['streams_DVFI_1992-2018.xlsx','streams_DVFI_2019-.xlsx']}

# Specify the names of the corresponding linkage files
linkage = {'streams':'streams_stations_VP2.xlsx'}

# WFS service for the current water body plan (VP2 is for 2015-2021)
wfs_service = 'http://wfs2-miljoegis.mim.dk/vp2_2016/ows?service=wfs&version=1.1.0&request=GetCapabilities'

# Specify the names of the feature class (fc) for each type of water body
wfs_fc = {'streams':'vp2bek_2019_vandlob'}

# Specify the field (column) names in the fc that contain the water body ID
wfs_vpID = {'streams':'g_del_cd'}

# Specify the field (column) names in the fc that contain the water body size
wfs_size = {'streams':'g_len'}

# Specify the field (column) names in the fc that contain the water body type
wfs_typology = {'streams':'f_vl_typo'}


###############################################################################
#   3. Import module and run the functions                                    #
###############################################################################
# Import the module with all the homemade functions
import script_module

# Initialize the class for all data processing and mapping functions
c = script_module.Water_Quality(data, linkage, wfs_fc, wfs_vpID, wfs_size, wfs_typology, keep_gdb)

# Check that the folders with data and linkage files exist or create them
c.get_data()

# Loop over each type of water body (to be extended with lakes and coastal waters)
for waterbodyType in data:

    # Get the feature class from the WFS service
    c.get_fc_from_WFS(waterbodyType, wfs_service)

    # Create a Pandas DataFrame with observed indicator by year
    df_ind, years = c.observations(waterbodyType)

    # Report observed ecological status by year
        # include heatmap
    df_obs = c.observed_ecological_status(waterbodyType, df_ind, years)

    if waterbodyType == 'streams':
        # Create a map book with yearly maps of observed ecological status
        c.map_book(waterbodyType, df_obs, years)

    # Impute missing observations




            # Save time series to excel sheet for valuation (update)
            # df.to_csv('output\\' + waterbodyType + '_ecological_status.csv')

    # Delete geodatabase




# df_ind.describe()


# Import ArcPy package (requires ArcGIS Pro installed) and set workspace
import arcpy
path = root + r'\gis'
arcpy.env.workspace = path
arcpy.env.overwriteOutput = True    # set overwrite option

# WFS service
WFS_Service = 'http://wfs2-miljoegis.mim.dk/vp2_2016/ows?service=wfs&version=1.1.0&request=GetCapabilities'

# Name of the input layer to extract
WFS_FeatureType = 'vp2bek_2019_vandlob'

# Path of the geodatabase (must preexist)
arcPath = path + '\\gis.gdb'

# Name of the output feature class
fc = 'streams'

if not arcpy.Exists(fc):
    # Execute the WFSToFeatureClass tool to download the feature class.
    arcpy.conversion.WFSToFeatureClass(WFS_Service, WFS_FeatureType,
                                       arcPath, fc, max_features=15000)

# Set local variables
WFS_Service = "http://sampleserver6.arcgisonline.com/arcgis/services/SampleWorldCities/MapServer/WFSServer?request=GetCapabilities&service=WFS"
WFS_FeatureType = "Cities"
fc = "SampleWorldCities"

# Execute the WFSToFeatureClass tool
arcpy.WFSToFeatureClass_conversion(WFS_Service, WFS_FeatureType, Out_Location, Out_Name)

