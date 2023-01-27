"""
Name:       script.py

Label:      Construct and map longitudinal data of ecological status of streams.

Summary:    ThorNoe.GitHub.io/GreenGDP explains the approach and methodology.

Rqmts:      ArcGIS Pro must be installed on the system and be up to date.

Usage:      This script supports WaterbodiesScriptTool in the gis.tbx toolbox.
            See GitHub.com/ThorNoe/GreenGDP for instructions to run or update it all.

Licence:    MIT Copyright (c) 2020-2023
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
arcpy.env.overwriteOutput = True # set overwrite option

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
keep_gdb = 1


###############################################################################
#   2. Specifications (update for VP3 in primo 2022)                          #
###############################################################################
# Specify the years of interest
year_first = 1989
year_last  = 2020

# Specify the names of each type of water body and its data files
data = {'streams':'streams_DVFI.xlsx'}

# Specify the names of the corresponding linkage files
linkage = {'streams':'streams_stations_VP2.xlsx'}

# WFS service URL for the current water body plan (VP2 is for 2015-2021)
wfs_service = 'http://wfs2-miljoegis.mim.dk/vp2_2016/ows?service=wfs&version=1.1.0&request=GetCapabilities'

# Specify the name of the feature class (fc) for each type of water body
wfs_fc = {'streams':'vp2bek_2019_vandlob',
          'lakes'  :'theme_vp2_2016_soer',
          'coastal':'theme_vp2_2016_kystvande',
          'catch'  :'theme_vp2_2016nbel12_deloplande'}

# Specify the name of the field (column) in fc that contains the ID of the water body
wfs_vpID = {'streams':'g_del_cd',
            'lakes'  :'id',
            'coastal':'vandomrid',
            'catch'  :'kystom_2id'}

# Specify the name of the field (column) in fc that contains the typology of the water body
wfs_typo = {'streams':'f_vl_typo',
            'lakes'  :'typologi',
            'coastal':'typologi'}


###############################################################################
#   3. Import module and run the functions                                    #
###############################################################################
# Import the module with all the homemade functions
import script_module

# Initialize the class for all data processing and mapping functions
c = script_module.Water_Quality(year_first, year_last, data, linkage, 
                                wfs_service, wfs_fc, wfs_vpID, wfs_typo, keep_gdb)

# Loop over each type of water body (to be extended with lakes and coastal waters)
for waterbodyType in data:

    # Get the feature class from the WFS service
    c.get_fc_from_WFS(waterbodyType)

    # Create a Pandas DataFrame with observed indicator by year
    df_ind_obs = c.observed_indicator(waterbodyType)

    # Report observed ecological status by year
        # ADD HEATMAP
    df_eco_obs, stats = c.observed_ecological_status(waterbodyType, df_ind_obs)

    # if waterbodyType == 'streams':
    #     # Create a map book with yearly maps of observed ecological status
    #     c.map_book(waterbodyType, df_obs, years)

    # Impute missing observations


    # 

    # Save time series of total shares (weighted by length) of quality
        # df.to_csv('output\\' + waterbodyType + '_ecological_status.csv')

    # Assign 

    # Delete geodatabase

# finally: 
    # Clean up all feature classes
    for fc in arcpy.ListFeatureClasses():
        arcpy.Delete_management(fc)
    

# List feature classes and their respective fields
for fc in arcpy.ListFeatureClasses():
    print('\n'+fc)
    print(arcpy.GetCount_management(fc))
    for field in arcpy.ListFields(fc):
        print(field.name)

### GET AVERAGE OF DUMMY FOR AGE>45 BY CATHCMENT AREA
df = pd.read_csv('data\\' + 'demographics.csv', sep=';') # 1990-2018

# Select relevant columns
df = df[['Kystvnd', 'aar', 'alder', 'antal_pers']]
df['flat'] = 1

# Dummy for mean age > 45 years for a given catchment area and year
df['D_age'] = np.select([df['alder']>45], [1])

# Average share of catchment areas with mean age > 45 years weighted by no. persons
df = pd.DataFrame(df.groupby(['aar']).apply(lambda x: np.average(x['D_age'], weights=x['antal_pers'])))

# Extrapolate using a linear trend
df = df.append(pd.DataFrame([np.nan, np.nan, np.nan], index=[1989, 2019, 2020])).sort_index()
kw = dict(method="index", fill_value="extrapolate", limit_direction="both")
df.interpolate(**kw, inplace=True)

# Save to CSV
df.to_csv('output\\'+'D_age.csv')
