###############################################################################
#   0. Imports                                                                #
###############################################################################
# Import packages
import os
import pandas as pd
import numpy as np
import arcpy, sys, traceback
arcpy.env.overwriteOutput = True    # set overwrite option


###############################################################################
#   1. Setup                                                                  #
###############################################################################
# Specify the parent folder as the working directory of the operating system
path = r'C:\Users\jwz766\Documents\GitHub\gnnp\gis'
os.chdir(path)


###############################################################################
#   2. Specifications (update for VP3 in primo 2022)                          #
###############################################################################
# Specify the names of each type of water body and its data files
data = {'streams':['streams_DVFI_1992-2018.xlsx','streams_DVFI_2019-.xlsx']}

# Specify the names of the corresponding linkage files
linkage = {'streams':'streams_stations_VP2.xlsx'}

# WFS service for the current water body plan (VP2 is for 2015-2021)
wfs_service = 'http://wfs2-miljoegis.mim.dk/vp2_2016/ows?version=1.1.0&request=GetCapabilities&service=wfs'

# Specify the names of the feature class (fc) for each type of water body
wfs_fc = {'streams':'vp2bek_2019_vandlob'}

# Specify the field (column) names in the fc that contain the water body ID
wfs_vpID = {'streams':'g_del_cd'}

# Specify the field (column) names in the fc that contain the water body size
wfs_size = {'streams':'g_len'}


###############################################################################
#   3. Import module and run the functions                                    #
###############################################################################
# Import the module with all the homemade functions
import script_module

# Initialize the class for all data processing and mapping functions
c = script_module.Water_Quality(data, linkage, wfs_fc, wfs_vpID, wfs_size)

## Check that the folders with data and linkage files exist or create them
#c.get_data()
#
# For each type of water body (code only covers streams so far)
#for waterbodyType in data:
#
#    # Get the feature class from the WFS service
#    c.get_fc_from_WFS(waterbodyType, wfs_service)
#
#    # Create a Pandas DataFrame with ecological status by year
#    df, years = c.ecological_status(waterbodyType)
#
#    # Create a map book with yearly maps of ecological status
#    c.map_book(waterbodyType, df, years)


waterbodyType = 'streams'
c.get_fc_from_WFS(waterbodyType, wfs_service)


# Create Pandas DataFrame with ecological status for each water body by year
df, years = c.ecological_status(waterbodyType)

# Calculate total size of all water bodies in current water body plan (VP2)
#####################size = self.wfs_size[waterbodyType]
size = wfs_size[waterbodyType]
totalSize = df[size].sum()

# Create an empty df for statistics
stats = pd.DataFrame(index=['Status known (%)',
                            'Share of known is high (%)',
                            'Share of known is good (%)',
                            'Share of known is moderate (%)',
                            'Share of known is poor (%)',
                            'Share of known is bad (%)'])

# Calculate the above statistics for each year
for i in years:
    y = df[[size, i]].reset_index(drop=True)
    y['Known'] = np.select([y[i].notna()], [y[size]])
    y['High'] = np.select([y[i]==5], [y[size]])
    y['Good'] = np.select([y[i]==4], [y[size]])
    y['Moderate'] = np.select([y[i]==3], [y[size]])
    y['Poor'] = np.select([y[i]==2], [y[size]])
    y['Bad'] = np.select([y[i]==1], [y[size]])
    
    # Add shares of total size to stats
    knownSize = y['Known'].sum()
    stats[i] = [100*knownSize/totalSize,
                100*y['High'].sum()/knownSize,
                100*y['Good'].sum()/knownSize,
                100*y['Moderate'].sum()/knownSize,
                100*y['Poor'].sum()/knownSize,
                100*y['Bad'].sum()/knownSize]

# Convert statistics to integers
stats = stats.astype(int)

stats.to_html('data\\' + waterbodyType + '_stats.md')


# Create the map book
#c.map_book(waterbodyType, df, years)
