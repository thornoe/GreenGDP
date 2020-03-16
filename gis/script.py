###############################################################################
#   0. Imports                                                                #
###############################################################################
# Import packages
import os, sys
import pandas as pd
# import arcpy
# arcpy.env.overwriteOutput = True    # set overwrite option


###############################################################################
#   1. Setup                                                                  #
###############################################################################
# Specify the working folder as the ArcPy workspace and OS working directory
path = r'C:\Users\jwz766\Documents\GitHub\gnnp\gis'
os.chdir(path)
# arcpy.env.workspace = path


###############################################################################
#   2. Specifications                                                         #
###############################################################################
# Specify the names of the data files
data = {'streams':['streams_DVFI_1990-2018.xlsx','streams_DVFI_2019-.xlsx']}

# Specify the names of the linkage files
linkage = {'streams':'streams_stations_VP2.xlsx'}


###############################################################################
#   3. Import module and run the functions                                    #
###############################################################################
# Import the module with all the homemade functions
import script_module

# Dictionary for all data and linkage files
allFiles = {'data': [y for x in list(data.values()) for y in x],
            'linkage': [y for y in list(linkage.values())]}

# Check that the folders with data and linkage files exist or create them
for key, filenames in allFiles.items():
    script_module.get_data(key, filenames)

# Read data and set up longitudinal data



key = 'streams'
filenames = data['streams']

df   = pd.read_excel('data\\' + data[0])
df2  = pd.read_excel('data\\' + data[1])
link =


d = pd.Series([0,3])
d.median()
