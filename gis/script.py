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

# Initialize the class for all data processing and mapping functions
c = script_module.dataClass(data, linkage)

# Check that the folders with data and linkage files exist or create them
c.get_data()

# Create a longitudinal DataFrame for stations in streams
long, years = c.longitudinal('streams')
long.tail(1)

### Read the linkage table
waterbodyType = 'streams'

link = pd.read_excel('linkage\\' + linkage[waterbodyType])



### Link station numbers with stream IDs
