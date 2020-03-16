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
# Specify the list of data files
filenames_data = ['streams_DVFI_1990-2018.xlsx','streams_DVFI_2019-.xlsx']

# Specify the list of linkage tables
filenames_linkage = ['streams_stations_VP2.xlsx']

dict = {'data':filenames_data,
        'linkage':filenames_linkage}

dict


###############################################################################
#   3. Import module and run the two functions                                #
###############################################################################

# Import the module with all the homemade functions
import script_module

for key, filenames in dict.items():
    script_module.get_data(key, filenames)









os.chdir('C:/Users/jwz766/Documents/GitHub/gnnp') # one level up

# Eelgrass
eg = pd.read_excel('sea/Hav_Vegetation_Paravane træk_1998-2009.xlsx')

eg.head(1)


wf = pd.read_excel('streams/DFI_1970-2000.xlsx')

wf['Indekstype'].value_counts()



### Danish Water Fauna Index (DVFI) only
wf = wf[wf['Indekstype'].str.contains('DVFI')]

wf['År'] = wf['Dato'].astype(str).str.slice(0,4).astype(int)
wf = wf[['KommuneNavn', 'Lokalitetsnavn', 'MC-stationsnr', 'Indekstype', 'Indeks', 'År']]
wf.head(1)

år, antal, kommuner = [], [], []

for y in range(1990,2001):
    df = wf[wf['År']==y]
    l = df['KommuneNavn'].value_counts().index.to_list()
    l.sort()
    år.append(y)
    antal.append(len(l))
    kommuner.append(', '.join([str(elem) for elem in l]))

pd.DataFrame({'År':år, 'Antal kommuner':antal, 'Kommuner':kommuner})

wf['Lokalitetsnavn'].value_counts().nlargest(50)
wf['Lokalitetsnavn'].value_counts().nsmallest(50)

wf['MC-stationsnr'].value_counts().nlargest(50)
wf['MC-stationsnr'].value_counts().nsmallest(50)
