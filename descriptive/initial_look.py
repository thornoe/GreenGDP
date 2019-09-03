# Imports
import numpy as np
import pandas as pd
import os

os.chdir('C:/Users/jwz766/Documents/GitHub/gnnp') # one level up

# Eelgrass
eg = pd.read_excel('data/Hav_Vegetation_Paravane træk_2018-19.xlsx')

eg.head(1)

eg['MarinReferenceKode'].value_counts().count()
eg['Farvandsområde'].value_counts().count()
eg['Lokalitetsnavn'].value_counts().count()
eg['ObservationsStedNr'].value_counts().describe()
eg['MC-stationsnr'].value_counts().count()
eg['StartDato'].value_counts().count()
eg['Kvalitet'].value_counts().count()
eg['Kontrol'].value_counts()
eg['Kvalitetsnote'].value_counts().count()
eg['Navn'].value_counts().count()
eg['Beskrivelse'].value_counts().count()
eg['Dækningsgrad i %'].value_counts().count()



eg[eg['Lokalitetsnavn']=='Hevring Bugt']['Farvandsområde'].value_counts()
eg[eg['Lokalitetsnavn']=='Hevring Bugt']['ObservationsStedNr'].value_counts()
eg[eg['Lokalitetsnavn']=='Hevring Bugt']['Farvandsområde'].value_counts()


eg['Lokalitetsnavn'].value_counts().nlargest(40)
eg['Lokalitetsnavn'].value_counts().nsmallest(41)
