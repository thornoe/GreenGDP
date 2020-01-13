# Imports
import numpy as np
import pandas as pd
import os

# Work pc
# os.chdir('C:/Users/jwz766/Documents/GitHub/gnnp') # one level up
# Home pc
os.chdir('C:/Users/thorn/OneDrive/Dokumenter/GitHub/gnnp') # one level up

# Danish Water Fauna Index (DVFI)
wf = pd.read_excel('streams/DFI_1970-2000.xlsx')

wf.drop(columns=['MiljøcenterNavn', 'RegionNavn'], leve=1, inplace=True)
wf.head(1)

len(wf['MarinReferenceKode'].value_counts())
len(wf['Farvandsområde'].value_counts())
len(wf['Lokalitetsnavn'].value_counts())
len(wf['ObservationsStedNavn'].value_counts())
len(wf['ObservationsStedNr'].value_counts())
len(wf['MC-stationsnr'].value_counts())



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

eg['Beskrivelse'].value_counts()
