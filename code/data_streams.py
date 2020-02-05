# Imports
import pandas as pd
import os

### Work pc:
os.chdir('C:/Users/jwz766/Documents/GitHub/gnnp') # one level up
### Home pc:
# os.chdir('C:/Users/thorn/OneDrive/Dokumenter/GitHub/gnnp') # one level up

### Read data and lingage table
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
