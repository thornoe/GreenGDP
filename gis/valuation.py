"""
Name:       valuation.py

Label:      Valuing the cost of less-than-good ecological status using benefit transfer.

Summary:    ThorNoe.GitHub.io/GreenGDP explains the overall approach and methodology.

Usage:      This is a sandbox to test out code blocks before joining them in script.py, 
            which supports WaterbodiesScriptTool in the gis.tbx toolbox.
            See GitHub.com/ThorNoe/GreenGDP for instructions to run or update it all.
waterbodies
License:    MIT Copyright (c) 2023
Author:     Thor Donsby Noe
"""

###############################################################################
#   0. Imports                                                                #
###############################################################################
import os

import pandas as pd

###############################################################################
#   1. Setup                                                                  #
###############################################################################
os.chdir(r"C:\Users\au687527\GitHub\GreenGDP\gis")

# Read wide format for ecological status of waterbodies
df = pd.read_csv("data/demographics.csv").drop("Unnamed: 0", axis=1)

# Years in the dataset
year_first = 1990
year_last = 2018
years = list(range(year_first, year_last + 1))
years

###############################################################################
#   2. First look at the data                                                 #
###############################################################################
df.describe()
df.tail()
df["Kystom_2ID"].nunique()  # 90 coastal catchment areas

# Yearly totals of inhabitants, households, and household size
for y in years:
    if y == year_first:
        print("year persons househ. avg. household size")
    df_year = df.loc[df["aar"] == y]
    print(
        y,
        df_year["sum_pers"].sum(),
        df_year["sum_hust"].sum(),
        round(df_year["sum_pers"].sum() / df_year["sum_hust"].sum(), 2),
    )
# Sammenlignet med DST (hhv. tabel BEFOLK1 og FAM55N)
print(
    5135409 - 4999453,
    "færre personer end BEFOLK1 i 1990\n",
    5781190 - 5747469,
    "færre personer end BEFOLK1 i 2018\n",
    2278293 - 2219886,
    "færre husholdninger end FAM55N i 1990\n",
    2688472 - 2681529,
    "færre husholdninger end FAM55N i 2018\n",
    2705614 - 2688472,
    "flere husholdninger i VP3-kørsels end FAM55N i 2018",
)
5020718 - 4999453

# Households in coastal catchment areas
# df.drop(columns=['alder', 'ind', 'gen_alder', 'gen_ind'], inplace=True)
sum(df["sum_hust"] < 1000) / len(df)  # share with <= 1000 households
sum(df["sum_hust"] < 5000) / len(df)  # share with <= 5000 households

df.loc[df["sum_pers"] < 5]
df.loc[df["sum_hust"] < 20]
df.loc[(df["sum_hust"] > 20) & (df["sum_hust"] < 100)]
df.loc[df["Kystom_2ID"] == 18000061]  # 55 km vestkyst fra Nymindegab til Sønder Nissum
df.loc[df["Kystom_2ID"] == 18000062]  # Kystvandopland Ringkøbing Fjord
df.loc[df["Kystom_2ID"] == 14000062]  # 50 km vestkyst fra Sønder Nissum til Thyborøn
df.loc[df["Kystom_2ID"] == 11000084]  # 200 km vestkyst fra Agger Tange til Grenen
df.loc[df["Kystom_2ID"] == 114000047]  # Fyns og Langelands østkyst
df.loc[df["Kystom_2ID"] == 113000046]  # Odense Fjord
df.loc[df["Kystom_2ID"] == 23000074]  # Sjællands nordkyst fra Hundested til Helsingør
df.loc[df["Kystom_2ID"] == 2300003]  # Øresund fra Helsingør til Dragør

###############################################################################
#   2. First look at the data                                                 #
###############################################################################
