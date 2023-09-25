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

###############################################################################
#   2. First look at the data                                                 #
###############################################################################
df.describe()
df.tail()
df["Kystom_2ID"].nunique()  # 90 coastal catchment areas
df["Kystom_2ID"].unique()
df.loc[df["sum_pers"] < 5]
