"""
Name:       CV.py

Label:      Report results of CV of imputation of missing values for surface waters.

Summary:    ThorNoe.GitHub.io/GreenGDP explains the overall approach and methodology.

Usage:      This is a standalone script that only serves to evaluates the robustness of 
            the imputation method coded up in script_module.py and applied by script.py, 
            which supports WaterbodiesScriptTool in the gis.tbx toolbox.
            See GitHub.com/ThorNoe/GreenGDP for instructions to run or update it all.

License:    MIT Copyright (c) 2024
Author:     Thor Donsby Noe
"""

########################################################################################
#   0. Functions
########################################################################################
import os

import pandas as pd

########################################################################################
#   1. Data setup
########################################################################################
# Create an empty DataFrame with an index for "number of dummies" from 0 to 8
model = ["No dummies", "1 dummy"] + [f"{i} dummies" for i in range(2, 9)]
scores = pd.DataFrame(index=model)
status = pd.DataFrame(index=["Observed"] + model)


# Add accuracy scores and imputed ecological status (share < GES) to the DataFrames
j = "lakes"  #  coastal waters
a = "status"
b = "LessThanGood"  #  accuracy score
scores, status = {}, {}  #  create empty dictionaries for accuracy scores and eco status
for j in ("coastal", "lakes", "streams"):
    for a, b, c in zip(d.keys(), ["accuracy", "LessThanGood"]):
        print(j, b)
        path = f"output/{j}_eco_imp_{b}"
        data = pd.read_csv(f"{path}.csv", index_col=0).iloc[-1, :]
        d[a][j] = data.iloc[1:].reset_index(drop=True)
        d[a].loc["n", j] = data.iloc[0]  #  number of observations
        if j != "coastal":
            df = pd.read_csv(f"{path}_sparse.csv", index_col=0).iloc[-1, :]
            d[a][f"{j} sparse"] = df.iloc[1:].reset_index(drop=True)
            d[a].loc["n", f"{j} sparse"] = df.iloc[0]  #  number of observations

pd.read_csv("output/coastal_eco_imp_LessThanGood.csv", index_col=0).iloc[-1, 1:]

# Accuracy score by year and selected predictors
scores.index = scores.index.astype(str)  #  convert index to string (to mimic read_csv)
sco = scores.drop(columns="n").drop(["1989", "Total"])  #  subset to relevant years
f1 = (  #  bar plot accuracy scores
    sco.plot(kind="bar", ylabel="Accuracy in predicting observed ecological status")
    .legend(loc="lower left")
    .get_figure()
)
f1.savefig("output/coastal_eco_imp_accuracy.pdf", bbox_inches="tight")  #  save PDF

# Share of streams with less than good ecological status by year and selected predictors
status.index = status.index.astype(str)  #  convert index to string (to mimic read_csv)
status_years = status.drop(["1989", "Total"])  #  cover years in natural capital account
imp = status_years.drop(columns=["n", "Obs"])  #  imputed status by selected predictors
obs = status_years[["Obs"]]  #  ecological status of streams observed the given year
obs.columns = ["Observed"]  #  rename 'Obs' to 'Observed'
sta = imp.merge(obs, left_index=True, right_index=True)  #  add Observed as last column
f2 = sta.plot(  # Plot share of coastal waters with less than good ecological status
    ylabel="Share of coastal waters with less than good ecological status"
).get_figure()
f2.savefig("output/coastal_eco_imp_LessThanGood.pdf", bbox_inches="tight")  #  save PDF


# Specify the working directory of the operating system
os.chdir(r"C:\Users\au687527\GitHub\GreenGDP\gis")

# Limit LOO-CV to loop over years used directly for the natural capital account
years = list(range(1990, 2020 + 1))

# Read DataFrames for observed ecological status and typology
dfEcoObs = pd.read_csv("output/coastal_eco_obs.csv", index_col="wb")
dfEcoObs.columns = dfEcoObs.columns.astype(int)
dfVP = pd.read_csv("output\\coastal_VP.csv", index_col="wb")

# Share of waterbodies by number of non-missing values
for n in range(0, len(dfEcoObs.columns) + 1):
    n, round(100 * sum(dfEcoObs.notna().sum(axis=1) == n) / len(dfEcoObs), 2)  # percent
dfEcoObs.count().count()

# Merge DataFrames for ecological status (observed and basis analysis for VP3)
dfObs = dfEcoObs.merge(dfVP[["Basis"]], on="wb")

# Convert typology to integers
typ = dfVP[["ov_typ"]].copy()

# Define the dictionaries
dict1 = {
    "No": "North Sea",  # Nordsø
    "K": "Kattegat",  # Kattegat
    "B": "Belt Sea",  # Bælthav
    "Ø": "Baltic Sea",  # Østersøen
    "Fj": "Fjord",  # Fjord
    "Vf": "North Sea fjord",  # Vesterhavsfjord
}
dict2 = {
    "Vu": "Water exchange",  # vandudveksling
    "F": "Freshwater inflow",  # ferskvandspåvirkning
    "D": "Deep",  # vanddybde
    "L": "Stratified",  # lagdeling
    "Se": "Sediment",  # sediment
    "Sa": "Saline",  # salinitet
    "T": "Tide",  # tidevand
}

# Apply the function to typ["ov_typ"] to create a new DataFrame with the dummy variables
dummies = typ["ov_typ"].apply(process_string).apply(pd.Series).fillna(0).astype(int)
dum = dummies.loc[:, dict1.keys()]
dum["sum"] = dum.sum(axis=1)

# Combine the dictionaries
dicts = {**dict1, **dict2}

# Dummies for typology
cols = list(dicts.values())
cols_abbreviations = list(dicts.keys())

# Merge DataFrames for typology and observed ecological status
dfTyp = dfObs.merge(dummies[cols_abbreviations], on="wb")

# Subset dfTyp to waterbodies where ecological status is observed at least once
obs = dfTyp.loc[dfEcoObs.notna().any(axis=1)]  #  96 out of 108 waterbodies

# Empty dfs for storing yearly share, share by dummy, and status by dummy respectively
d = pd.DataFrame(index=dfEcoObs.columns)  #  yearly number of obs and share by dummy
VPstats = pd.DataFrame(columns=["Observed subset", "All in VP3"])  #  share by dummy
VPbasis = pd.DataFrame(columns=["Observed subset", "All in basis analysis"])  #  status

# Yearly distribution of observed coastal waters by typology and district
for c in cols_abbreviations:
    d[c] = obs[obs[c] == 1].count() / obs.count()
    d.loc["All obs", c] = len(obs[obs[c] == 1]) / len(obs)
    d.loc["All in VP3", c] = len(dummies[dummies[c] == 1]) / len(dummies)
    # Basis analysis share with less than good ecological status (< GES) by dummy
    VPbasis.loc[c, "Observed subset"] = (obs[obs[c] == 1]["Basis"] < 3).mean()
    VPbasis.loc[c, "All in basis analysis"] = (dfTyp[dfTyp[c] == 1]["Basis"] < 3).mean()
# Share with < GES as assessed in the basis analysis for VP3 by dummy and subset
VPbasis.loc["All", "Observed subset"] = (obs["Basis"] < 3).mean()  # observed subset
VPbasis.loc["All", "All in basis analysis"] = (dfTyp["Basis"] < 3).mean()  # in basis a.
d["n"] = dfEcoObs.count().astype(int)  #  number of coastal waters observed each year
d.loc["All obs", "n"] = len(obs)  #  96 coastal waters are observed at least once
d.loc["All in VP3", "n"] = len(dfVP)  #  108 is the total number coastal waters in VP3
d = d.rename(columns=dicts)  #  rename columns from abbreviations to full names

# Save descriptive statistics by year to CSV
d.to_csv("output/coastal_VP_stats_yearly.csv")  #  yearly distributions

# Descriptive statistics by subset
VPstats["Observed subset"] = d.loc["All obs", :]  # observed at least onces
VPstats["All in VP3"] = d.loc["All in VP3", :]  #  distribution of all in VP3
VPstats  #  Kattegat, Belt Sea, Baltic Sea, Water Exchange, Deep, Stratified less obs

# Mean ecological status in basis analysis by dummy and subset
VPbasis = VPbasis.rename(index=dicts)  #  rename index from abbreviations to full names
VPbasis.loc["n", "Observed subset"] = obs["Basis"].mean()  #  for observed subset
VPbasis.loc["n", "All in basis analysis"] = dfTyp["Basis"].mean()  # in basis analysis
VPbasis.iloc[-1, :] = VPstats.iloc[-1, :]  #  number of coastal waters by subset
VPbasis  # GES is overrepresented in observed subset for dummies Kattegat
# GES is slightly underrepresented in observed subset for Water exchange and Saline
# Besides, share < GES is about the same (±.02) for the observed subset given the dummy

# Save descriptive statistics and mean basis analysis to CSV and LaTeX
for a, b in zip([VPstats, VPbasis], ["VP_stats", "VP_basis"]):
    a.to_csv("output/coastal_" + b + ".csv")  #  save means by subset to CSV
    f = {row: "{:0.0f}".format if row == "n" else "{:0.2f}".format for row in a.index}
    with open("output/coastal_" + b + ".tex", "w") as tf:
        tf.write(a.apply(f, axis=1).to_latex(column_format="lcc"))  #  column alignment

########################################################################################
#   3. Visualization: Accuracy and share with less than good ecological status by year
########################################################################################
