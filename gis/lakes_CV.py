"""
Name:       imputation.py

Label:      Impute missing values in longitudinal data on ecological status of lakes.

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

import numpy as np
import pandas as pd
import tqdm
from matplotlib import pyplot as plt
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.metrics import accuracy_score

# Color-blind-friendly ordinal scale, modified from: gist.github.com/thriveth/8560036
ColorCycle = {
    "blue": "#377eb8",
    "orange": "#ff7f00",
    "green": "#4daf4a",
    "gray": "#999999",  #  moved up
    "pink": "#f781bf",
    "brown": "#a65628",
    "purple": "#984ea3",
    "yellow": "#dede00",
    "red": "#e41a1c",  #  moved down
}

# Set the default color map and figure size for pyplots
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", list(ColorCycle.values()))
plt.rcParams["figure.figsize"] = [12, 6]  #  wide format (appendix with wide margins)


# Function for score
def AccuracyScore(y_true, y_pred):
    """Convert continuous prediction of ecological status to categorical index and return accuracy score, i.e., the share of observed lakes each year where predicted ecological status matches the true ecological status (which LOO-CV omits)."""
    eco_true, eco_pred = [], []  #  empy lists for storing transformed observations
    for a, b in zip([y_true, y_pred], [eco_true, eco_pred]):
        # Demarcation for categorical ecological status: Bad, Poor, Moderate, Good, High
        conditions = [
            a < 0.5,  # Bad
            (a >= 0.5) & (a < 1.5),  #  Poor
            (a >= 1.5) & (a < 2.5),  #  Moderate
            a >= 2.5,  #  Good or High
        ]
        b.append(np.select(conditions, [0, 1, 2, 3], default=np.nan))  #  add to list
    return accuracy_score(eco_true[0], eco_pred[0])


########################################################################################
#   1. Data setup
########################################################################################
# Specify the working directory of the operating system
os.chdir(r"C:\Users\au687527\GitHub\GreenGDP\gis")

# Limit LOO-CV to loop over years used directly for the natural capital account
years = list(range(1989, 2020 + 1))

# Read DataFrames for observed biophysical indicator and typology
dfEcoObs = pd.read_csv("output/lakes_eco_obs.csv", index_col="wb")
dfEcoObs.columns = dfEcoObs.columns.astype(int)
dfVP = pd.read_csv("output\\lakes_VP.csv", index_col="wb")

# Convert typology to integers
typ = dfVP[["ov_typ"]].copy()
typ.loc[:, "type"] = typ["ov_typ"].str.slice(6).astype(int)

# Create dummies for high alkalinity, brown, saline, and deep lakes
cond1 = [(typ["type"] >= 9) & (typ["type"] <= 16), typ["type"] == 17]
typ["alkalinity"] = np.select(cond1, [1, np.nan], default=0)
cond2 = [typ["type"].isin([5, 6, 7, 8, 13, 14, 15, 16]), typ["type"] == 17]
typ["brown"] = np.select(cond2, [1, np.nan], default=0)
cond3 = [typ["type"].isin([2, 3, 7, 8, 11, 12, 15, 16]), typ["type"] == 17]
typ["saline"] = np.select(cond3, [1, np.nan], default=0)
cond4 = [typ["type"].isin(np.arange(2, 17, 2)), typ["type"] == 17]
typ["deep"] = np.select(cond4, [1, np.nan], default=0)

# List dummies for typology
col = ["alkalinity", "brown", "saline", "deep"]

# Merge DataFrames for typology and observed biophysical indicator
dfTypology = dfEcoObs.merge(typ[col], how="inner", on="wb")

# Create dummies for water body districts
distr = pd.get_dummies(dfVP["distr_id"]).astype(int)

# Extend dfTypology with dummy for district DK2 (Sealand, Lolland, Falster, and Møn)
dfDistrict = dfTypology.merge(distr["DK2"], how="inner", on="wb")
col.append("DK2")

# DataFrame for storing number of observed lakes and yearly distribution by dummies
d = pd.DataFrame(dfEcoObs.count(), index=dfEcoObs.columns, columns=["n"]).astype(int)

# Yearly distribution of observed lakes by typology and district
for c in col:
    d[c] = 100 * dfDistrict[dfDistrict[c] == 1].count() / dfDistrict.count()
    d.loc["All VP3 lakes", c] = (
        100 * len(dfDistrict[dfDistrict[c] == 1]) / len(dfDistrict)
    )
d.loc["All VP3 lakes", "n"] = len(dfDistrict)
d.to_csv("output/lakes_VP_stats.csv")


########################################################################################
#   2. Multivariate feature imputation (note: LOO-CV takes ≤ 23 hours for each model)
########################################################################################
# Iterative imputer using the BayesianRidge() estimator with increased tolerance
imputer = IterativeImputer(tol=1e-1, random_state=0)

# Example data for testing LOO-CV below (takes ~3 seconds rather than ~3 days to run)
# dfEcoObs = pd.DataFrame(
#     {
#         1988: [0.5, 1.0, 1.5, 2.0, np.nan],
#         1989: [0.6, 1.1, 1.6, np.nan, 2.6],
#         1990: [0.7, 1.2, np.nan, 2.2, 2.7],
#         1991: [0.8, np.nan, 1.8, 2.3, 2.8],
#         1992: [np.nan, 1.4, 1.9, 2.4, 2.9],
#     }
# )
# dfTypology = dfEcoObs.copy()
# dfTypology["brown"] = [1, 1, 0, 0, 0]
# dfDistrict = dfTypology.copy()
# dfDistrict["DK2"] = [1, 0, 1, 0, 0]
# years = list(range(1989, 1992 + 1))

# DataFrame for storing accuracy scores by year and calculating weighted average
scores = pd.DataFrame(dfEcoObs.count(), index=years, columns=["n"]).astype(int)

# DataFrame for storing ecological status by year and calculating weighted average
status = pd.DataFrame(dfEcoObs.count(), index=dfEcoObs.columns, columns=["n"])
status["Obs"] = (dfEcoObs < 2.5).sum() / status["n"]  #  ecological status < good
status.loc["Total", "Obs"] = (status["Obs"] * status["n"]).sum() / status["n"].sum()

# Leave-one-out cross-validation (LOO-CV) loop over every observed stream and year
dfEcoObs.name = "No dummies"  #  name model without any dummies
dfTypology.name = "Typology"  #  name model with dummies for typology
dfDistrict.name = "Typology & DK2"  #  name model with dummies for typology and district
for df in (dfEcoObs, dfTypology, dfDistrict):  #  LOO-CV using different dummies
    # Estimate share with less than good ecological status
    df_imp = pd.DataFrame(
        imputer.fit_transform(np.array(df)), index=df.index, columns=df.columns
    )

    # Store predicted share with less than good ecological status
    status[df.name] = (df_imp[dfEcoObs.columns] < 2.5).sum() / len(df)

    # loop over each year t
    print(df.name, "used for imputation. LOO-CV of observed lakes each year:")
    for t in tqdm.tqdm(years):  #  time each model and report its progress in years
        y = df[df[t].notnull()].index  #  index for observed values at year t
        Y = pd.DataFrame(index=y)  #  empty df for observed and predicted values
        Y["true"] = df.loc[y, t]  #  column with the observed ('true') values
        Y["pred"] = pd.NA  #  empty column for storing predicted values
        for i in y:  #  loop over each observed value at year t
            X = df.copy()  #  use a copy of the given DataFrame
            X.loc[i, t] = pd.NA  #  set the observed value as missing
            # Fit imputer and transform the dataset
            X_imp = pd.DataFrame(
                imputer.fit_transform(np.array(X)), index=X.index, columns=X.columns
            )
            Y.loc[i, "pred"] = X_imp.loc[i, t]  #  store predicted value

        # Accuracy of ecological status after converting DVFI fauna index for lakes
        accuracy = AccuracyScore(Y["true"], Y["pred"])

        # Save accuracy score each year to DataFrame for scores
        scores.loc[t, df.name] = accuracy

    # Totals weighted by number of observations
    for s in (scores, status):
        s.loc["Total", df.name] = (s[df.name] * s["n"]).sum() / s["n"].sum()

# Total observations used for LOO-CV
for s in (scores, status):
    s.loc["Total", "n"] = s["n"].sum()
scores
status

# Save accuracy scores and share with less than good ecological status to CSV
scores.to_csv("output/lakes_eco_imp_accuracy.csv")
status.to_csv("output/lakes_eco_imp_not good.csv")

########################################################################################
#   3. Visualization: Accuracy and share with less than good ecological status by year
########################################################################################
# Read accuracy scores and share with less than good ecological status from CSV
scores = pd.read_csv("output/lakes_eco_imp_accuracy.csv", index_col=0)
sco = scores.drop(columns="n").drop("Total")
status = pd.read_csv("output/lakes_eco_imp_not good.csv", index_col=0)
sta = status[["No dummies", "Typology", "Typology & DK2", "Obs"]].drop("Total")
sta.columns = ["No dummies", "Typology", "Typology & DK2", "Observed"]  #  rename 'Obs'

# Bar plot accuracy scores
f1 = sco.plot(
    kind="bar", ylabel="Accuracy in predicting observed ecological status"
).get_figure()
f1.savefig("output/lakes_eco_imp_accuracy.pdf", bbox_inches="tight")

# Plot share of lakes with less than good ecological status
f2 = sta.plot(
    ylabel="Share of lakes with less than good ecological status"
).get_figure()
f2.savefig("output/lakes_eco_imp_not good.pdf", bbox_inches="tight")

# Bar plot share of lakes with less than good ecological status
# f3 = sta.plot(
#     kind="bar", ylabel="Share of lakes with less than good ecological status"
# ).get_figure()
# f3.savefig("output/lakes_eco_imp_not good_bar.pdf", bbox_inches="tight")
