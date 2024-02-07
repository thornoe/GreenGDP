"""
Name:       lakes_CV.py

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
plt.rcParams["figure.figsize"] = [12, 7.4]  #  wide format (appendix with wide margins)


# Function for score
def AccuracyScore(y_true, y_pred):
    """Convert continuous prediction of ecological status to categorical index and return accuracy score, i.e., the share of observed lakes each year where predicted ecological status matches the true ecological status (which LOO-CV omits from the dataset before applying imputation)."""
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


# Iterative imputer using the BayesianRidge() estimator with increased tolerance
imputer = IterativeImputer(tol=1e-1, max_iter=50, random_state=0)

########################################################################################
#   1. Data setup
########################################################################################
# Specify the working directory of the operating system
os.chdir(r"C:\Users\au687527\GitHub\GreenGDP\gis")

# Limit LOO-CV to loop over years used directly for the natural capital account
years = list(range(1989, 2020 + 1))

# Read DataFrames for observed ecological status and typology
dfEcoObs = pd.read_csv("output/lakes_eco_obs.csv", index_col="wb")
dfEcoObs.columns = dfEcoObs.columns.astype(int)
dfVP = pd.read_csv("output\\lakes_VP.csv", index_col="wb")

# Share of water bodies by number of non-missing values
for n in range(0, len(dfEcoObs.columns) + 1):
    n, round(100 * sum(dfEcoObs.notna().sum(axis=1) == n) / len(dfEcoObs), 2)  # percent

# Subset of rows where only 1-4 values are non-missing
sparse = dfEcoObs[dfEcoObs.notna().sum(axis=1).isin([1, 2, 3, 4])]
sparse.count()  #  lowest number of non-missing values with support in all years
sparse.count().sum()  #  994 non-missing values in total to loop over with LOO-CV

# Include ecological status as assessed in basis analysis for VP3
basis = dfVP[["til_oko_fy"]].copy()  #  phytoplankton measured as chlorophyll

# Define a dictionary to map the Danish strings to an ordinal scale
status_dict = {
    "Dårlig økologisk tilstand": 0,
    "Ringe økologisk tilstand": 1,
    "Moderat økologisk tilstand": 2,
    "God økologisk tilstand": 3,
    "Høj økologisk tilstand": 4,
    "Dårligt økologisk potentiale": 0,
    "Ringe økologisk potentiale": 1,
    "Moderat økologisk potentiale": 2,
    "Godt økologisk potentiale": 3,
    "Maksimalt økologisk potentiale": 4,
    "Ukendt": np.nan,
}

# Replace the Danish strings in the DataFrame with the corresponding ordinal values
basis.replace(status_dict, inplace=True)
basis.columns = ["basis"]
basis["basis"].unique()

# Merge DataFrames for ecological status (observed and basis analysis for VP3)
dfObs = dfEcoObs.merge(basis, on="wb")

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
cols = ["alkalinity", "brown", "saline", "deep"]

# Merge DataFrames for typology and observed ecological status
dfTypology = dfObs.merge(typ[cols], on="wb")

# Create dummies for water body districts
distr = pd.get_dummies(dfVP["distr_id"]).astype(int)

# Extend dfTypology with dummy for district DK2 (Sealand, Lolland, Falster, and Møn)
dfDistrict = dfTypology.merge(distr["DK2"], on="wb")
cols.append("DK2")

# Subset dfDistrict to water bodies where ecological status is observed at least once
obs = dfDistrict.loc[dfEcoObs.notna().any(axis=1)]  #  779 out of 986 water bodies

# df for storing number of observed coastal waters and yearly distribution by dummies
d = pd.DataFrame(dfEcoObs.count(), index=dfEcoObs.columns, columns=["n"]).astype(int)

# Yearly distribution of observed coastal waters by typology and district
for c in cols:
    d[c] = 100 * obs[obs[c] == 1].count() / obs.count()
    d.loc["Obs of n", c] = 100 * len(obs[obs[c] == 1]) / len(obs)
    d.loc["Obs of all", c] = 100 * len(obs[obs[c] == 1]) / len(dfDistrict)
    d.loc["All VP3", c] = 100 * len(dfDistrict[dfDistrict[c] == 1]) / len(dfDistrict)
d.loc["Obs of n", "n"] = len(obs)
d.loc["Obs of all", "n"] = len(dfDistrict)
d.loc["All VP3", "n"] = len(dfDistrict)
d.to_csv("output/lakes_VP_stats.csv")  #  save distributions to csv
d.loc[("Obs of n", "Obs of all", "All VP3"), :]  #  report in percent


########################################################################################
#   2. Multivariate feature imputation for subset of sparsely observed lakes
#   2.a Forward stepwise selection of dummies (if their inclusion improves accuracy)
########################################################################################

# DataFrame for storing accuracy scores by year and calculating weighted average
scores = pd.DataFrame(dfEcoObs.count(), index=years, columns=["n"]).astype(int)

# DataFrame for storing ecological status by year and calculating weighted average
status = pd.DataFrame(dfEcoObs.count(), index=dfEcoObs.columns, columns=["n"])
status["Obs"] = (dfEcoObs < 2.5).sum() / status["n"]  #  ecological status < good
status.loc["Total", "Obs"] = (status["Obs"] * status["n"]).sum() / status["n"].sum()

# Compare inclusion of single variables using leave-one-out cross-validation (LOO-CV)
for name in ["No dummies", "alkalinity", "brown", "saline", "deep", "DK2"]:
    if name == "No dummies":
        df = dfObs.copy()
    else:
        df = dfObs.merge(dfDistrict[name], on="wb")
    df.name = name  #  name DataFrame for each model

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

        # Accuracy of predicted ecological status
        accuracy = AccuracyScore(Y["true"], Y["pred"])

        # Save accuracy score each year to DataFrame for scores
        scores.loc[t, df.name] = accuracy

    # Totals weighted by number of observations
    for s in (scores, status):
        s.loc["Total", df.name] = (s[df.name] * s["n"]).sum() / s["n"].sum()

# Change in accuracy or predicted status due to dummy (difference from "No dummies")
scores.loc["Change", :] = scores.loc["Total", :] - scores.loc["Total", "No dummies"]
scores.rename(columns=cols).loc["Change", :] * 100  #  report in percentage points
# freshwater inflow, salinity, and DK2 worsens prediction accuracy (as a single dummy)

# Total number of observations that LOO-CV was performed over
for s in (scores, status):
    s.loc["Total", "n"] = s["n"].sum()

# List dummies that improve prediction accuracy (> 0.006 to omit fjords as reference)
cols_pos_single = [col for col in scores.columns if scores.loc["Change", col] > 0.006]

# Save accuracy scores and share with less than good ecological status to CSV
scores.to_csv("output/lakes_eco_imp_accuracy_single.csv")
status.to_csv("output/lakes_eco_imp_LessThanGood_single.csv")

########################################################################################
#   2. Multivariate feature imputation (note: LOO-CV takes ≤ 1 hour for each model)
#   2.a Forward stepwise selection of dummies (if their inclusion improves accuracy)
########################################################################################

# DataFrame for storing accuracy scores by year and calculating weighted average
scores = pd.DataFrame(dfEcoObs.count(), index=years, columns=["n"]).astype(int)

# DataFrame for storing ecological status by year and calculating weighted average
status = pd.DataFrame(dfEcoObs.count(), index=dfEcoObs.columns, columns=["n"])
status["Obs"] = (dfEcoObs < 2.5).sum() / status["n"]  #  ecological status < good
status.loc["Total", "Obs"] = (status["Obs"] * status["n"]).sum() / status["n"].sum()

# Compare inclusion of single variables using leave-one-out cross-validation (LOO-CV)
for name in ["No dummies", "alkalinity", "brown", "saline", "deep", "DK2"]:
    if name == "No dummies":
        df = dfObs.copy()
    else:
        df = dfObs.merge(dfDistrict[name], on="wb")
    df.name = name  #  name DataFrame for each model

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

        # Accuracy of predicted ecological status
        accuracy = AccuracyScore(Y["true"], Y["pred"])

        # Save accuracy score each year to DataFrame for scores
        scores.loc[t, df.name] = accuracy

    # Totals weighted by number of observations
    for s in (scores, status):
        s.loc["Total", df.name] = (s[df.name] * s["n"]).sum() / s["n"].sum()

# Change in accuracy or predicted status due to dummy (difference from "No dummies")
scores.loc["Change", :] = scores.loc["Total", :] - scores.loc["Total", "No dummies"]
scores.rename(columns=cols).loc["Change", :] * 100  #  report in percentage points
# freshwater inflow, salinity, and DK2 worsens prediction accuracy (as a single dummy)

# Total number of observations that LOO-CV was performed over
for s in (scores, status):
    s.loc["Total", "n"] = s["n"].sum()

# List dummies that improve prediction accuracy (> 0.006 to omit fjords as reference)
cols_pos_single = [col for col in scores.columns if scores.loc["Change", col] > 0.006]

# Save accuracy scores and share with less than good ecological status to CSV
scores.to_csv("output/lakes_eco_imp_accuracy_single.csv")
status.to_csv("output/lakes_eco_imp_LessThanGood_single.csv")

########################################################################################
#   2.b Additive inclusion of single variables if its addition improves accuracy
########################################################################################
# Dummies that improved prediction accuracy w. single inclusion (to skip section 2.a)
cols_pos_single = ["No", "K", "B", "Ø", "Vf", "Vu", "D", "L", "Se", "T"]
imputer = IterativeImputer(tol=1e-1, max_iter=50, random_state=0)

# Empty list for cols that improve accuracy w. additive inclusion
cols_pos_additive = []

# DataFrame for storing accuracy scores by year and calculating weighted average
scores = pd.DataFrame(dfEcoObs.count(), index=years, columns=["n"]).astype(int)

# DataFrame for storing ecological status by year and calculating weighted average
status = pd.DataFrame(dfEcoObs.count(), index=dfEcoObs.columns, columns=["n"])
status["Obs"] = (dfEcoObs < 2.5).sum() / status["n"]  #  ecological status < good
status.loc["Total", "Obs"] = (status["Obs"] * status["n"]).sum() / status["n"].sum()

# Compare additive inclusion of single variables (keep if it improves accuracy)
for name in cols_pos_single:
    if name == "No dummies":
        df = dfObs.copy()
    else:
        df = dfObs.merge(dfTypology[name], on="wb")
    df.name = name  #  name DataFrame for each model

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

        # Accuracy of predicted ecological status
        accuracy = AccuracyScore(Y["true"], Y["pred"])

        # Save accuracy score each year to DataFrame for scores
        scores.loc[t, df.name] = accuracy

    # Totals weighted by number of observations
    for s in (scores, status):
        s.loc["Total", df.name] = (s[df.name] * s["n"]).sum() / s["n"].sum()

    # Save accuracy scores and share with less than good ecological status to CSV
    scores.to_csv("output/lakes_eco_imp_accuracy_" + df.name + ".csv")
    status.to_csv("output/lakes_eco_imp_LessThanGood_" + df.name + ".csv")

# Improvement in accuracy or predicted status due to dummy (difference from "No dummies")
for s in (scores, status):
    s.loc["Change", :] = s.loc["Total", :] - s.loc["Total", "No dummies"]
    s.loc["Total", "n"] = s["n"].sum()  #  total observations used for LOO-CV
scores
status

# Save accuracy scores and share with less than good ecological status to CSV
scores.to_csv("output/lakes_eco_imp_accuracy_additive.csv")
status.to_csv("output/lakes_eco_imp_LessThanGood_additive.csv")

########################################################################################
#   2.c Comparison of models for illustrative figures
########################################################################################

# Example data for testing LOO-CV below (takes ~3 seconds rather than ~3 days to run)
# dfEco = pd.DataFrame(
#     {
#         1988: [0.5, 1.0, 1.5, 2.0, np.nan],
#         1989: [0.6, 1.1, 1.6, np.nan, 2.6],
#         1990: [0.7, 1.2, np.nan, 2.2, 2.7],
#         1991: [0.8, np.nan, 1.8, 2.3, 2.8],
#         1992: [np.nan, 1.4, 1.9, 2.4, 2.9],
#     }
# )
# dfTypology = dfEco.copy()
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
dfObs.name = "No dummies"  #  name model without any dummies
dfTypology.name = "Typology"  #  name model with dummies for typology
dfDistrict.name = "Typology & DK2"  #  name model with dummies for typology and district
for df in (dfObs, dfTypology, dfDistrict):  #  LOO-CV using different dummies
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

        # Accuracy of predicted ecological status
        accuracy = AccuracyScore(Y["true"], Y["pred"])

        # Save accuracy score each year to DataFrame for scores
        scores.loc[t, df.name] = accuracy

    # Totals weighted by number of observations
    for s in (scores, status):
        s.loc["Total", df.name] = (s[df.name] * s["n"]).sum() / s["n"].sum()

# Change in accuracy or predicted status due to dummy (difference from "No dummies")
scores.loc["Change", :] = scores.loc["Total", :] - scores.loc["Total", "No dummies"]
scores.loc["Change", :] * 100  #  report in percentage points

# Total observations used for LOO-CV
for s in (scores, status):
    s.loc["Total", "n"] = s["n"].sum()
scores
status

# Save accuracy scores and share with less than good ecological status to CSV
scores.to_csv("output/lakes_eco_imp_accuracy.csv")
status.to_csv("output/lakes_eco_imp_LessThanGood.csv")

########################################################################################
#   3. Visualization: Accuracy and share with less than good ecological status by year
########################################################################################
# Read accuracy scores and share with less than good ecological status from CSV
# scores = pd.read_csv("output/lakes_eco_imp_accuracy.csv", index_col=0)
sco = scores.drop(columns="n").drop(["Total", "Change"])
# status = pd.read_csv("output/lakes_eco_imp_LessThanGood.csv", index_col=0)
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
f2.savefig("output/lakes_eco_imp_LessThanGood.pdf", bbox_inches="tight")
