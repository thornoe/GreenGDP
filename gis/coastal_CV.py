"""
Name:       coastal_CV.py

Label:      Impute missing longitudinal data on ecological status of coastal waters.

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


# Define a function to process each string in typology
def process_string(s):
    # Drop the hyphen and everything following it
    s = s.split("-")[0]

    # Create a dictionary with the relevant abbreviations as keys and 1s as values
    dummies = {}

    # Check for abbreviations from dict1 first
    for abbr in dict1:
        if abbr in s:
            dummies[abbr] = 1
            s = s.replace(abbr, "")  # Remove the matched abbreviation from the string

    # Then check for abbreviations from dict2
    for abbr in dict2:
        if abbr in s:
            dummies[abbr] = 1

    return dummies


# Function for score
def AccuracyScore(y_true, y_pred):
    """Convert continuous prediction of ecological status to categorical index and return accuracy score, i.e., the share of observed coastal waters each year where predicted ecological status matches the true ecological status (which LOO-CV omits from the dataset before applying imputation)."""
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
dfEcoObs = pd.read_csv("output/coastal_eco_obs.csv", index_col="wb")
dfEcoObs.columns = dfEcoObs.columns.astype(int)
dfVP = pd.read_csv("output\\coastal_VP.csv", index_col="wb")

# Share of water bodies by number of non-missing values
for n in range(0, len(dfEcoObs.columns) + 1):
    n, round(100 * sum(dfEcoObs.notna().sum(axis=1) == n) / len(dfEcoObs), 2)  # percent

# Subset of rows where 1-15 values are non-missing
sparse = dfEcoObs[dfEcoObs.notna().sum(axis=1).isin(list(range(1, 15 + 1)))]
sparse.count()  #  lowest number of non-missing values with support in all years
sparse.count().sum()  #  302 non-missing values in total to loop over with LOO-CV

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

# Merge DataFrames for ecological status (observed and basis analysis for VP3)
dfObs = dfEcoObs.merge(basis, on="wb")

# Convert typology to integers
typ = dfVP[["ov_typ"]].copy()

# Define the dictionaries
dict1 = {
    "Nordsø": "No",
    "Kattegat": "K",
    "Bælthav": "B",
    "Østersøen": "Ø",
    "Fjord": "Fj",
    "Vesterhavsfjord": "Vf",
}
dict2 = {
    "water exchange": "Vu",
    "freshwater inflow": "F",
    "water depth": "D",
    "stratification": "L",
    "sediment": "Se",
    "salinity": "Sa",
    "tide": "T",
}

# Reverse the dictionaries so the abbreviations are the keys
dict1 = {v: k for k, v in dict1.items()}
dict2 = {v: k for k, v in dict2.items()}

# Combine the dictionaries
dicts = {**dict1, **dict2}

# Apply the function to typ["ov_typ"] to create a new DataFrame with the dummy variables
dummies = typ["ov_typ"].apply(process_string).apply(pd.Series).fillna(0).astype(int)

# Dummies for typology
cols = ["No", "K", "B", "Ø", "Fj", "Vf", "Vu", "F", "D", "L", "Se", "Sa", "T"]

# Merge DataFrames for typology and observed ecological status
dfTypology = dfObs.merge(dummies[cols], on="wb")
dfSparse = dfTypology.merge(sparse[[]], on="wb")  #  sparse subset of dfTypology

# Empty DataFrame for storing distribution of typology in
VPstats = pd.DataFrame(columns=["All VP3", "Sparse"])

# Df for storing number of observed coastal waters and yearly distribution by dummies
d = pd.DataFrame(dfEcoObs.count(), index=dfEcoObs.columns, columns=["n"]).astype(int)

# Yearly distribution of observed coastal waters by typology and district
for c in cols:
    d[c] = 100 * dfTypology[dfTypology[c] == 1].count() / dfTypology.count()
    d.loc["All VP3", c] = 100 * len(dfTypology[dfTypology[c] == 1]) / len(dfTypology)
d.loc["All VP3", "n"] = len(dfTypology)
d.to_csv("output/coastal_VP_stats.csv")
d.rename(columns=dicts).loc["All VP3", :]  #  report in percent

########################################################################################
#   2. Multivariate feature imputation
#   2.a Compare inclusion of single variables (LOO-CV takes ≤ 1 hour for each model)
########################################################################################
# DataFrame for storing accuracy scores by year and calculating weighted average
scores = pd.DataFrame(dfEcoObs.count(), index=years, columns=["n"]).astype(int)

# DataFrame for storing ecological status by year and calculating weighted average
status = pd.DataFrame(dfEcoObs.count(), index=dfEcoObs.columns, columns=["n"])
status["Obs"] = (dfEcoObs < 2.5).sum() / status["n"]  #  ecological status < good
status.loc["Total", "Obs"] = (status["Obs"] * status["n"]).sum() / status["n"].sum()

# Compare inclusion of single variables using leave-one-out cross-validation (LOO-CV)
for name in [
    "No dummies",
    "No",
    "K",
    "B",
    "Ø",
    "Fj",
    "Vf",
    "Vu",
    "F",
    "D",
    "L",
    "Se",
    "Sa",
    "T",
]:  #  LOO-CV of models using different dummies
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
    print(df.name, "used for imputation. LOO-CV of observed coastal waters each year:")
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
scores.rename(columns=dicts).loc["Change", :] * 100  #  report in percentage points
# freshwater "deep" worsens prediction accuracy (as a single dummy)

# Total number of observations that LOO-CV was performed over
for s in (scores, status):
    s.loc["Total", "n"] = s["n"].sum()

# List dummies that improve prediction accuracy
cols_pos_single = [col for col in scores.columns if scores.loc["Change", col] > 0]

# Save accuracy scores and share with less than good ecological status to CSV
scores.to_csv("output/coastal_eco_imp_accuracy_single.csv")
status.to_csv("output/coastal_eco_imp_LessThanGood_single.csv")

########################################################################################
#   2. Multivariate feature imputation
#   2.a Compare inclusion of single variables (LOO-CV takes ≤ 1 hour for each model)
########################################################################################
# DataFrame for storing accuracy scores by year and calculating weighted average
scores = pd.DataFrame(dfEcoObs.count(), index=years, columns=["n"]).astype(int)

# DataFrame for storing ecological status by year and calculating weighted average
status = pd.DataFrame(dfEcoObs.count(), index=dfEcoObs.columns, columns=["n"])
status["Obs"] = (dfEcoObs < 2.5).sum() / status["n"]  #  ecological status < good
status.loc["Total", "Obs"] = (status["Obs"] * status["n"]).sum() / status["n"].sum()

# Compare inclusion of single variables using leave-one-out cross-validation (LOO-CV)
for name in [
    "No dummies",
    "No",
    "K",
    "B",
    "Ø",
    "Fj",
    "Vf",
    "Vu",
    "F",
    "D",
    "L",
    "Se",
    "Sa",
    "T",
]:  #  LOO-CV of models using different dummies
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
    print(df.name, "used for imputation. LOO-CV of observed coastal waters each year:")
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
scores.rename(columns=dicts).loc["Change", :] * 100  #  report in percentage points
# freshwater inflow, salinity, and DK2 worsens prediction accuracy (as a single dummy)

# Total number of observations that LOO-CV was performed over
for s in (scores, status):
    s.loc["Total", "n"] = s["n"].sum()

# List dummies that improve prediction accuracy (> 0.006 to omit fjords as reference)
cols_pos_single = [col for col in scores.columns if scores.loc["Change", col] > 0.006]

# Save accuracy scores and share with less than good ecological status to CSV
scores.to_csv("output/coastal_eco_imp_accuracy_single.csv")
status.to_csv("output/coastal_eco_imp_LessThanGood_single.csv")

########################################################################################
#   2.b Additive inclusion of single variables if its addition improves accuracy
########################################################################################
# Dummies that improved prediction accuracy w. single inclusion (to skip section 2.a)
cols_pos_single = ["alkalinity", "brown", "saline", "DK2"]
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
    print(df.name, "used for imputation. LOO-CV of observed coastal waters each year:")
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
    scores.to_csv("output/coastal_eco_imp_accuracy_" + df.name + ".csv")
    status.to_csv("output/coastal_eco_imp_LessThanGood_" + df.name + ".csv")

# Improvement in accuracy or predicted status due to dummy (difference from "No dummies")
for s in (scores, status):
    s.loc["Change", :] = s.loc["Total", :] - s.loc["Total", "No dummies"]
    s.loc["Total", "n"] = s["n"].sum()  #  total observations used for LOO-CV
scores
status

# Save accuracy scores and share with less than good ecological status to CSV
scores.to_csv("output/coastal_eco_imp_accuracy_additive.csv")
status.to_csv("output/coastal_eco_imp_LessThanGood_additive.csv")

########################################################################################
#   2.c Comparison of models for illustrative figures
########################################################################################
# Iterative imputer using the BayesianRidge() estimator with increased tolerance
imputer = IterativeImputer(tol=1e-1, max_iter=50, random_state=0)

# DataFrame for storing accuracy scores by year and calculating weighted average
scores = pd.DataFrame(dfEcoObs.count(), index=years, columns=["n"]).astype(int)

# DataFrame for storing ecological status by year and calculating weighted average
status = pd.DataFrame(dfEcoObs.count(), index=dfEcoObs.columns, columns=["n"])
status["Obs"] = (dfEcoObs < 2.5).sum() / status["n"]  #  ecological status < good
status.loc["Total", "Obs"] = (status["Obs"] * status["n"]).sum() / status["n"].sum()

# Leave-one-out cross-validation (LOO-CV) loop over every observed stream and year
dfEcoObs.name = "No dummies"  #  name model without any dummies
dfTypology.name = "Typology"  #  name model with dummies for typology
for df in (dfObs, dfTypology, dfTypology):  #  LOO-CV of models using different dummies
    # Estimate share with less than good ecological status
    df_imp = pd.DataFrame(
        imputer.fit_transform(np.array(df)), index=df.index, columns=df.columns
    )

    # Store predicted share with less than good ecological status
    status[df.name] = (df_imp[dfEcoObs.columns] < 2.5).sum() / len(df)

    # loop over each year t
    print(df.name, "used for imputation. LOO-CV of observed coastal waters each year:")
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
    scores.to_csv("output/coastal_eco_imp_accuracy_" + df.name + ".csv")
    status.to_csv("output/coastal_eco_imp_LessThanGood_" + df.name + ".csv")

# Total observations used for LOO-CV
for s in (scores, status):
    s.loc["Total", "n"] = s["n"].sum()
scores
status

# Save accuracy scores and share with less than good ecological status to CSV
scores.to_csv("output/coastal_eco_imp_accuracy.csv")
status.to_csv("output/coastal_eco_imp_LessThanGood.csv")

########################################################################################
#   3. Visualization: Accuracy and share with less than good ecological status by year
########################################################################################
# Read accuracy scores and share with less than good ecological status from CSV
# scores = pd.read_csv("output/coastal_eco_imp_accuracy.csv", index_col=0)
sco = scores.drop(columns="n").drop("Total")
# status = pd.read_csv("output/coastal_eco_imp_LessThanGood.csv", index_col=0)
# sta = status[["No dummies", "Typology", "Typology & DK2", "Obs"]].drop("Total")
# sta.columns = ["No dummies", "Typology", "Typology & DK2", "Observed"]  #  rename 'Obs'
sta = status.drop(columns="n").drop("Total")

# Bar plot accuracy scores
f1 = sco.plot(
    kind="bar", ylabel="Accuracy in predicting observed ecological status"
).get_figure()
f1.savefig("output/coastal_eco_imp_accuracy.pdf", bbox_inches="tight")

# Plot share of coastal waters with less than good ecological status
f2 = sta.plot(
    ylabel="Share of coastal waters with less than good ecological status"
).get_figure()
f2.savefig("output/coastal_eco_imp_LessThanGood.pdf", bbox_inches="tight")

# Bar plot share of coastal waters with less than good ecological status
# f3 = sta.plot(
#     kind="bar", ylabel="Share of coastal waters with less than good ecological status"
# ).get_figure()
# f3.savefig("output/coastal_eco_imp_LessThanGood_bar.pdf", bbox_inches="tight")
