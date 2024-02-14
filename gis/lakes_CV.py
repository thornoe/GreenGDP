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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
from cycler import cycler
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.metrics import accuracy_score

# Iterative imputer using the BayesianRidge() estimator with increased tolerance
imputer = IterativeImputer(tol=1e-1, max_iter=100, random_state=0)

# Color-blind-friendly color scheme for qualitative data by Tol: personal.sron.nl/~pault
colors = {
    "blue": "#4477AA",
    "cyan": "#66CCEE",
    "grey": "#BBBBBB",  #  moved up to be used for ecological status of observed lakes
    "green": "#228833",
    "yellow": "#CCBB44",
    "red": "#EE6677",
    "purple": "#AA3377",
}

# Set the default property-cycle and figure size for pyplots
color_cycler = cycler(color=list(colors.values()))  #  color cycler with 7 colors
linestyle_cycler = cycler(linestyle=["-", "--", ":", "-.", "-", "--", ":"])  #  7 styles
plt.rc("axes", prop_cycle=(color_cycler + linestyle_cycler))
plt.rc("figure", figsize=[12, 7.4])  #  golden ratio for appendix with wide margins


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


def stepwise_selection(subset, dummies, data, dfDummies, years):
    """Forward stepwise selection of predictors p to include in the model."""
    predictors = ["No dummies"] + dummies  #  list of possible predictors to include
    selected = []  #  empty list for storing selected predictors
    current_score, best_new_score = 0.0, 0.0  #  initial scores

    # DataFrame for storing accuracy scores by year and calculating weighted average
    scores = pd.DataFrame(subset.count(), index=years, columns=["n"]).astype(int)
    scores.loc["Total", "n"] = np.nan  #  row to calculate weighted average of scores

    # DataFrame for storing ecological status by year and calculating weighted average
    status = pd.DataFrame(subset.count(), index=subset.columns, columns=["n"])
    status["Obs"] = (subset < 2.5).sum() / status["n"]  #  ecological status < good
    status.loc["Total", "Obs"] = (status["Obs"] * status["n"]).sum() / status["n"].sum()

    while current_score == best_new_score:
        names = []  #  empty list for storing model names
        scores_total = []  #  empty list for storing total score for each predictor
        sco = scores[["n"]].copy()  #  df for calculating weighted average of scores
        sta = status[["n"]].copy()  #  df for calculating weighted average of status

        for p in predictors:
            if p == "No dummies":  #  baseline model without any dummies
                df = data.copy()
                df.name = "No dummies"  #  name baseline model
            else:
                predictors_used = selected + [p]  #  selected predictors remain in model
                df = data.merge(dfDummies[predictors_used], on="wb")
                df.name = ", ".join(predictors_used)  #  name model after its predictors
            names.append(df.name)  #  add model name to list of model names

            # Estimate share with less than good ecological status
            dfImp = pd.DataFrame(
                imputer.fit_transform(np.array(df)), index=df.index, columns=df.columns
            )

            # Subset to rows in "subset"
            dfImpSubset = dfImp.loc[
                subset.index, subset.columns
            ]  # CAN I ADD COLS HERE?

            # Store predicted share with less than good ecological status
            sta[df.name] = (dfImpSubset[subset.columns] < 2.5).sum() / len(subset)

            # loop over each year t and waterbody i in (subset of) observed waterbodies
            for t in tqdm.tqdm(years):  #  time each model and report progress in years
                y = subset[subset[t].notnull()].index  #  index for LOO-CV at year t
                Y = pd.DataFrame(index=y)  #  empty df for observed and predicted values
                Y["true"] = df.loc[y, t]  #  column with the observed ('true') values
                Y["pred"] = pd.NA  #  empty column for storing predicted values
                for i in y:  #  loop over each observed value at year t
                    X = df.copy()  #  use a copy of the given DataFrame
                    X.loc[i, t] = pd.NA  #  set the observed value as missing
                    # Fit imputer and transform the dataset
                    X_imp = pd.DataFrame(
                        imputer.fit_transform(np.array(X)),
                        index=X.index,
                        columns=X.columns,
                    )
                    Y.loc[i, "pred"] = X_imp.loc[i, t]  #  store predicted value

                # Accuracy of predicted ecological status
                accuracy = AccuracyScore(Y["true"], Y["pred"])

                # Save accuracy score each year to DataFrame for scores
                sco.loc[t, df.name] = accuracy

            # Total accuracy weighted by number of observations used for LOO-CV each year
            for s in (sco, sta):
                s.loc["Total", df.name] = (s[df.name] * s["n"]).sum() / s["n"].sum()
            scores_total.append(sco.loc["Total", df.name])  #  score for each predictor

            print(df.name, "used for imputation. Accuracy score:", scores_total[-1])

            if p == "No dummies":
                break  #  save baseline model before stepwise selection of dummies

        best_new_score = max(scores_total)  #  best score
        if best_new_score > current_score:
            current_score = best_new_score  #  update current score
            i = scores_total.index(best_new_score)  #  index for predictor w. best score

            # Saves scores and status by year for the predictor with the new best score
            for a, b in zip([scores, status], [sco, sta]):
                a[names[i]] = b[names[i]]  #  scores & status by year for best predictor

            # Move predictor with the best new score from "predictors" to "selected"
            selected.append(predictors.pop(i))

            if p == "No dummies":
                selected = []  #  after baseline model, start actual stepwise selection

        else:  #  if best_new_score == current_score
            break  #  stop stepwise selection

        if predictors == []:  #  if all predictors have been included in the best model
            break  #  stop stepwise selection

    # Total number of observations that LOO-CV was performed over
    for s in (scores, status):
        s.loc["Total", "n"] = s["n"].sum()

    # Save accuracy scores and share with less than good ecological status to CSV
    if subset is sparse:
        scores.to_csv("output/waterbodies_eco_imp_accuracy_sparse.csv")
        status.to_csv("output/waterbodies_eco_imp_LessThanGood_sparse.csv")
    else:
        scores.to_csv("output/waterbodies_eco_imp_accuracy.csv")
        status.to_csv("output/waterbodies_eco_imp_LessThanGood.csv")

    return selected, scores, status  #  selected predictors; scores and stats by year


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

# Share of waterbodies by number of non-missing values
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

# Create dummies for high Alkalinity, Brown, Saline, and Deep lakes
cond1 = [(typ["type"] >= 9) & (typ["type"] <= 16), typ["type"] == 17]
typ["Alkalinity"] = np.select(cond1, [1, np.nan], default=0)
cond2 = [typ["type"].isin([5, 6, 7, 8, 13, 14, 15, 16]), typ["type"] == 17]
typ["Brown"] = np.select(cond2, [1, np.nan], default=0)
cond3 = [typ["type"].isin([2, 3, 7, 8, 11, 12, 15, 16]), typ["type"] == 17]
typ["Saline"] = np.select(cond3, [1, np.nan], default=0)
cond4 = [typ["type"].isin(np.arange(2, 17, 2)), typ["type"] == 17]
typ["Deep"] = np.select(cond4, [1, np.nan], default=0)

# List dummies for typology
cols = ["Alkalinity", "Brown", "Saline", "Deep"]

# Merge DataFrames for typology and observed ecological status
dfTypology = dfObs.merge(typ[cols], on="wb")

# Create dummies for waterbody districts
distr = pd.get_dummies(dfVP["distr_id"]).astype(int)

# Extend dfTypology with dummy for district DK2 (Sealand, Lolland, Falster, and Møn)
dfDistrict = dfTypology.merge(distr["DK2"], on="wb")
dfSparse = dfDistrict.merge(sparse[[]], on="wb")  #  subset w. status observed 1-4 times
cols.append("DK2")

# Empty DataFrame for storing total distribution by dummies
VPstats = pd.DataFrame(columns=["Sparse subset", "Observed subset", "All in VP3"])

# Yearly distribution of observed lakes by typology and district
for a, b in zip([dfEcoObs, sparse], [dfDistrict, dfSparse]):
    # Subset dfDistrict to lakes where ecological status is observed at least once
    obs = b.loc[a.notna().any(axis=1)]  #  779 out of 986 lakes in dfDistrict

    # df for storing number of observed lakes and yearly distribution by dummies
    d = pd.DataFrame(a.count(), index=a.columns, columns=["n"]).astype(int)

    # Yearly distribution of observed lakes by typology and district
    for c in cols:
        d[c] = 100 * b[b[c] == 1].count() / b.count()
        d.loc["Obs of n", c] = 100 * len(obs[obs[c] == 1]) / len(obs)
        d.loc["Obs of all", c] = 100 * len(obs[obs[c] == 1]) / len(dfVP)
        d.loc["All in VP3", c] = 100 * len(dfDistrict[dfDistrict[c] == 1]) / len(dfVP)
    d.loc["Obs of n", "n"] = len(obs)  #  number of lakes observed at least once
    d.loc["Obs of all", "n"] = len(dfVP)  #  number of lakes in VP3
    d.loc["All in VP3", "n"] = len(dfVP)  #  number of lakes in VP3

    if b is dfSparse:
        VPstats["Sparse subset"] = d.loc["Obs of n", :]  #  distribution in sparse df
        d.to_csv("output/lakes_VP_stats_sparse.csv")  #  save to CSV
    else:
        VPstats["Observed subset"] = d.loc["Obs of n", :]  #  distribution of observed
        VPstats["All in VP3"] = d.loc["All in VP3", :]  #  distribution of all in VP3
        d.to_csv("output/lakes_VP_stats.csv")  #  save distributions to CSV
VPstats  #  overrepresentation of Brown and Saline lakes in sparse (share is ~50% above)

########################################################################################
#   2. Multivariate feature imputation (note: Forward Stepwise Selection takes ~6 hours)
########################################################################################
# # Example data for testing Forward Stepwise Selection with LOO-CV (takes ~5 seconds)
# dfEcoObs = pd.DataFrame(
#     {
#         1988: [0.5, 1.0, 1.5, 2.0, np.nan, 3.0],
#         1989: [0.6, 1.1, 1.6, np.nan, 2.6, 3.1],
#         1990: [0.7, 1.2, np.nan, 2.2, 2.7, 3.2],
#         1991: [0.8, np.nan, 1.8, 2.3, 2.8, 3.3],
#         1992: [np.nan, 1.4, 1.9, 2.4, 2.9, 3.4],
#         1993: [1.0, 1.5, 1.8, 2.4, 3.1, 3.5],
#     }
# )
# dfEcoObs.index.name = "wb"
# sparse = dfEcoObs[dfEcoObs.notna().sum(axis=1) == 5]
# dfObs = dfEcoObs.copy()
# dfTypology = dfObs.copy()
# dfTypology["Brown"] = [0, 0, 1, 1, 0, 0]  #  effect: 0.2 worse in 1993
# dfDistrict = dfTypology.copy()
# dfDistrict["DK1"] = [0, 0, 0, 1, 1, 0]  #  effect: 0.1 better in 1993
# cols = ["Brown", "DK1"]
# years = list(range(1989, 1993 + 1))

# Forward stepwise selection of dummies - CV over subset of sparsely observed lakes
kwargs = {"dummies": cols, "data": dfObs, "dfDummies": dfDistrict, "years": years}
selectedSparse, scoresSparse, statusSparse = stepwise_selection(subset=sparse, **kwargs)
scoresSparse
statusSparse

# Forward stepwise selection of dummies - CV over all observed values in all lakes
selected, scores, status = stepwise_selection(subset=dfEcoObs, **kwargs)
scores
status

########################################################################################
#   3. Visualization: Accuracy and share with less than good ecological status by year
########################################################################################
# Skip step 2 by reading DataFrames of accuracy score and ecological status from CSV
# scores = pd.read_csv("output/lakes_eco_imp_accuracy.csv", index_col=0)
sco = scores.drop(columns="n").drop("Total")
sco.columns = ["No dummies", "Saline dummy"]  #  elaborate on "Saline" column name
# status = pd.read_csv("output/lakes_eco_imp_LessThanGood.csv", index_col=0)
sta = status[["No dummies", "Saline", "Obs"]].drop("Total")  #  reorder: "Obs" last
sta.columns = ["No dummies", "Saline dummy", "Observed"]  #  elaborate column names

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
