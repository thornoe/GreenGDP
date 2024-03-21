"""
Name:       streams_CV.py

Label:      Impute missing values in longitudinal data on ecological status of streams.

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
    "green": "#228833",
    "yellow": "#CCBB44",
    "grey": "#BBBBBB",  #  moved up to be used for ecological status of observed streams
    "red": "#EE6677",
    "purple": "#AA3377",
}

# Set the default property-cycle and figure size for pyplots
color_cycler = cycler(color=list(colors.values()))  #  color cycler with 7 colors
linestyle_cycler = cycler(linestyle=["-", "--", "-.", ":", "-", "--", ":"])  #  7 styles
plt.rc("axes", prop_cycle=(color_cycler + linestyle_cycler))  #  set color and linestyle
plt.rc("figure", figsize=[10, 6.2])  #  golden ratio


# Function for accuracy score of predicted fauna class
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


def stepwise_selection(subset, dummies, data, dfDummies, years, select_all=False):
    """Forward stepwise selection of predictors p to include in the model."""
    predictors = ["No dummies"] + dummies  #  list of possible predictors to include
    selected = []  #  empty list for storing selected predictors
    predictors = [dummies[-1]]  #  only include the last element in the dummies list
    selected = dummies[:-1]  #  slice to include all but the last element in dummies
    current_score, best_new_score = 0.0, 0.0  #  initial scores
    prefixes = ["streams_eco_imp_accuracy", "streams_eco_imp_LessThanGood"]  # CSV names

    # DataFrame for storing accuracy scores by year and calculating weighted average
    scores = pd.DataFrame(subset.count(), index=years, columns=["n"]).astype(int)
    scores.loc["Total", "n"] = np.nan  #  row to calculate weighted average of scores
    scores_all = scores.copy()  #  scores for all sets of predictors being tested

    # DataFrame for storing ecological status by year and calculating weighted average
    status = pd.DataFrame(subset.count(), index=subset.columns, columns=["n"])
    status["Obs"] = (subset < 2.5).sum() / status["n"]  #  ecological status < good
    status.loc["Total", "Obs"] = (status["Obs"] * status["n"]).sum() / status["n"].sum()
    status_all = status.copy()  #  eco status for all sets of predictors being tested

    while current_score == best_new_score:
        names = []  #  empty list for storing model names
        scores_total = []  #  empty list for storing total score for each predictor
        sco = scores[["n"]].copy()  #  df for calculating weighted average of scores
        sta = status[["n"]].copy()  #  df for calculating weighted average of status

        for p in predictors:
            if p == "No dummies":  #  baseline model without any dummies
                df = data.copy()  #  df without predictors
                df.name = "No dummies"  #  name baseline model
            else:
                predictors_used = selected + [p]  #  selected predictors remain in model
                df = data.merge(dfDummies[predictors_used], on="wb")  #  with predictors
                df.name = ", ".join(predictors_used)  #  name model after its predictors
            names.append(df.name)  #  add model name to list of model names

            # Estimate share with less than good ecological status
            dfImp = pd.DataFrame(
                imputer.fit_transform(np.array(df)), index=df.index, columns=df.columns
            )

            # Subset to the waterbodies included in the subset and drop predictors
            dfImpSubset = dfImp.loc[subset.index, subset.columns]

            # Predicted share with less than good ecological status for relevant years
            sta[df.name] = (dfImpSubset[years] < 2.5).sum() / len(subset)

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
            for a, b in zip([scores_all, status_all], [sco, sta]):
                b.loc["Total", df.name] = (b[df.name] * b["n"]).sum() / b["n"].sum()
                a[df.name] = b[df.name]  #  scores & status by year for all predictors
            scores_total.append(sco.loc["Total", df.name])  #  score for each predictor

            print(df.name, "used for imputation. Accuracy score:", scores_total[-1])

            if p == "No dummies":
                break  #  save baseline model before stepwise selection of dummies

            elif select_all is True:
                break  #  proceed to select every predictor in the order they are listed

        best_new_score = max(scores_total)  #  best accuracy score among predictors

        if select_all is True:
            current_score = best_new_score  #  update current score to continue process

            # Move the given dummy from the list of predictors to the list of selected
            selected.append(predictors.pop(0))

            # Save scores and status by year for every set of predictors
            for a, b, c in zip([scores, status], [scores_all, status_all], prefixes):
                a[names[-1]] = b[names[-1]]  #  scores & status by year for predictor
                b.to_csv("output/" + c + ".csv")  # overwrite CSV

        elif best_new_score > current_score:
            current_score = best_new_score  #  update current score
            i = scores_total.index(best_new_score)  #  index for predictor w. best score

            # Move dummy with the best new score from the list of predictors to selected
            selected.append(predictors.pop(i))

            # # Save scores and status by year subject to the selected set of predictors
            for a, b, c in zip([scores, status], [sco, sta], prefixes):
                a[names[i]] = b[names[i]]  #  scores & status by year for best predictor
                a.to_csv("output/" + c + "_sparse.csv")  # overwrite CSV

            # Save scores and status by year for every set of predictors
            for a, b in zip([scores_all, status_all], prefixes):
                a.to_csv("output/" + b + "_sparse_all.csv")  # overwrite CSV

        else:  #  if best_new_score == current_score (i.e., identical accuracy score)
            break  #  stop selection (including the predictor would increase variance)

        if p == "No dummies":
            selected = []  #  after baseline model, start actual stepwise selection

        elif predictors == []:  #  if all predictors have been included
            break  #  stop stepwise selection

    # Total number of observations that LOO-CV was performed over
    for s in (scores, status, scores_all, status_all):
        s.loc["Total", "n"] = s["n"].sum()

    # Overwrite CSV of accuracy scores and share with less than good ecological status
    if subset is sparse:
        scores.to_csv("output/streams_eco_imp_accuracy_sparse.csv")
        status.to_csv("output/streams_eco_imp_LessThanGood_sparse.csv")
        scores_all.to_csv("output/streams_eco_imp_accuracy_sparse_all.csv")
        status_all.to_csv("output/streams_eco_imp_LessThanGood_sparse_all.csv")
    else:
        scores.to_csv("output/streams_eco_imp_accuracy.csv")
        status.to_csv("output/streams_eco_imp_LessThanGood.csv")

    return selected, scores, status  #  selected predictors; scores and stats by year


########################################################################################
#   1. Data setup
########################################################################################
# Specify the working directory of the operating system
os.chdir(r"C:\Users\au687527\GitHub\GreenGDP\gis")

# Limit LOO-CV to loop over years used directly for the natural capital account
years = list(range(1989, 2020 + 1))

# Read DataFrames for observed ecological status and typology
dfEcoObs = pd.read_csv("output/streams_eco_obs.csv", index_col="wb")
dfEcoObs.columns = dfEcoObs.columns.astype(int)
dfVP = pd.read_csv("output\\streams_VP.csv", index_col="wb")

# Share of waterbodies by number of non-missing values
for n in range(0, len(dfEcoObs.columns) + 1):
    n, round(100 * sum(dfEcoObs.notna().sum(axis=1) == n) / len(dfEcoObs), 2)  # percent

# Subset of rows where only 1-3 values are non-missing
sparse = dfEcoObs[dfEcoObs.notna().sum(axis=1).isin([1, 2, 3])]
sparse.count()  #  lowest number of non-missing values with support in all years
sparse.count().sum()  #  5419 non-missing values in total to loop over with LOO-CV

# Merge DataFrames for ecological status (observed and basis analysis for VP3)
dfObs = dfEcoObs.merge(dfVP[["Basis"]], on="wb")

# Create dummies for typology
typ = pd.get_dummies(dfVP["ov_typ"]).astype(int)
typ["Soft bottom"] = typ["RW4"] + typ["RW5"]
typ.columns = [
    "Small",
    "Medium",
    "Large",
    "Small w. soft bottom",
    "Medium w. soft bottom",
    "Soft bottom",
]

# List dummies for typology
cols = ["Small", "Medium", "Large", "Soft bottom"]

# Merge DataFrames for typology and observed biophysical indicator
dfTypology = dfObs.merge(typ[cols], on="wb")

# Create dummies for natural, artificial, and heavily modified waterbodies
natural = pd.get_dummies(dfVP["na_kun_stm"]).astype(int)
natural.columns = ["Artificial", "Natural", "Heavily modified"]

# Merge DataFrames for typology and natural waterbodies
dfNatural = dfTypology.merge(natural["Natural"], on="wb")
cols.append("Natural")  #  add to list of dummies

# Create dummies for waterbody district
distr = pd.get_dummies(dfVP["distr_id"]).astype(int)

# Extend dfNatural with dummy for district DK2 (Sealand, Lolland, Falster, and Møn)
dfDistrict = dfNatural.merge(distr["DK2"], on="wb")
dfSparse = dfDistrict.merge(sparse[[]], on="wb")  #  subset w. status observed 1-3 times
cols.append("DK2")  #  add to list of dummies

# Empty DataFrame for storing total distribution by dummies
VPstats = pd.DataFrame(columns=["Sparse subset", "Observed subset", "All in VP3"])

# Yearly distribution of observed streams by typology and district
for a, b in zip([dfEcoObs, sparse], [dfDistrict, dfSparse]):
    # Subset dfDistrict to streams where ecological status is observed at least once
    obs = b.loc[a.notna().any(axis=1)]  #  6151 out of 6703 streams in dfDistrict

    # df for storing number of observed streams and yearly distribution by dummies
    # d = pd.DataFrame(a.count(), index=a.columns, columns=["n"]).astype(int)
    d = pd.DataFrame(index=a.columns)

    # Yearly distribution of observed streams by typology and district
    for c in cols:
        d[c] = b[b[c] == 1].count() / b.count()  #  share w. typology c each year
        d.loc["All obs", c] = len(obs[obs[c] == 1]) / len(obs)
        d.loc["All in VP3", c] = len(dfDistrict[dfDistrict[c] == 1]) / len(dfVP)

    # Mean ecological status as assessed in basis analysis for VP3
    d.loc["All obs", "Basis"] = obs["Basis"].mean()  #  streams observed at least once
    d.loc["All in VP3", "Basis"] = dfDistrict["Basis"].mean()  #  all streams in VP3

    # Number of streams
    d["n"] = a.count().astype(int)  #  number of streams observed each year
    d.loc["All obs", "n"] = len(obs)  #  number of streams observed at least once
    d.loc["All in VP3", "n"] = len(dfVP)  #  number of streams in VP3

    if b is dfSparse:
        VPstats["Sparse subset"] = d.loc["All obs", :]  #  only observed 1-3 times
        d = d.rename(index={"All obs": "All sparse"})  #  specify it's the sparse subset
        d.to_csv("output/streams_VP_stats_yearly_sparse.csv")  #  save yearly distribut.
    else:
        VPstats["Observed subset"] = d.loc["All obs", :]  # observed at least once
        VPstats["All in VP3"] = d.loc["All in VP3", :]  #  distribution of all VP3
        d.to_csv("output/streams_VP_stats_yearly.csv")  #  save yearly distributions
VPstats  #  sparse has underrepresentation of Large and DK2 (share < 50% of average VP3)

# Save descriptive statistics by subset
VPstats.to_csv("output/streams_VP_stats.csv")  #  save means to CSV
f = {row: "{:0.0f}".format if row == "n" else "{:0.4f}".format for row in VPstats.index}
with open("output/streams_VP_stats.tex", "w") as tf:
    tf.write(VPstats.apply(f, axis=1).to_latex())  #  apply formatter and save to LaTeX


########################################################################################
#   2. Subset selection (note: CV takes ~30hours for sparse + ~54h for all observations)
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
# dfTypology["Small"] = [0, 0, 1, 1, 0, 0]  #  effect: 0.2 worse in 1993
# dfDistrict = dfTypology.copy()
# dfDistrict["DK1"] = [0, 0, 0, 1, 1, 0]  #  effect: 0.1 better in 1993
# cols = ["Small", "DK1"]
# years = list(range(1989, 1993 + 1))

# Forward stepwise selection of dummies - CV over subset of sparsely observed streams
kwargs = {"data": dfObs, "dfDummies": dfDistrict, "years": years}  #  shared arguments
selectedSparse, scoresSparse, statusSparse = stepwise_selection(
    subset=sparse, dummies=cols, **kwargs
)
scoresSparse
statusSparse

# Selection of dummies from the dummies selected above - CV over all observed values
selected, scores, status = stepwise_selection(
    subset=dfEcoObs, dummies=selectedSparse, select_all=True, **kwargs
)
scores
status


########################################################################################
#   3. Visualization: Accuracy and share with less than good ecological status by year
########################################################################################
# Skip step 2 by reading DataFrames of accuracy score and ecological status from CSV
# scores = pd.read_csv("output/streams_eco_imp_accuracy.csv", index_col=0)
# status = pd.read_csv("output/streams_eco_imp_LessThanGood.csv", index_col=0)

# Accuracy score by year and selected predictors
sco = scores.drop(columns="n").drop("Total")

# Bar plot accuracy scores
f1 = sco.plot(
    kind="bar", ylabel="Accuracy in predicting observed ecological status"
).get_figure()
f1.savefig("output/streams_eco_imp_accuracy.pdf", bbox_inches="tight")  #  save PDF

# Share of streams with less than good ecological status by year and selected predictors
status.index = status.index.astype(str)  #  convert index to string (to mimic read_csv)
listYears = [str(t) for t in range(1990, 2020 + 1)]  #  1990 to 2020 as strings
status_years = status.loc[listYears, :]  #  subset to years in natural capital account
imp = status_years.drop(columns=["n", "Obs"])  #  imputed status by selected predictors
obs = status_years[["Obs"]]  #  ecological status of streams observed the given year
obs.columns = ["Observed"]  #  rename 'Obs' to 'Observed'
sta = imp.merge(obs, left_index=True, right_index=True)  #  add Observed as last column

# Plot share of streams with less than good ecological status
f2 = sta.plot(
    ylabel="Share of streams with less than good ecological status"
).get_figure()
f2.savefig("output/streams_eco_imp_LessThanGood.pdf", bbox_inches="tight")  #  save PDF
