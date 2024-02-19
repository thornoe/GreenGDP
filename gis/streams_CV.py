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
plt.rc("axes", prop_cycle=(color_cycler + linestyle_cycler))
plt.rc("figure", figsize=[12, 7.4])  #  golden ratio for appendix with wide margins


# Function for accuracy score of predicted fauna class
def AccuracyScore(y_true, y_pred):
    """Convert DVFI fauna class for bottom fauna in streams to categorical index of ecological status and return accuracy score, i.e., the share of observed streams each year where predicted ecological status matches the true ecological status (which LOO-CV omits from the dataset before applying imputation)."""
    eco_true, eco_pred = [], []  #  empy lists for storing transformed observations
    for a, b in zip([y_true, y_pred], [eco_true, eco_pred]):
        # Categorical variable for ecological status: Bad, Poor, Moderate, Good, High
        conditions = [
            a < 1.5,  # Bad
            (a >= 1.5) & (a < 3.5),  #  Poor
            (a >= 3.5) & (a < 4.5),  #  Moderate
            a >= 4.5,  #  Good or High
        ]
        b.append(np.select(conditions, [0, 1, 2, 3], default=np.nan))  #  add to list
    return accuracy_score(eco_true[0], eco_pred[0])


def stepwise_selection(subset, dummies, data, dfDummies, years, select_all=False):
    """Forward stepwise selection of predictors p to include in the model."""
    predictors = ["No dummies"] + dummies  #  list of possible predictors to include
    selected = []  #  empty list for storing selected predictors
    current_score, best_new_score = 0.0, 0.0  #  initial scores

    # DataFrame for storing accuracy scores by year and calculating weighted average
    scores = pd.DataFrame(subset.count(), index=years, columns=["n"]).astype(int)
    scores.loc["Total", "n"] = np.nan  #  row to calculate weighted average of scores

    # DataFrame for storing ecological status by year and calculating weighted average
    status = pd.DataFrame(subset.count(), index=subset.columns, columns=["n"])
    status["Obs"] = (subset < 4.5).sum() / status["n"]  #  ecological status < good
    status.loc["Total", "Obs"] = (status["Obs"] * status["n"]).sum() / status["n"].sum()

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

            # Subset to the waterbodies included in the subset
            dfImpSubset = dfImp.loc[subset.index, subset.columns]

            # Store predicted share with less than good ecological status
            sta[df.name] = (dfImpSubset[subset.columns] < 4.5).sum() / len(subset)

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

                # Accuracy of ecological status implied by predicted fauna class
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

            elif select_all is True:
                break  #  proceed to select every predictor in the order they are listed

        best_new_score = max(scores_total)  #  best accuracy score among predictors

        if select_all is True:
            current_score = best_new_score  #  update current score to continue process

            # Move the given dummy from the list of predictors to the list of selected
            selected.append(predictors.pop(0))

            # Save scores and status by year subject to the selected set of predictors
            for a, b in zip([scores, status], [sco, sta]):
                a[names[0]] = b[names[0]]  #  scores & status by year for predictor

        elif best_new_score > current_score:
            current_score = best_new_score  #  update current score
            i = scores_total.index(best_new_score)  #  index for predictor w. best score

            # Move dummy with the best new score from the list of predictors to selected
            selected.append(predictors.pop(i))

            # Save scores and status by year subject to the selected set of predictors
            for a, b in zip([scores, status], [sco, sta]):
                a[names[i]] = b[names[i]]  #  scores & status by year for best predictor

        else:  #  if best_new_score == current_score (i.e., identical accuracy score)
            break  #  stop selection (including the predictor would increase variance)

        if p == "No dummies":
            selected = []  #  after baseline model, start actual stepwise selection

        elif predictors == []:  #  if all predictors have been included
            break  #  stop stepwise selection

    # Total number of observations that LOO-CV was performed over
    for s in (scores, status):
        s.loc["Total", "n"] = s["n"].sum()

    # Save accuracy scores and share with less than good ecological status to CSV
    if subset is sparse:
        scores.to_csv("output/streams_eco_imp_accuracy_sparse.csv")
        status.to_csv("output/streams_eco_imp_LessThanGood_sparse.csv")
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

# Read DataFrames for observed biophysical indicator and typology
dfIndObs = pd.read_csv("output/streams_ind_obs.csv", index_col="wb")
dfIndObs.columns = dfIndObs.columns.astype(int)
dfVP = pd.read_csv("output\\streams_VP.csv", index_col="wb")

# Share of waterbodies by number of non-missing values
for n in range(0, len(dfIndObs.columns) + 1):
    n, round(100 * sum(dfIndObs.notna().sum(axis=1) == n) / len(dfIndObs), 2)  # percent

# Subset of rows where only 1-3 values are non-missing
sparse = dfIndObs[dfIndObs.notna().sum(axis=1).isin([1, 2, 3])]
sparse.count()  #  lowest number of non-missing values with support in all years
sparse.count().sum()  #  5453 non-missing values in total to loop over with LOO-CV

# Include ecological status as assessed in basis analysis for VP3
basis = dfVP[["til_oko_bb"]].copy()  #  bottom fauna measured as DVFI fauna class

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
basis.columns = ["Basis"]
basis["Basis"].unique()

# Transform ecological status to fauna class (to match the scale used for imputation)
conditions = [
    basis["Basis"] == 0.0,  # Bad
    basis["Basis"] == 1.0,  #  Poor
    basis["Basis"] == 2.0,  #  Moderate
    basis["Basis"] == 3.0,  #  Good
    basis["Basis"] == 4.0,  #  High
]
# Fauna class ranges from 1-7 where Poor eco status is FK2-FK3 and Moderate is FK5-FK6
basis["Basis"] = np.select(conditions, [1, 2.5, 4, 5.5, 7], default=np.nan)

# Merge DataFrames for observed biophysical indicator with basis analysis for VP 3
dfObs = dfIndObs.merge(basis, on="wb")

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
for a, b in zip([dfIndObs, sparse], [dfDistrict, dfSparse]):
    # Subset dfDistrict to streams where ecological status is observed at least once
    obs = b.loc[a.notna().any(axis=1)]  #  6151 out of 6703 streams in dfDistrict

    # df for storing number of observed streams and yearly distribution by dummies
    d = pd.DataFrame(a.count(), index=a.columns, columns=["n"]).astype(int)

    # Yearly distribution of observed streams by typology and district
    for c in cols:
        d[c] = 100 * b[b[c] == 1].count() / b.count()
        d.loc["Obs of n", c] = 100 * len(obs[obs[c] == 1]) / len(obs)
        d.loc["Obs of all", c] = 100 * len(obs[obs[c] == 1]) / len(dfVP)
        d.loc["All in VP3", c] = 100 * len(dfDistrict[dfDistrict[c] == 1]) / len(dfVP)
    d.loc["Obs of n", "n"] = len(obs)  #  number of streams observed at least once
    d.loc["Obs of all", "n"] = len(dfVP)  #  number of streams in VP3
    d.loc["All in VP3", "n"] = len(dfVP)  #  number of streams in VP3

    if b is dfSparse:
        VPstats["Sparse subset"] = d.loc["Obs of n", :]  #  distribution in sparse df
        d.to_csv("output/streams_VP_stats_sparse.csv")  #  save to CSV
    else:
        VPstats["Observed subset"] = d.loc["Obs of n", :]  #  distribution of observed
        VPstats["All in VP3"] = d.loc["All in VP3", :]  #  distribution of all in VP3
        d.to_csv("output/streams_VP_stats.csv")  #  save distributions to CSV
VPstats  #  sparse has underrepresentation of Large and DK2 (share < 50% of average VP3)


########################################################################################
#   2. Multivariate feature imputation (note: Forward Stepwise Selection takes ~5 days)
########################################################################################
# # Example data for testing Forward Stepwise Selection with LOO-CV (takes ~5 seconds)
# dfIndObs = pd.DataFrame(
#     {
#         1988: [2.5, 3.0, 3.5, 4.0, np.nan, 5.0],
#         1989: [2.6, 3.1, 3.6, np.nan, 4.6, 5.1],
#         1990: [2.7, 3.2, np.nan, 4.2, 4.7, 5.2],
#         1991: [2.8, np.nan, 3.8, 4.3, 4.8, 5.3],
#         1992: [np.nan, 3.4, 3.9, 4.4, 4.9, 5.4],
#         1993: [3.0, 3.5, 3.8, 4.4, 5.1, 5.5],
#     }
# )
# dfIndObs.index.name = "wb"
# sparse = dfIndObs[dfIndObs.notna().sum(axis=1) == 5]
# dfObs = dfIndObs.copy()
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
    subset=dfIndObs, dummies=selectedSparse, select_all=True, **kwargs
)
scores
status


### temporary correction of a mistake in the original code
# DataFrame for storing accuracy scores by year and calculating weighted average
scores = pd.DataFrame(dfIndObs.count(), index=years, columns=["n"]).astype(int)

# DataFrame for storing ecological status by year and calculating weighted average
status = pd.DataFrame(dfIndObs.count(), index=dfIndObs.columns, columns=["n"])

# Mean ecological status by year for the n observations that don'nt have missing values
status["Obs"] = (dfIndObs < 4.5).sum() / status["n"]  #  ecological status < good
status.loc["Total", "Obs"] = (status["Obs"] * status["n"]).sum() / status["n"].sum()

# Leave-one-out cross-validation (LOO-CV) loop over every observed stream and year
df1 = dfNatural.drop(columns=["Small", "Medium", "Large"])
df2 = dfNatural.drop(columns=["Medium", "Large"])
df1.name = ", ".join(["Soft bottom", "Natural"])  #  name model
df2.name = ", ".join(["Soft bottom", "Natural", "Small"])  #  name model
for df in (df1, df2):  #  LOO-CV with different dummies
    # Impute missing values based on all observations (without cross-validation)
    df_imp = pd.DataFrame(
        imputer.fit_transform(np.array(df)), index=df.index, columns=df.columns
    )
    # From imputed data, store estimated share with less than good ecological status
    status[df.name] = (df_imp[dfIndObs.columns] < 4.5).sum() / len(df)

    # For each year t, apply leave-one-out cross-validation
    print(df.name, "used for imputation. LOO-CV of observed streams each year:")
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

        # Accuracy of ecological status after converting DVFI fauna index for streams
        accuracy = AccuracyScore(Y["true"], Y["pred"])

        # Save accuracy score each year to DataFrame for scores
        scores.loc[t, df.name] = accuracy

    # Totals weighted by number of observations
    for s in (scores, status):
        s.loc["Total", df.name] = (s[df.name] * s["n"]).sum() / s["n"].sum()

    # Save accuracy scores and share with less than good ecological status to CSV
    scores.to_csv("output/streams_eco_imp_accuracy_" + df.name + ".csv")
    status.to_csv("output/streams_eco_imp_LessThanGood_" + df.name + ".csv")


########################################################################################
#   3. Visualization: Accuracy and share with less than good ecological status by year
########################################################################################
# Skip step 2 by reading DataFrames of accuracy score and ecological status from CSV
# scores = pd.read_csv("output/streams_eco_imp_accuracy.csv", index_col=0)
sco = scores.drop(columns="n").drop("Total")
# status = pd.read_csv("output/streams_eco_imp_LessThanGood.csv", index_col=0)
imp = status.drop(columns=["n", "Obs"]).drop("Total")  #  imputed status by predictors
obs = status[["Obs"]].drop("Total")  #  df for ecological status of observed streams
obs.columns = ["Observed"]  #  rename 'Obs' to 'Observed'
sta = imp.merge(obs, left_index=True, right_index=True)  #  add Observed as last column

# Bar plot accuracy scores
f1 = sco.plot(
    kind="bar", ylabel="Accuracy in predicting observed ecological status"
).get_figure()
f1.savefig("output/streams_eco_imp_accuracy.pdf", bbox_inches="tight")

# Plot share of streams with less than good ecological status
f2 = sta.plot(
    ylabel="Share of streams with less than good ecological status"
).get_figure()
f2.savefig("output/streams_eco_imp_LessThanGood.pdf", bbox_inches="tight")
