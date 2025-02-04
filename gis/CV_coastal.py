"""
Name:       CV_coastal.py

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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
from cycler import cycler
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.metrics import accuracy_score

# Multivariate imputer using BayesianRidge() estimator with increased tolerance
imputer = IterativeImputer(tol=1e-1, max_iter=100, random_state=0)

# Color-blind-friendly color scheme for qualitative data by Tol: personal.sron.nl/~pault
colors = {
    "blue": "#1965B0",
    "light blue": "#5289C7",
    "jordy blue": "#7BAFDE",
    "green": "#4EB265",
    "pale green": "#CAE0AB",
    "yellow": "#F7F056",
    "orange": "#EE8026",
    "red": "#DC050C",
    "dark red": "#72190E",
    "grey": "#BBBBBB",
}

# Set the default property-cycle and figure size for pyplots
color_cycler = cycler(color=list(colors.values()))  #  color cycler with 10 colors
linestyle_cycler = cycler(
    linestyle=["-", "-", "-", "--", "--", "--", ":", ":", ":", "-"]  # 10 styles
)
plt.rc("axes", prop_cycle=(color_cycler + linestyle_cycler))
plt.rc("figure", figsize=[10, 6.2])  #  golden ratio


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


# Function for accuracy score of predicted ecological status
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


def stepwise_selection(subset, dummies, data, dfDummies, years):
    """Forward stepwise selection of predictors p to include in the model."""
    predictors = ["No dummies"] + dummies  #  list of possible predictors to include
    selected = []  #  empty list for storing selected predictors
    current_score, best_new_score = 0.0, 0.0  #  initial scores
    # DataFrame for storing accuracy scores by year and calculating weighted average
    scores = pd.DataFrame(subset.count(), index=years, columns=["n"]).astype(int)
    scores.loc["Total", "n"] = np.nan  #  row to calculate weighted average of scores
    scores_all = scores.copy()  #  scores for all sets of predictors being tested

    # DataFrame for storing ecological status by year and calculating weighted average
    status = scores.copy()  #  likewise, covers the years in the natural capital account
    status["Obs"] = (subset[years] < 2.5).sum() / status["n"]  #  eco status < good
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

        best_new_score = max(scores_total)  #  best score

        if best_new_score > current_score:
            current_score = best_new_score  #  update current score
            i = scores_total.index(best_new_score)  #  index for predictor w. best score

            # Move dummy with the best new score from the list of predictors to selected
            selected.append(predictors.pop(i))

            # Save scores and status by year subject to the selected set of predictors
            for a, b in zip([scores, status], [sco, sta]):
                a[names[i]] = b[names[i]]  #  scores & status by year for best predictor

            if p == "No dummies":
                selected = []  #  after baseline model, start actual stepwise selection

        else:  #  if best_new_score == current_score (i.e., identical accuracy score)
            break  #  stop selection (including the predictor would increase variance)

        if predictors == []:  #  if all predictors have been included in the best model
            break  #  stop stepwise selection

    # Total number of observations that LOO-CV was performed over
    for s in (scores, status, scores_all, status_all):
        s.loc["Total", "n"] = s["n"].sum()

    # Save accuracy scores and share with less than good ecological status to CSV
    scores.to_csv("output/coastal_eco_imp_accuracy.csv")
    status.to_csv("output/coastal_eco_imp_LessThanGood.csv")
    scores_all.to_csv("output/coastal_eco_imp_accuracy_all.csv")
    status_all.to_csv("output/coastal_eco_imp_LessThanGood_all.csv")

    return selected, scores, status  #  selected predictors; scores and stats by year


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
#   2. Subset selection of dummies for multivariate imputation (note: CV takes ~53hours)
########################################################################################
# Forward stepwise selection of dummies - CV over all observed values in coastal waters
kwargs = {
    "subset": dfEcoObs,
    "dummies": cols_abbreviations,
    "data": dfObs,
    "dfDummies": dfTyp,
    "years": years,
}
selected, scores, status = stepwise_selection(**kwargs)
scores
status
selected = ["Vu", "B", "K", "No", "Se", "Vf", "Ø", "Fj"]
[dicts.get(item, item) for item in selected]
# Dummies were chosen via forward stepwise selection in the following order: water exchange, Belt Sea, Kattegat, North Sea, sediment, North Sea fjord, Baltic Sea, and fjord

########################################################################################
#   3. Visualization: Accuracy and share with less than good ecological status by year
########################################################################################
# Skip step 2 by reading DataFrames of accuracy score and ecological status from CSV
# scores = pd.read_csv("output/coastal_eco_imp_accuracy.csv", index_col=0)
# status = pd.read_csv("output/coastal_eco_imp_LessThanGood.csv", index_col=0)

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
