"""
Name:       imputation.py

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
#   0. Function
########################################################################################
import os

import numpy as np
import pandas as pd
import tqdm

# To use this experimental feature, we need to explicitly ask for it:
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.metrics import accuracy_score


# Function for score
def AccuracyScore(y_true, y_pred):
    """Convert DVFI fauna index for streams to index of ecological status and return accuracy score, i.e., the share of observed streams each year where predicted ecological status matches the true ecological status (which LOO-CV omits)."""
    eco_true, eco_pred = [], []  #  empy lists for storing transformed observations
    for a, b in zip([y_true, y_pred], [eco_true, eco_pred]):
        # Categorical variable for ecological status: Bad, Poor, Moderate, Good, High
        conditions = [
            a < 1.5,
            (a >= 1.5) & (a < 3.5),
            (a >= 3.5) & (a < 4.5),
            (a >= 4.5) & (a < 6.5),
            a >= 6.5,
        ]
        b.append(np.select(conditions, [0, 1, 2, 3, 4], default=np.nan))
    return accuracy_score(eco_true[0], eco_pred[0])


########################################################################################
#   1. Setup
########################################################################################
# Specify the working directory of the operating system
os.chdir(r"C:\Users\au687527\GitHub\GreenGDP\gis")

# Read longitudinal data for streams (DVFI fauna index)
dfIndicator = pd.read_csv("output/streams_ind_obs.csv", index_col="wb")
dfIndicator.columns = dfIndicator.columns.astype(int)
years = dfIndicator.columns

# Create typology dummies
df_VP = pd.read_csv("output\\streams_VP.csv", index_col="wb")
d = pd.get_dummies(df_VP["ov_typ"]).astype(int)
d["softBottom"] = d["RW4"] + d["RW5"]
d.columns = [
    "small",
    "medium",
    "large",
    "smallSoftBottom",
    "mediumSoftBottom",
    "softBottom",
]

# Merge DataFrames for typology and DVFI fauna index
col = ["small", "medium", "large", "softBottom"]
dfTypology = dfIndicator.merge(d[col], how="inner", on="wb")

# Typology for observed water bodies by year
typ = pd.DataFrame(index=years)  #  empty DataFrame to store typology by year
for c in col:
    typ[c] = 100 * dfTypology[dfTypology[c] == 1].count() / dfTypology.count()
    typ.loc["all", c] = 100 * len(d[d[c] == 1]) / len(d)
typ.to_csv("output/streams_VP_stats.csv")


########################################################################################
#   2. Multivariate feature imputation (note: LOO CV takes ~1 day to run for each model)
########################################################################################
# Iterative imputer using the BayesianRidge() estimator with increased tolerance
imputer = IterativeImputer(random_state=0, tol=1e-1)

# List number of observations and mean (before and after imputation) by year
# pd.concat(
#     [
#         dfIndicator.count(),
#         dfIndicator.mean(),
#         pd.DataFrame(
#             imputer.fit_transform(np.array(dfIndicator)), columns=dfIndicator.columns
#         ).mean(),
#         pd.DataFrame(
#             imputer.fit_transform(np.array(dfTypology)), columns=dfTypology.columns
#         ).mean(),
#     ],
#     keys=["count_obs", "mean_obs", "mean_imp", "mean_imp_dummies"],
#     axis=1,
# )

# Example data used to test the code below (takes ~1 second rather than ~2 days to run)
# dfIndicator = pd.DataFrame(
#     {
#         1988: [0.5, 1.0, 1.5, 2.0, np.nan],
#         1989: [0.6, 1.1, 1.6, np.nan, 2.6],
#         1990: [0.7, 1.2, np.nan, 2.2, 2.7],
#         1991: [0.8, np.nan, 1.8, 2.3, 2.8],
#         1992: [np.nan, 1.4, 1.9, 2.4, 2.9],
#     }
# )
# dfTypology = dfIndicator.copy()
# dfTypology["small"] = [1, 1, 0, 0, 0]
# years = dfIndicator.columns

# Data subset to measure the prediction improvement from including 1988
# dfIndicator = dfIndicator.drop(columns=1988)
# dfTypology = dfTypology.drop(columns=1988)
# years = [1989]
# accuracy dfIndicator & dfTypology in 1989 drops by  up to if omitting year 1988
#   0.6073825503355704 & 0.6191275167785235 respectively

# DataFrame for storing accuracy scores by year and calculating weighted average
scores = pd.DataFrame(dfIndicator.count(), index=years, columns=["obs"])

# Leave-one-out cross-validation (LOO-CV) loop over every observed stream and year
dfIndicator.name = "No typology"
dfTypology.name = "Typology"
for df in (dfIndicator, dfTypology):
    print("Total:", (scores[df.name] * scores["obs"]).sum() / scores["obs"].sum())
    print(df.name, "used for imputation. LOO-CV of observed streams each year:")
    for t in tqdm.tqdm(years):
        y = df[df[t].notnull()].index
        Y = pd.DataFrame(index=y)
        Y["true"] = df.loc[y, t]
        Y["pred"] = pd.NA
        for i in y:
            X = df.copy()
            X.loc[i, t] = pd.NA
            # Fit imputer and transform the dataset
            X_imp = pd.DataFrame(
                imputer.fit_transform(np.array(X)), index=X.index, columns=X.columns
            )
            Y.loc[i, "pred"] = X_imp.loc[i, t]

        # Accuracy of ecological status after converting DVFI fauna index for streams
        accuracy = AccuracyScore(Y["true"], Y["pred"])

        # Save accuracy score each year to DataFrame for scores
        scores.loc[t, df.name] = accuracy

    # Total accuracy (weighted by number of observations)
    print("Total:", (scores[df.name] * scores["obs"]).sum() / scores["obs"].sum())
    # accuracy dfIndicator - accuracy dfTypology (improvement of .14 percentage points)
    #   0.6241328429537296 - 0.6255202942277622

# Save accuracy scores to CSV
scores.to_csv("output/streams_eco_imp_accuracy.csv")

########################################################################################
#   4. Visualization: Distribution by year                                    #
########################################################################################
# Read accuracy scores from CSV
scores = pd.read_csv("output/streams_eco_imp_accuracy.csv", index_col=0)
s = scores[["No typology", "Typology"]]

# Plot accuracy scores
fig = s.plot(kind="bar", figsize=(12, 6), ylabel="Accuracy score").get_figure()
fig.savefig("output/streams_eco_imp_accuracy.pdf", bbox_inches="tight")
