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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm

# import matplotlib.colors
# import seaborn as sns
# To use this experimental feature, we need to explicitly ask for it:
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.metrics import accuracy_score

# from sklearn.kernel_approximation import Nystroem
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.neighbors import KNeighborsRegressor


# Function for score
def AccuracyScore(y_true, y_pred):
    """Return average precision (AP) of predicted ecological status compared to known ecological status after converting DVFI fauna index for streams to index of ecological status."""
    eco_true, eco_pred = [], []
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
for RW in ["RW1", "RW2", "RW3", "RW4", "RW5", "RW6"]:
    len(df_VP[df_VP.ov_typ == RW]), round(
        100 * len(df_VP[df_VP.ov_typ == RW]) / len(df_VP)
    )
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
dfTypology = dfIndicator.merge(
    d[["small", "medium", "large", "softBottom"]], how="inner", on="wb"
)


########################################################################################
#   2. Multivariate feature imputation - leave-one-out CV makes it very slow!
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

# Example data used to test the code below (takes ~1 second rather than ~ hours to run)
# dfIndicator = pd.DataFrame(
#     {
#         1988: [0.5, 1.0, 1.5, 2.0, np.nan],
#         1989: [0.6, 1.1, 1.6, np.nan, 2.6],
#         1990: [0.7, 1.2, np.nan, 2.2, 2.7],
#         1991: [0.8, np.nan, 1.8, 2.3, 2.8],
#         1992: [np.nan, 1.4, 1.9, 2.4, 2.9],
#     }
# )
# years = dfIndicator.columns
# dfTypology = dfIndicator.copy()
# dfTypology["small"] = [1, 1, 0, 0, 0]

# Leave-one-out cross-validation (LOO-CV) loop over every observed stream and year
scores = pd.DataFrame()  #  empty DataFrame for accuracy scores by year
dfIndicator.name = "No typology"
dfTypology.name = "Typology"
for df in (dfIndicator, dfTypology):
    print(df.name, "used for imputation. LOO-CV applied to observed streams each year:")
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
        acc = AccuracyScore(Y["true"], Y["pred"])

        # Save accuracy score each year to DataFrame for scores
        scores.loc[t, df.name] = acc

scores

# Save accuracy to CSV
scores.to_csv("output/streams_eco_imp_accuracy.csv")

# Total accuracy, i.e., weight by observations: dfIndicator.count()

########################################################################################
#   4. Visualization: Distribution by year                                    #
########################################################################################
b = "#1f77b4"  #  blue from d3.scale.category10c()
o = "#ff7f0e"  #  orange from d3.scale.category10c()


# plot accuracy scores - find pd.plot() bar example!; wider page margins in appendix?
fig, ax = plt.figure(figsize=(12, 7.4))  #  create new figure
plt.bar(ax - 0.2, scores["No typology"], 0.4, label="No typology")
plt.bar(ax + 0.2, scores["Typology"], 0.4, label="Typology")
ax.set(title="Accuracy of imputation using LOO-CV", ylabel="Accuracy score")
ax.set_xticks(np.array(years))
plt.legend()
plt.show()
fig.savefig("output/streams_eco_imp_accuracy.pdf", bbox_inches="tight")

for t in xval:
    ax.bar(
        t,
        scores[t],
        xerr=stds_diabetes[t],
        color=colors[t],
        alpha=0.6,
        align="center",
    )

ax.set_title("Imputation Techniques with Diabetes Data")
ax.set_xlim(left=np.min(mses_diabetes) * 0.9, right=np.max(mses_diabetes) * 1.1)
ax.set_yticks(xval)
ax.set_xlabel("MSE")
ax.invert_yaxis()
ax.set_yticklabels(x_labels)
