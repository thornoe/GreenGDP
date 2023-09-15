"""
Name:       script.py

Label:      Impute missing values in longitudinal data of ecological status of waterbodies.

Summary:    ThorNoe.GitHub.io/GreenGDP explains the approach and methodology.

Usage:      This script supports WaterbodiesScriptTool in the gis.tbx toolbox.
            See GitHub.com/ThorNoe/GreenGDP for instructions to run or update it all.

Licence:    MIT Copyright (c) 2023
Author:     Thor Donsby Noe
"""

###############################################################################
#   0. Imports                                                                #
###############################################################################
import os
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.colors
# import seaborn as sns

# To use this experimental feature, we need to explicitly ask for it:
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
# from sklearn.impute import SimpleImputer
# from sklearn.linear_model import BayesianRidge, Ridge
# from sklearn.kernel_approximation import Nystroem
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.pipeline import make_pipeline
# from sklearn.model_selection import cross_validate



###############################################################################
#   1. Setup                                                                  #
###############################################################################
os.chdir(r'C:\Users\au687527\GitHub\GreenGDP\gis')
year_first = 1989
year_last  = 2020
years = list(range(year_first, year_last+1))
years

# Read wide format for ecological status of waterbodies
df = pd.read_csv('output/streams_ind_obs.csv') #.drop('g_del_cd', axis=1)
df.describe()

### Set seed
s=42

###############################################################################
#   2. Multivariate feature imputation                                        #
###############################################################################
imp = IterativeImputer(max_iter=10, random_state=s, min_value=1, max_value=7, keep_empty_features=True))
df_imp = imp.fit_transform(np.array(df.drop('g_del_cd', axis=1)))
df_imp.describe()




###############################################################################
#   2.a k-fold cross validation                                               #
###############################################################################

### randomize (shuffle data order)
df = df.reindex(np.random.RandomState(seed=s).permutation(df.index))
df.head()

### make the folds


### make the train and test dataframe


### randomly remove a share of actual observations in test dataframe


### train imputation


### validate imputation on test dataframe



###############################################################################
#   2.b Imputation sandbox                                                    #
###############################################################################
### Simple imputer
imp_mean = SimpleImputer(strategy='mean')
df_mean  = pd.DataFrame(imp_mean.fit_transform(df), columns=df.columns)
df_mean.describe()
imp_mean.fit_transform(df)
df.head()

array = imp_mean.fit_transform(df)
type(array)
array
array.shape
df_mean  = pd.DataFrame(array)
df_mean

### Iterative imputer
imp_mean = IterativeImputer(random_state=0)
df_mean  = pd.DataFrame(imp_mean.fit_transform(df), columns=df.columns)
df_mean.describe()

scores = cross_validate(imp_mean, df, cv=10,
                        scoring=('neg_mean_absolute_error', 'neg_mean_squared_error'),
                        return_train_score=True)



###############################################################################
#   3. Visualization: Missing Values                                          #
###############################################################################
# Share of streams with non-missing value at least one year (76%)
100*len(df[list(map(str, years))].dropna(how="all"))/len(df)

# Share of streams with non-missing values by year
df[list(map(str, years))].count()/len(df)
np.mean(df[list(map(str, years))].count()/len(df))

# Sort by number of missing values
df['nan'] = df.shape[1] - df.count(axis=1)
df = df.sort_values(['nan'], ascending=False)[list(map(str, years))]

# missing values graph (heatmap):
def mvg(frame, waterbodyType, suffix):
    df = frame.copy()
    df.fillna(0, inplace=True)
    cm = sns.xkcd_palette(['grey', 'red', 'orange', 'yellow', 'green', 'blue'])
    plt.figure(figsize=(12, 7.4))
    ax = sns.heatmap(df, cmap=cm, cbar=False,
                     cbar_kws={'ticks': [0, 1, 2, 3, 4, 5, 6]})
    ax.set(yticklabels=[])
    plt.ylabel(waterbodyType+" (N="+str(len(df))+")", fontsize=14)
    plt.xlabel("")
    plt.title(('Ecological status of ' + waterbodyType + ':' + '\nmissing value (grey), bad (red), poor (orange), moderate (yellow), good (green), high (blue)'),
              fontsize=14)
    plt.tight_layout()
    plt.savefig('output/'+waterbodyType+'_eco_'+suffix+'.png', bbox_inches='tight')
    plt.show()

mvg(df, 'streams', 'missing')

###############################################################################
#   4. Visualization: Distribution by year                                    #
###############################################################################
# Original sample
df.mean()
df.std()

df

df_mean.mean(), df_mean.std()