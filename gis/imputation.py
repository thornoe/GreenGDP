"""
Name:       script.py

Label:      Construct and map longitudinal data of ecological status of streams.

Summary:    ThorNoe.github.io/GNNP/ explains the approach and methodology.

Rqmts:      ArcGIS Pro must be installed on the system and be up to date.

Usage:      This script supports WaterbodiesScriptTool in the gis.tbx toolbox.
            See GitHub.com/ThorNoe/GNNP for instructions to run or update it all.

Created:    25/03/2020
Author:     Thor Donsby Noe
"""

###############################################################################
#   0. Imports                                                                #
###############################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors
import seaborn as sns
import os


###############################################################################
#   1. Setup                                                                  #
###############################################################################
os.chdir(r'C:\Users\jwz766\Documents\GitHub\gnnp\gis')

# Read wide format for ecological status of waterbodies
w = pd.read_csv('output/streams_ecological_status.csv').drop('g_len', axis=1)

# Stack into long format
# l = pd.melt(w, id_vars='g_del_cd', var_name='y', value_name='')
#
# # Sort by no. of missing values
# w['nan'] = w.shape[1] - w.count(axis=1)
# l = l.merge(w[['g_del_cd', 'nan']], how='left', on='g_del_cd')\
#      .sort_values(['nan', 'g_del_cd'], ascending=False).drop(['nan'], axis=1)
# l.columns = ['id', 'y', 'e']
# l.head(2)

w = w.sort_values(['nan'], ascending=False).drop(['nan', 'g_del_cd'], axis=1)

w.head(2)
w.tail(2)


###############################################################################
#   2. Visualization                                                          #
###############################################################################

# missing values graph (heatmap):
def mvg(frame, obs, time, var, Yname, prefix):
    df = frame.copy()
    df.fillna(0, inplace=True)
    cm = sns.xkcd_palette(['grey', 'red', 'orange', 'yellow', 'green', 'blue'])
    plt.figure(figsize=(12, 7.4))
    ax = sns.heatmap(df, cmap=cm, cbar=False,
                     cbar_kws={'ticks': [0, 1, 2, 3, 4, 5, 6]})
    ax.set(yticklabels=[])
    plt.ylabel(Yname+ " (N=" + str(len(df)) + ")", fontsize=14)
    plt.xlabel("")
    plt.title(('Ecological status of ' + Yname + ':' + '\nmissing values (grey), bad (red), poor (orange), moderate (yellow), good (green), high (blue)'),
              fontsize=14)
    plt.tight_layout()
    plt.savefig('output/' + prefix + '_' + Yname+'.png', bbox_inches='tight')
    plt.show()

mvg(w, 'id', 'y', 'e', 'streams', 'missing')
