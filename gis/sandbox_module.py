"""
Name:       sandbox_module.py

Label:      Implementation of the parts of script_module.py that does not use ArcPy.

Summary:    ThorNoe.GitHub.io/GreenGDP explains the overall approach and methodology.

Rqmts:      Does not require ArcGIS Pro to be installed.

Usage:      Compared to script_module.py, this module does not contain ArcPy commands.

            Therefore, it lags the following 6 functions: get_data(), get_fc_from_WFS(), map_book(), values_by_catchment_area(), observed_indicator(), longitudinal()

            Instead, this class only contains 7 functions that are all nested:
            - impute_missing() calls:
                - ecological_status(), which calls:
                    - indicator_to_status()
                    - missing_values_graph()
            - decompose() calls:
                - valuation(), which calls:
                    - BT()

            Descriptions can be seen under each function.

License:    MIT Copyright (c) 2024–2026
Author:     Thor Donsby Noe
"""

import os
import sys
import traceback
import urllib.request

# import arcpy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from cycler import cycler

# from scipy import interpolate
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer


class Water_Quality:
    """Class for all data processing and mapping functions"""

    def __init__(
        self,
        yearFirst,
        yearLast,
        dataFilenames,
        linkageFilenames,
        # WFS_ServiceURL,
        # WFS_featureClassNames,
        # WFS_fieldsInFeatureClass,
        # WFS_replaceFeatureClasses,
        # keepGeodatabase,
    ):
        self.year_first = yearFirst
        self.year_last = yearLast
        self.years = list(range(yearFirst, yearLast + 1))
        self.data = dataFilenames
        self.linkage = linkageFilenames
        # self.wfs_service = WFS_ServiceURL
        # self.wfs_fc = WFS_featureClassNames
        # self.wfs_fields = WFS_fieldsInFeatureClass
        # self.wfs_replace = WFS_replaceFeatureClasses
        # self.keep_gdb = keepGeodatabase
        self.path = os.getcwd()
        # self.arcPath = self.path + "\\gis.gdb"

        # Color-blind-friendly color scheme by Paul Tol: https://personal.sron.nl/~pault
        colors = {
            "blue": "#4477AA",
            "cyan": "#66CCEE",
            "green": "#228833",
            "yellow": "#CCBB44",
            "red": "#EE6677",
            "purple": "#AA3377",
            "grey": "#BBBBBB",
        }

        # Set the default property-cycle and figure size for pyplots
        color_cycler = cycler(color=list(colors.values()))  #  color cycler w. 7 colors
        linestyle_cycler = cycler(linestyle=["-", "--", ":", "-", "--", ":", "-."])  # 7
        plt.rc("axes", prop_cycle=(color_cycler + linestyle_cycler))
        plt.rc("figure", figsize=[10, 5])  #  golden ratio is 10 × 6.18

        # Set the default display format for floating-point numbers
        pd.options.display.float_format = "{:.2f}".format
        # pd.reset_option("display.float_format")

        # Setup ArcPy
        # arcpy.env.workspace = self.arcPath  # set the ArcPy workspace
        # arcpy.env.overwriteOutput = True  # set overwrite option

        # Check that folders for data, output, and linkage files exist or create them
        self.get_data()

        # Get feature class for coastal catchment areas from the WFS service
        # self.get_fc_from_WFS("catch")

    def get_data(self):
        """Function to check that the folders and their files exist.
        Otherwise creates the folders and downloads the files from GitHub."""
        try:
            # Dictionary for folders and their data and linkage files
            allFiles = {
                "data": [a for b in list(self.data.values()) for a in b],
                "linkage": [a for b in list(self.linkage.values()) for a in b],
                "output": [],
            }

        except:
            # Report severe error messages
            tb = sys.exc_info()[2]  # get traceback object for Python errors
            tbinfo = traceback.format_tb(tb)[0]
            msg = "Could not set up dictionary with all files:\nTraceback info:\n{0}Error Info:\n{1}".format(
                tbinfo, str(sys.exc_info()[1])
            )
            print(msg)  # print error message in Python
            # arcpy.AddError(msg)  # return error message in ArcGIS
            sys.exit(1)

        for key, filenames in allFiles.items():
            try:
                # Create the folder if it doesn't exist
                newPath = self.path + "\\" + key
                os.makedirs(newPath, exist_ok=True)
                os.chdir(newPath)

                for f in filenames:
                    # Download the files if they don't exist
                    if not os.path.exists(f):
                        try:
                            url = (
                                "https://github.com/thornoe/GreenGDP/raw/master/gis/"
                                + key
                                + "/"
                                + f
                            )
                            urllib.request.urlretrieve(url, f)
                        except urllib.error.URLError as e:
                            ## Report URL error messages
                            urlmsg = "URL error for {0}:\n{1}".format(f, e.reason)
                            print(urlmsg)  # print URL error message in Python
                            # arcpy.AddError(urlmsg)  # return error message in ArcGIS
                            sys.exit(1)
                        except:
                            ## Report other severe error messages
                            tb = sys.exc_info()[
                                2
                            ]  # get traceback object for Python errors
                            tbinfo = traceback.format_tb(tb)[0]
                            msg = "Could not download {0}:\nTraceback info:\n{1}Error Info:\n{2}".format(
                                f, tbinfo, str(sys.exc_info()[1])
                            )
                            print(msg)  # print error message in Python
                            # arcpy.AddError(msg)  # return error message in ArcGIS
                            sys.exit(1)

            except OSError:
                # Report system errors
                tb = sys.exc_info()[2]  # get traceback object for Python errors
                tbinfo = traceback.format_tb(tb)[0]
                OSmsg = "System error for {0} folder:\nTraceback info:\n{1}Error Info:\n{2}".format(
                    key, tbinfo, str(sys.exc_info()[1])
                )
                print(OSmsg)  # print system error message in Python
                # arcpy.AddError(OSmsg)  # return system error message in ArcGIS
                sys.exit(1)

            finally:
                # Change the directory back to the original working folder
                os.chdir(self.path)

    def impute_missing(self, j, dfEcoObs, dfVP, index):
        # dfEcoObs, dfVP, index = df_eco_obs, df_VP, index_sorted
        """Impute ecological status for all water bodies from the observed indicator."""
        try:
            # Merge observed ecological status each year with Basis Analysis for VP3
            dfEco = dfEcoObs.merge(dfVP[["Basis"]], on="wb")

            if j == "streams":
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

                # Dummies for natural, artificial, and heavily modified water bodies
                natural = pd.get_dummies(dfVP["na_kun_stm"]).astype(int)
                natural.columns = ["Artificial", "Natural", "Heavily modified"]

                # Merge DataFrames for typology and natural water bodies
                typ = typ.merge(natural, on="wb")

                # Dummies used for imputation chosen via Forward Stepwise Selection (CV)
                cols = ["Soft bottom", "Natural", "Large"]

            elif j == "lakes":
                # Convert typology to integers
                typ = dfVP[["ov_typ"]].copy()
                typ.loc[:, "type"] = typ["ov_typ"].str.slice(6).astype(int)

                # Create dummies for high alkalinity, brown, saline, and deep lakes
                cond1 = [(typ["type"] >= 9) & (typ["type"] <= 16), typ["type"] == 17]
                typ["Alkalinity"] = np.select(cond1, [1, np.nan], default=0)
                cond2 = [
                    typ["type"].isin([5, 6, 7, 8, 13, 14, 15, 16]),
                    typ["type"] == 17,
                ]
                typ["Brown"] = np.select(cond2, [1, np.nan], default=0)
                cond3 = [
                    typ["type"].isin([2, 3, 7, 8, 11, 12, 15, 16]),
                    typ["type"] == 17,
                ]
                typ["Saline"] = np.select(cond3, [1, np.nan], default=0)
                cond4 = [typ["type"].isin(np.arange(2, 17, 2)), typ["type"] == 17]
                typ["Deep"] = np.select(cond4, [1, np.nan], default=0)

                # Dummies used for imputation chosen via Forward Stepwise Selection (CV)
                cols = ["Saline", "Brown", "Alkalinity"]

            else:  #  coastal waters
                # Get typology
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

                # Define a function to process each string
                def process_string(s):
                    # Drop the hyphen and everything following it
                    s = s.split("-")[0]

                    # Create a en empty dictionary for relevant abbreviations as keys
                    dummies = {}

                    # Check for abbreviations from dict1 first
                    for abbr in dict1:
                        if abbr in s:
                            dummies[abbr] = 1
                            s = s.replace(
                                abbr, ""
                            )  # Remove the matched abbreviation from the string

                    # Then check for abbreviations from dict2
                    for abbr in dict2:
                        if abbr in s:
                            dummies[abbr] = 1

                    return dummies

                # Apply the function to typ["ov_typ"] to create a df with the dummies
                typ = typ["ov_typ"].apply(process_string).apply(pd.Series)

                # Replace NaN values with 0
                typ = typ.fillna(0).astype(int)

                # Rename the dummies from abbreviations to full names
                dicts = {**dict1, **dict2}  #  combine the dictionaries
                typ = typ.rename(columns=dicts)  #  rename columns to full names

                # Dummies used for imputation chosen via Forward Stepwise Selection (CV)
                cols = [
                    "North Sea",
                    "Kattegat",
                    "Belt Sea",
                    "Baltic Sea",
                    "Fjord",
                    "North Sea fjord",
                    "Water exchange",
                    "Sediment",
                ]

            # Merge DataFrame for observed values with DataFrame for dummies
            dfEcoSelected = dfEco.merge(typ[cols], on="wb")  #  selected predictors

            # Multivariate imputer using BayesianRidge estimator w. increased tolerance
            imputer = IterativeImputer(tol=1e-1, max_iter=100, random_state=0)

            # Fit imputer, transform data iteratively, and drop dummies again
            dfImp = pd.DataFrame(
                imputer.fit_transform(np.array(dfEcoSelected)),
                index=dfEcoSelected.index,
                columns=dfEcoSelected.columns,
            )[dfEcoObs.columns]

            # Calculate a 5-year moving average (MA) for each water body to reduce noise
            dfImpMA = dfImp.T.rolling(window=5, min_periods=3, center=True).mean().T

            # Stats if converting imputed status to categorical scale ∈ {0, 1, 2, 3, 4}
            impStats = self.ecological_status(j, dfImp, dfVP, "imp", index)

            # Stats if converting moving average to categorical scale ∈ {0, 1, 2, 3, 4}
            impStatsMA = self.ecological_status(j, dfImpMA, dfVP, "imp_MA", index)

            return dfImp[self.years], dfImpMA[self.years], impStats, impStatsMA

        except:
            ## Report severe error messages
            tb = sys.exc_info()[2]  # get traceback object for Python errors
            tbinfo = traceback.format_tb(tb)[0]
            msg = "Could not impute biophysical indicator to ecological status for {0}:\nTraceback info:\n{1}Error Info:\n{2}".format(
                j, tbinfo, str(sys.exc_info()[1])
            )
            print(msg)  # print error message in Python
            # arcpy.AddError(msg)  # return error message in ArcGIS
            sys.exit(1)

    def ecological_status(self, j, dfIndicator, dfVP, suffix="obs", index=None):
        # dfIndicator, dfVP, suffix, index = df_ind_obs, df_VP, "obs", None
        """Call indicator_to_status() to convert the longitudinal DataFrame to the EU index of ecological status, i.e., from 0-4 for Bad, Poor, Moderate, Good, and High water quality based on the category and typology of each water body.

        Also call missing_values_graph() to map missing observations by year.

        Create a table of statistics and export it as an html table.

        Print the shore length and share of water bodies observed at least once."""
        try:
            if suffix == "obs":
                # Convert observed biophysical indicator to ecological status
                dfEcoObs = self.indicator_to_status(j, dfIndicator, dfVP)

            else:
                # Imputed ecological status using a continuous scale
                dfEcoObs = dfIndicator.copy()

            # Save CSV of data on mean ecological status by water body and year
            dfEcoObs.to_csv("output\\" + j + "_eco_" + suffix + ".csv")

            # Merge observed ecological status each year with Basis Analysis for VP3
            dfEco = dfEcoObs.merge(dfVP[["Basis"]], on="wb")

            if suffix != "obs":
                # Prepare for statistics and missing values graph
                for t in dfEco.columns:
                    # Convert imp status to categorical scale w. equidistant thresholds
                    conditions = [
                        dfEco[t] < 0.5,  # Bad
                        (dfEco[t] >= 0.5) & (dfEco[t] < 1.5),  #  Poor
                        (dfEco[t] >= 1.5) & (dfEco[t] < 2.5),  #  Moderate
                        (dfEco[t] >= 2.5) & (dfEco[t] < 3.5),  #  Good
                        dfEco[t] >= 3.5,  #  High
                    ]
                    # Ecological status as a categorical index from Bad to High quality
                    dfEco[t] = np.select(conditions, [0, 1, 2, 3, 4], default=np.nan)

            if suffix != "imp_MA":
                # Create missing values graph (heatmap of missing observations by year)
                indexSorted = self.missing_values_graph(j, dfEco, suffix, index)

            # Merge df for observed ecological status with df for characteristics
            dfEcoLength = dfEco.merge(dfVP[["length"]], on="wb")

            # Calculate total length of all water bodies in current water body plan (VP2)
            totalLength = dfEcoLength["length"].sum()

            # Create an empty df for statistics
            stats = pd.DataFrame(
                index=self.years + ["Basis"],
                columns=[
                    "high",
                    "good",
                    "moderate",
                    "poor",
                    "bad",
                    "not good",
                    "known",
                ],
            )

            # Calculate the above statistics for span of natural capital account & basis
            for t in self.years + ["Basis"]:
                y = dfEcoLength[[t, "length"]].reset_index(drop=True)
                y["high"] = np.select([y[t] == 4], [y["length"]])
                y["good"] = np.select([y[t] == 3], [y["length"]])
                y["moderate"] = np.select([y[t] == 2], [y["length"]])
                y["poor"] = np.select([y[t] == 1], [y["length"]])
                y["bad"] = np.select([y[t] == 0], [y["length"]])
                y["not good"] = np.select([y[t] < 3], [y["length"]])
                y["known"] = np.select([y[t].notna()], [y["length"]])

                # Add shares of total length to stats
                knownLength = y["known"].sum()
                stats.loc[t] = [
                    100 * y["high"].sum() / knownLength,
                    100 * y["good"].sum() / knownLength,
                    100 * y["moderate"].sum() / knownLength,
                    100 * y["poor"].sum() / knownLength,
                    100 * y["bad"].sum() / knownLength,
                    100 * y["not good"].sum() / knownLength,
                    100 * knownLength / totalLength,
                ]

            # For imputed ecological status, convert to integers and drop 'known' column
            if suffix != "obs":
                stats = stats.drop(columns="known")

            # Save statistics on mean ecological status by year weighted by shore length
            stats.to_csv("output\\" + j + "_eco_" + suffix + "_stats.csv")

            # Brief analysis of missing observations (not relevant for imputed data)
            if suffix == "obs":
                # Create df limited to water bodies that are observed at least one year
                observed = dfEcoObs.dropna(how="all").merge(
                    dfVP[["length"]],
                    how="inner",
                    on="wb",
                )

                # Report length and share of water bodies observed at least one year
                msg = "{0} km is the total shore length of {1} included in VP3, of which {2}% of {1} representing {3} km ({4}% of total shore length of {1}) have been assessed at least one year. On average, {5}% of {1} representing {6} km ({7}% of total shore length of {1}) are assessed each year.\n".format(
                    round(totalLength),
                    j,
                    round(100 * len(observed) / len(dfEco)),
                    round(observed["length"].sum()),
                    round(100 * observed["length"].sum() / totalLength),
                    round(100 * np.mean(dfEco[self.years].count() / len(dfEco))),
                    round(stats.drop("Basis")["known"].mean() / 100 * totalLength),
                    round(stats.drop("Basis")["known"].mean()),
                )
                print(msg)  # print statistics in Python
                # arcpy.AddMessage(msg)  # return statistics in ArcGIS

                return dfEco[dfEcoObs.columns], stats["not good"], indexSorted

            return stats["not good"]

        except:
            # Report severe error messages
            tb = sys.exc_info()[2]  # get traceback object for Python errors
            tbinfo = traceback.format_tb(tb)[0]
            msg = "Could not create df with {0} ecological status for {1}:\nTraceback info:\n{2}Error Info:\n{3}".format(
                suffix, j, tbinfo, str(sys.exc_info()[1])
            )
            print(msg)  # print error message in Python
            # arcpy.AddError(msg)  # return error message in ArcGIS
            sys.exit(1)

    def indicator_to_status(self, j, dfIndicator, dfVP):
        """Convert biophysical indicators to ecological status."""
        try:
            # Column names for chlorophyll thresholds for lakes or coastal waters
            cols = ["bad", "poor", "moderate", "good"]

            if j == "streams":
                # Copy DataFrame for the biophysical indicator
                df = dfIndicator.copy()

                # Convert DVFI fauna index for streams to index of ecological status
                for t in df.columns:
                    # Set conditions given the official guidelines for conversion
                    conditions = [
                        df[t] < 1.5,  # Bad
                        (df[t] >= 1.5) & (df[t] < 3.5),  #  Poor
                        (df[t] >= 3.5) & (df[t] < 4.5),  #  Moderate
                        (df[t] >= 4.5) & (df[t] < 6.5),  #  Good
                        df[t] >= 6.5,  #  High
                    ]

                    # Ecological status as a categorical scale from Bad to High quality
                    df[t] = np.select(conditions, [0, 1, 2, 3, 4], default=np.nan)

                return df

            elif j == "lakes":
                # Merge df for biophysical indicator with df for typology
                df = dfIndicator.merge(dfVP[["ov_typ"]], on="wb")

                def set_threshold(row):
                    if row["ov_typ"] in ["LWTYPE9", "LWTYPE11", "LWTYPE13", "LWTYPE15"]:
                        return pd.Series(
                            {
                                "bad": 90,
                                "poor": 56,
                                "moderate": 25,
                                "good": 11.7,
                            }
                        )

                    else:
                        return pd.Series(
                            {
                                "bad": 56,
                                "poor": 27,
                                "moderate": 12,
                                "good": 7,
                            }
                        )

                # For df, add the series of thresholds relative to High ecological status
                df[cols] = df.apply(set_threshold, axis=1)
                df = df.drop(columns=["ov_typ"])  #  drop typology column

            else:  #  coastal waters
                # Read table of thresholds of chlorophyll for each coastal water body
                thresholds = pd.read_csv(
                    "linkage\\" + self.linkage[j][1], index_col=0
                ).astype(int)

                # Merge df for biophysical indicator with df for thresholds
                df = dfIndicator.merge(thresholds[cols], on="wb")

            # Convert mean chlorophyll concentrations to index of ecological status
            for t in dfIndicator.columns:
                # Set conditions given the threshold for the typology of each lake
                conditions = [
                    df[t] >= df["bad"],
                    (df[t] < df["bad"]) & (df[t] >= df["poor"]),
                    (df[t] < df["poor"]) & (df[t] >= df["moderate"]),
                    (df[t] < df["moderate"]) & (df[t] >= df["good"]),
                    df[t] < df["good"],
                ]
                # Ordinal scale of ecological status: Bad, Poor, Moderate, Good, High
                df[t] = np.select(conditions, [0, 1, 2, 3, 4], default=np.nan)

            # Drop columns with thresholds
            df = df.drop(columns=cols)

            return df

        except:
            ## Report severe error messages
            tb = sys.exc_info()[2]  # get traceback object for Python errors
            tbinfo = traceback.format_tb(tb)[0]
            msg = "Could not convert biophysical indicator to ecological status for {0}:\nTraceback info:\n{1}Error Info:\n{2}".format(
                j, tbinfo, str(sys.exc_info()[1])
            )
            print(msg)  # print error message in Python
            # arcpy.AddError(msg)  # return error message in ArcGIS
            sys.exit(1)

    def missing_values_graph(self, j, frame, suffix="obs", index=None):
        # frame, suffix, index = dfEco, "obs", None
        """Heatmap visualizing observations of ecological status as either missing or using the EU index of ecological status, i.e., from 0-4 for Bad, Poor, Moderate, Good, and High water quality respectively.
        Saves a figure of the heatmap (using the same index for imputed data)."""
        try:
            # Missing values will be imputed to be around the mean (other things equal)
            frame["Mean"] = frame.iloc[:, :-1].mean(axis=1)
            mean = frame["Mean"].mean()

            # Subset frame to the years of the natural capital accounts and the Mean
            df = frame[self.years[1:] + ["Basis", "Mean"]].copy()

            # Check df for the presence of any missing values
            if df.isna().sum().sum() > 0:
                # Replace missing values with -1 for heatmap
                df.fillna(-1, inplace=True)

                # Specify heatmap to show missing values as white
                colors = ["white", "red", "orange", "yellow", "green", "blue"]
                uniqueValues = [-1, 0, 1, 2, 3, 4]

                # Description for heatmap of observed eco status (instead of fig legend)
                description = "Missing value (white), Bad (red), Poor (orange), Moderate (yellow), Good (green), High (blue)"

            else:
                # Specify heatmap without any missing values (only for imputed coastal)
                colors = ["red", "orange", "yellow", "green", "blue"]
                uniqueValues = [0, 1, 2, 3, 4]
                description = "Bad (red), Poor (orange), Moderate (yellow), Good (green), High (blue)"

            if suffix == "obs":
                # Sort by ecological status in Basis Analysis & observed values
                df = df.replace(-1, mean)  #  set missing as the observed mean
                df = df.sort_values(["Basis", "Mean"], ascending=False)
                df = df.drop(columns=["Mean"]).replace(mean, -1)  # set as -1 again

                # Save index to reuse the order after imputing the missing values
                index = df.index

            else:
                # Sort by ecological status in Basis Analysis & observed values as above
                df = df.drop(columns=["Mean"]).reindex(index)

            if j == "coastal":
                type = "coastal waters"

            else:
                type = j  # type of water body

            # Plot heatmap
            colorMap = sns.xkcd_palette(colors)
            plt.figure(figsize=(10, 11))
            ax = sns.heatmap(
                df,
                cmap=colorMap,
                cbar=False,
                cbar_kws={"ticks": uniqueValues},
            )
            ax.set(yticklabels=[])
            plt.ylabel(
                str(len(df)) + " " + type + " ordered by observed ecological status"
            )
            plt.title(description)
            plt.tight_layout()
            plt.savefig("output\\" + j + "_eco_" + suffix + ".pdf", bbox_inches="tight")

            return index

        except:
            # Report severe error messages
            tb = sys.exc_info()[2]  # get traceback object for Python errors
            tbinfo = traceback.format_tb(tb)[0]
            msg = "Could not create missing values graph (heatmap) for {0} ecological status of {1}:\nTraceback info:\n{2}Error Info:\n{3}".format(
                suffix, j, tbinfo, str(sys.exc_info()[1])
            )
            print(msg)  # print error message in Python
            # arcpy.AddError(msg)  # return error message in ArcGIS
            sys.exit(1)

    def decompose(self, dfBT, factor, baseYear=1990):
        """Decompose development by holding everything else equal at baseYear level"""
        try:
            # Define a function to ready each row for decomposition analysis
            def replace_row(row):
                j, t, v = row.name

                # Fix variables "Q", "ln PSL", and "SL" at base year level for j ≠ driver
                if j != driver:
                    row[["Q", "ln PSL", "SL"]] = df.loc[
                        (j, baseYear, v), ("Q", "ln PSL", "SL")
                    ]

                # Fix "ln y", "D age", and "N" at base year level for variable ≠ driver
                cols = [col for col in ["ln y", "D age", "N"] if col != driver]
                row[cols] = df.loc[(j, baseYear, v), cols]

                return row

            # Empty dictionaries for decomposed costs of pollution and investment value
            CWP_v, CWP_j, CWP, IV_v, IV = {}, {}, {}, {}, {}

            for driver in ["coastal", "lakes", "streams", "ln y", "D age", "N"]:
                # Copy df with the variables needed for the Benefit Transfer function
                df = dfBT.copy()

                # Isolate changes related to driver by holding other things equal
                df = df.apply(replace_row, axis=1)

                # Apply valuation function to decompose the development by driver
                CWP_vj, f = self.valuation(df, factor=factor)  #  CWP in v by category j

                # Costs of Water Pollution in real terms (million DKK, 2023 prices)
                CWP_v[driver] = CWP_vj.sum(axis=1)  #  total CWP in v
                CWP_j[driver] = CWP_vj.groupby("t").sum().rename_axis(None)  #  CWP by j
                CWP[driver] = CWP_j[driver].sum(axis=1)  #  total CWP

                categories = ["coastal", "lakes", "streams"]

                if driver in categories:
                    # Investment Value (IV) of water quality improvement in real terms
                    IV_v[driver] = self.valuation(df, investment=True)  #  IV in v by j

                    # Drop categories j where the given driver has no effect
                    for dict in [CWP_j, IV_v]:
                        cols = [col for col in categories if col != driver]
                        dict[driver] = dict[driver].drop(columns=cols)
                    IV_v[driver] = IV_v[driver][driver]  #  j is redundant (= driver)

                    # IV of water quality improvement in real terms by t and j
                    IV[driver] = IV_v[driver].groupby("t").sum().rename_axis(None)

            # Concatenate DataFrames for each driver and name hierarchical index
            CWPdriver_v = pd.concat(CWP_v, axis=1, names=["driver"])
            CWPdriver_j = pd.concat(CWP_j, axis=1, names=["driver", "j"])
            CWPdriver = pd.concat(CWP, axis=1, names=["driver"])
            IVdriver_v = pd.concat(IV_v, axis=1, names=["driver"])
            IVdriver = pd.concat(IV, axis=1, names=["driver"])

            return CWPdriver_v, CWPdriver_j, CWPdriver, IVdriver_v, IVdriver

        except:
            # Report severe error messages
            tb = sys.exc_info()[2]  # get traceback object for Python errors
            tbinfo = traceback.format_tb(tb)[0]
            msg = "Could not decompose development using {0} as a driver:\nTraceback info:\n{1}Error Info:\n{2}".format(
                driver, tbinfo, str(sys.exc_info()[1])
            )
            print(msg)  # print error message in Python
            # arcpy.AddError(msg)  # return error message in ArcGIS
            sys.exit(1)

    def valuation(self, dfBT, real=True, investment=False, factor=None):
        """Valuation as either Cost of Water Pollution (CWP) or Investment Value (IV).
        If not set to return real values (2018 prices), instead returns values in the prices of both the current year and the preceding year (for chain linking).
        """
        try:
            # Copy DataFrame with the variables needed for the Benefit Transfer function
            df = dfBT.copy()

            # Define a small constant to avoid RuntimeWarning due to taking the log of 0
            epsilon = 1e-6  #  a millionth part

            if investment is False:
                # MWTP = 0 if all water bodies of type j have ≥ good ecological status
                df["nonzero"] = np.select([df["Q"] < 3 - epsilon], [1])  #  dummy

                # Distance from current to Good: transform mean Q to lnΔQ ≡ ln(good - Q)
                df["Q"] = df["Q"].mask(
                    df["Q"] < 3 - epsilon,  # if some water bodies have < good status
                    np.log(3 - df["Q"] + epsilon),  #  log-transform difference good - Q
                )

                # lnΔQ = 0 if all water bodies of type j have ≥ good ecological status
                df["Q"] = df["Q"] * df["nonzero"]

            else:
                # Actual change in ecological status since preceding year
                df = df.reorder_levels(["j", "v", "t"]).sort_index()  #  series by j & v
                df["Q"] = df["Q"].diff()  #  transform Q to be the change in Q since t-1
                df = df.reorder_levels(["j", "t", "v"]).sort_index()  #  series by j & t

                # Dummy used to set MWTP = 0 if actual change in water quality is zero
                df["nonzero"] = np.select([df["Q"] != 0], [1])  #  dummy

                # Mark if actual change is negative (used to switch MWTP to negative)
                df["neg"] = np.select([df["Q"] < 0], [1])  #  dummy

                # Transform Q to the log of the actual change in water quality since t-1
                df["Q"] = df["Q"].mask(
                    df["Q"] != 0,  #  if actual change in water quality is nonzero
                    np.log(np.abs(df["Q"]) + epsilon),  #  log-transform absolute value
                )

            # Drop year 1989 and specify integer values
            df = df.drop(df[df.index.get_level_values("t") == 1989].index)
            df[["D age", "D lakes", "N"]] = df[["D age", "D lakes", "N"]].astype(int)

            # Consumer Price Index by year t (1990-2020)
            CPI_NPV = pd.read_excel("data\\" + self.data["shared"][0], index_col=0)

            # Merge data with CPI to correct for assumption of unitary income elasticity
            kwargs = dict(how="left", left_index=True, right_index=True)
            df1 = df.merge(CPI_NPV, **kwargs)
            df1["unityMWTP"] = self.BT(df1)  #  MWTP assuming unitary income elasticity

            if factor is None:
                # Calculate factor that MWTP is increased by if using estimated income ε
                df2018 = df1[df1.index.get_level_values("t") == 2018].copy()
                df2018["elastMWTP"] = self.BT(df2018, elast=1.453)  #  meta reg income ε
                df2018["factor"] = df2018["elastMWTP"] / df2018["unityMWTP"]
                df2018 = df2018.droplevel("t")
                factor = df2018.loc[("coastal"), :][["factor"]]
            df2 = df1.merge(factor, "left", left_index=True, right_index=True)
            df2 = df2.reorder_levels(["j", "t", "v"]).sort_index()

            # Adjust with factor of actual ε over unitary ε; set MWTP to 0 for certain Q
            df2["MWTP"] = df2["unityMWTP"] * df2["factor"] * df2["nonzero"]

            # Aggregate real MWTP per hh over households in coastal catchment area
            df2["CWP"] = df2["MWTP"] * df2["N"] / 1e06  #  million DKK (2018 prices)

            # Real costs of water pollution (million DKK, 2023 prices) by j, t, and v
            df2["CWP"] = (
                df2["CWP"] * CPI_NPV.loc[2023, "CPI"] / CPI_NPV.loc[2018, "CPI"]
            )

            if investment is True:
                # Switch MWTP to negative if actual change is negative
                cond = [df2["neg"] == 1]
                df2["CWP"] = np.select(cond, [-df2["CWP"]], default=df2["CWP"])

                # Apply net present value (NPV) factor using different discount rates r
                if real is False:
                    # r as prescribed by Ministry of Finance of Denmark the given year
                    df2["CWP"] = df2["CWP"] * df2["NPV"]

                else:
                    # Declining r as prescribed by Ministry of Finance during 2014-2020
                    df2["CWP"] = df2["CWP"] * CPI_NPV.loc[2020, "NPV"]

                    # Rename CWP to IV (investment value of water quality improvements)
                    df2 = df2.rename(columns={"CWP": "IV"})  # million DKK (2023 prices)

                    # Return real investment value (IV) by t, v, and j
                    return df2["IV"].unstack(level=0)

            if real is True:
                #  Return real cost of water pollution (CWP) by t, v, and j
                return df2["CWP"].unstack(level=0), factor

            # Aggregate nominal MWTP per hh over households in coastal catchment area
            df2["CWPn"] = df2["CWP"] * df2["CPI"] / CPI_NPV.loc[2023, "CPI"]

            # CWP in prices of the preceding year (for year-by-year chain linking)
            df2["D"] = df2["CWPn"] * df2["CPI t-1"] / df2["CPI"]

            if investment is True:
                # IV in prices of the preceding year (for year-by-year chain linking)
                df2["D"] = df2["D"] * df2["NPV t-1"] / df2["NPV"]

            # Aggregate over coastal catchment areas
            grouped = (
                df2[["CWPn", "D"]]
                .groupby(["j", "t"])
                .sum()
                .unstack(level=0)
                .rename_axis(None)
                .rename_axis([None, None], axis=1)
            )

            if investment is True:
                # Rename IV in prices of current year, and preceding year respectively
                grouped.columns = grouped.columns.set_levels(
                    [
                        "Investment value (current year's prices, million DKK)",
                        "Investment value (preceding year's prices, million DKK)",
                    ],
                    level=0,
                )

            else:
                # Rename CWP in prices of current year, and preceding year respectively
                grouped.columns = grouped.columns.set_levels(
                    [
                        "Cost (current year's prices, million DKK)",
                        "Cost (preceding year's prices, million DKK)",
                    ],
                    level=0,
                )

            return grouped  #  in prices of current year and preceding year respectively

        except:
            # Report severe error messages
            tb = sys.exc_info()[2]  # get traceback object for Python errors
            tbinfo = traceback.format_tb(tb)[0]
            msg = "Could not apply valuation to df {0}:\nTraceback info:\n{1}Error Info:\n{2}".format(
                df, tbinfo, str(sys.exc_info()[1])
            )
            print(msg)  # print error message in Python
            # arcpy.AddError(msg)  # return error message in ArcGIS
            sys.exit(1)

    def BT(self, df, elast=1):
        """Apply Benefit Transfer function from meta study (Zandersen et al., 2022)"""
        try:
            # ln MWTP for improvement from current ecological status to "Good"
            lnMWTP = (
                4.142
                + 0.551 * df["Q"]
                + elast * df["ln y"]
                + 0.496 * df["D age"]
                + 0.121 * df["ln PSL"]
                - 0.072 * df["ln PAL"]
                - 0.005 * df["SL"]
                - 0.378 * df["D lakes"]
            )

            # Real MWTP per household (DKK, 2018 prices) using the meta study variance
            MWTP = np.exp(lnMWTP + (0.136 + 0.098) / 2)  #  variance components

            return MWTP

        except:
            # Report severe error messages
            tb = sys.exc_info()[2]  # get traceback object for Python errors
            tbinfo = traceback.format_tb(tb)[0]
            msg = "Could not apply Benefit Transfer function to df {0} with elasticity {1}:\nTraceback info:\n{1}Error Info:\n{2}".format(
                df, tbinfo, str(sys.exc_info()[1])
            )
            print(msg)  # print error message in Python
            # arcpy.AddError(msg)  # return error message in ArcGIS
            sys.exit(1)
            sys.exit(1)
