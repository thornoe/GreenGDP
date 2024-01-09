"""
Name:       script_module.py

Label:      Construct and map longitudinal data of ecological status of streams.

Summary:    ThorNoe.GitHub.io/GreenGDP explains the overall approach and methodology.

Rqmts:      ArcGIS Pro must be installed on the system and be up to date.

Usage:      This module supports script.py and WaterbodiesScriptTool in gis.tbx.
            See GitHub.com/ThorNoe/GreenGDP for instructions to run or update it all.

Functions:  The class in this module contains 10 functions of which some are nested:
            - get_data(), get_fc_from_WFS(), and map_book() are standalone functions.
            - observed_indicator() calls:
                - stations_to_streams(), which calls:
                    - longitudinal(), which again calls:
                        - frame()
            - ecological_status() calls:
              - indicator_to_status()
              - missing_values_graph()
            Descriptions can be seen under each function.

License:    MIT Copyright (c) 2020-2023
Author:     Thor Donsby Noe
"""
import os
import sys
import traceback
import urllib.request

import arcpy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

arcpy.env.overwriteOutput = True  # set overwrite option


class Water_Quality:
    """Class for all data processing and mapping functions"""

    def __init__(
        self,
        yearFirst,
        yearLast,
        dataFilenames,
        linkageFilenames,
        WFS_ServiceURL,
        WFS_featureClassNames,
        WFS_replaceFeatureClasses,
        keepGeodatabase,
    ):
        self.years = list(range(yearFirst, yearLast + 1))
        self.data = dataFilenames
        self.linkage = linkageFilenames
        self.wfs_service = WFS_ServiceURL
        self.wfs_fc = WFS_featureClassNames
        self.wfs_replace = WFS_replaceFeatureClasses
        self.keep_gdb = keepGeodatabase
        self.path = os.getcwd()
        self.arcPath = self.path + "\\gis.gdb"

        # Set the ArcPy workspace
        arcpy.env.workspace = self.arcPath

        # Check that folders for data, output, and linkage files exist or create them
        self.get_data()

        # Get feature class for coastal catchment areas from the WFS service
        self.get_fc_from_WFS("catch")

    def get_data(self):
        """Function to check that the folders and their files exist.
        Otherwise creates the folders and downloads the files from GitHub."""
        try:
            # Dictionary for folders and their data and linkage files
            allFiles = {
                "data": [a for b in list(self.data.values()) for a in b],
                "linkage": [a for a in list(self.linkage.values())],
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
            arcpy.AddError(msg)  # return error message in ArcGIS
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
                            arcpy.AddError(urlmsg)  # return URL error message in ArcGIS
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
                            arcpy.AddError(msg)  # return error message in ArcGIS
                            sys.exit(1)

            except OSError:
                # Report system errors
                tb = sys.exc_info()[2]  # get traceback object for Python errors
                tbinfo = traceback.format_tb(tb)[0]
                OSmsg = "System error for {0} folder:\nTraceback info:\n{1}Error Info:\n{2}".format(
                    key, tbinfo, str(sys.exc_info()[1])
                )
                print(OSmsg)  # print system error message in Python
                arcpy.AddError(OSmsg)  # return system error message in ArcGIS
                sys.exit(1)

            finally:
                # Change the directory back to the original working folder
                os.chdir(self.path)

    def get_fc_from_WFS(self, fc):
        """Create a feature class from a WFS service given the type of water body.
        Also create a template with only the most necessary fields."""
        try:
            # Set names of the feature class for the given type of water body
            WFS_FeatureType = self.wfs_fc[fc]

            # Set the names of the fields (columns) in fc that contain the ID (and typology)
            if fc == "catch":
                fields = ["op_id"]
            else:
                fields = ["ov_id", "ov_navn", "ov_typ"]

            if self.wfs_replace != 0:
                # Delete the fc template to create it anew
                if arcpy.Exists(fc):
                    arcpy.Delete_management(fc)

            if not arcpy.Exists(fc):
                # Execute the WFSToFeatureClass tool to download the fc
                arcpy.conversion.WFSToFeatureClass(
                    self.wfs_service,
                    WFS_FeatureType,
                    self.arcPath,
                    fc,
                    max_features=10000,
                )

                # Create a list of unnecessary fields
                fieldsUnnecessary = []
                fieldObjList = arcpy.ListFields(fc)
                for field in fieldObjList:
                    if not field.required:
                        if field.name not in fields:
                            fieldsUnnecessary.append(field.name)

                # Remove unnecessary fields (columns) to reduce the size of the feature class
                arcpy.DeleteField_management(fc, fieldsUnnecessary)

        except:
            # Report severe error messages from Python or ArcPy
            tb = sys.exc_info()[2]  # get traceback object for Python errors
            tbinfo = traceback.format_tb(tb)[0]
            pymsg = "Python errors while using WFS for {0}:\nTraceback info:\n{1}Error Info:\n{2}".format(
                fc, tbinfo, str(sys.exc_info()[1])
            )
            arcmsg = "ArcPy errors while using WFS for {0}:\n{1}".format(
                fc, arcpy.GetMessages(severity=2)
            )
            print(pymsg)  # print Python error message in Python
            print(arcmsg)  # print ArcPy error message in Python
            arcpy.AddError(pymsg)  # return Python error message in ArcGIS
            arcpy.AddError(arcmsg)  # return ArcPy error message in ArcGIS
            sys.exit(1)

    def observed_indicator(self, waterbodyType):
        """Based on the type of water body, set up a longitudinal DataFrame with the
        observed indicators for all water bodies."""
        try:
            if waterbodyType == "streams":
                # Create longitudinal df and use linkage table to assign stations to water bodies
                df, dfVP = self.stations_to_streams(
                    waterbodyType,
                    fileName=self.data[waterbodyType][0],
                )

            # Save observations to CSV for later statistical work
            df.to_csv("output\\" + waterbodyType + "_ind_obs.csv")

            # Save characteristics of water bodies to CSV for later work
            dfVP.to_csv("output\\" + waterbodyType + "_VP.csv")

            return df, dfVP

        except:
            # Report severe error messages
            tb = sys.exc_info()[2]  # get traceback object for Python errors
            tbinfo = traceback.format_tb(tb)[0]
            msg = "Could not create df with observations for {0}:\nTraceback info:\n{1}Error Info:\n{2}".format(
                waterbodyType, tbinfo, str(sys.exc_info()[1])
            )
            print(msg)  # print error message in Python
            arcpy.AddError(msg)  # return error message in ArcGIS
            sys.exit(1)

    def stations_to_streams(self, waterbodyType, fileName, radius=15):
        """Streams: Assign monitoring stations to water bodies via linkage table.

        For unmatched stations: Assign to stream within a radius of 15 meters where the location names match the name of the stream.

        For a given year, finds DVFI for a stream with multiple stations
        by taking the median and rounding down.

        Finally, extends to all streams in the current water body plan (VP3) and adds the ID, catchment area, and length of each stream in km using the feature classes collected via get_fc_from_WFS()
        """
        try:
            # Create longitudinal DataFrame for stations in streams by monitoring approach
            DVFI = self.longitudinal(waterbodyType, fileName, "DVFI")
            DVFI_MIB = self.longitudinal(waterbodyType, fileName, "DVFI, MIB")
            DVFI_F = self.longitudinal(waterbodyType, fileName, "Faunaklasse, felt")

            # Group by station and keep first non-null entry each year DVFI>MIB>felt
            long = (
                pd.concat([DVFI, DVFI_MIB, DVFI_F])
                .groupby(["station"], as_index=False)
                .first()
            )

            # Read the linkage table
            dfLinkage = pd.read_csv("linkage\\" + self.linkage[waterbodyType])

            # Convert station ID to integers
            dfLinkage["station"] = dfLinkage["station_id"].str.slice(7).astype(int)

            # Merge longitudinal DataFrame with linkage table for water bodies in VP3
            df = long.merge(dfLinkage[["station", "ov_id"]], how="left", on="station")

            # Stations covered by the linkage tabel for the third water body plan VP3
            link = df.dropna(subset=["ov_id"])

            # Convert waterbody ID (wb) to integers
            link["wb"] = link["ov_id"].str.slice(7).astype(int)

            # Stations not covered by the linkage table for VP3
            noLink = df[df["ov_id"].isna()].drop(columns=["ov_id"])

            # Create a spatial reference object with the same geographical coordinate system
            spatialRef = arcpy.SpatialReference("ETRS 1989 UTM Zone 32N")

            # Specify name of feature class for stations (points)
            fcStations = waterbodyType + "_stations"

            # Create new feature class shapefile (will overwrite if it already exists)
            arcpy.CreateFeatureclass_management(
                self.arcPath, fcStations, "POINT", spatial_reference=spatialRef
            )

            # Create fields for 'station' and 'location'
            arcpy.AddField_management(fcStations, "station", "INTEGER")
            arcpy.AddField_management(fcStations, "location", "TEXT")

            # Create cursor to insert stations that were not in the linkage table
            try:
                with arcpy.da.InsertCursor(
                    fcStations, ["SHAPE@XY", "station", "location"]
                ) as cursor:
                    # Loop over each station-ID in df:
                    for index, row in noLink.iterrows():
                        try:
                            # Use cursor to insert new row in feature class
                            cursor.insertRow(
                                [(row["x"], row["y"]), row["station"], row["location"]]
                            )

                        except:
                            # Report other severe error messages from Python or ArcPy
                            tb = sys.exc_info()[
                                2
                            ]  # get traceback object for Python errors
                            tbinfo = traceback.format_tb(tb)[0]
                            print(
                                "Python errors while inserting station {0} in {1}:\nTraceback info:{2}\nError Info:\n{3}\n".format(
                                    str(row["station"]),
                                    fcStations,
                                    tbinfo,
                                    str(sys.exc_info()[1]),
                                )
                            )
                            print(
                                "ArcPy errors while inserting station {0} in {1}:\n{2}".format(
                                    str(row["station"]),
                                    fcStations,
                                    tbinfo,
                                )
                            )
                            sys.exit(1)

                        finally:
                            # Clean up for next iteration
                            del index, row

            finally:
                del cursor

            # Specify name of feature class for streams in VP3 (lines)
            fc = waterbodyType

            # Specify name of joined feature class (lines)
            fcJoined = fcStations + "_joined"

            # Spatial Join unmatched stations with streams within given radius
            arcpy.SpatialJoin_analysis(
                fc,
                fcStations,
                fcJoined,  #  will overwrite if it already exists
                "JOIN_ONE_TO_MANY",
                "KEEP_COMMON",
                match_option="CLOSEST",
                search_radius=radius,
                distance_field_name="Distance",
            )

            # Specify fields of interest of fcJoined
            fieldsJ = ["station", "ov_id", "ov_navn", "location", "Distance"]

            # Create DataFrame from fcJoined and sort by distance (ascending)
            stations = [row for row in arcpy.da.SearchCursor(fcJoined, fieldsJ)]
            j = pd.DataFrame(stations, columns=fieldsJ).sort_values("Distance")

            # Convert waterbody ID (wb) to integers
            j["wb"] = j["ov_id"].str.slice(7).astype(int)

            # Capitalize water body names
            j["ov_navn"] = j["ov_navn"].str.upper()

            # Rename unnamed water bodies to distinguish from named water bodies
            j["location"].mask(
                j["location"] == "[IKKE NAVNGIVET]", "UDEN NAVN", inplace=True
            )

            # Indicate that station and stream has the same water body name
            j["match"] = np.select([j["location"] == j["ov_navn"]], [True])

            # Subset to unique stations with their closest matching water body
            jClosest = (
                j[j["match"] == True].groupby(["station"], as_index=False).first()
            )

            # Inner merge of noLink stations and jClosest water body with matching name
            noLinkClosest = noLink.merge(
                jClosest[["station", "wb"]], how="inner", on="station"
            )

            # df containing all stations that have been matched to a water body
            allMatches = pd.concat([link, noLinkClosest]).drop(
                columns=["station", "location", "x", "y", "ov_id"]
            )

            # Group multiple stations in a water body: Take the median and round down
            waterbodies = allMatches.groupby("wb").median().apply(np.floor)

            # Fields in fc that contain waterbody ID, typology, and shape length
            fields = ["ov_id", "ov_typ", "Shape_Length"]

            # Create df from fc with characteristics of all streams in VP
            dataVP = [row for row in arcpy.da.SearchCursor(fc, fields)]
            dfVP = pd.DataFrame(dataVP, columns=fields)

            # Shore length is counted on both sides of the stream
            dfVP[["length"]] = dfVP[["Shape_Length"]] * 2

            # Convert waterbody ID (wb) to integers and sort (ascending)
            dfVP["wb"] = dfVP["ov_id"].str.slice(7).astype(int)
            dfVP = dfVP.sort_values("wb")

            # Merge df for all streams in VP3 with df for streams with observed status
            allVP = dfVP[["wb"]].merge(waterbodies, how="left", on="wb")

            # Set water body ID as index
            allVP = allVP.set_index("wb")
            dfVP = dfVP[["wb", "ov_typ", "length"]].set_index("wb")

            # Report stations matched by linkage table and distance+name respectively
            msg = "{0}:\n{1} out of {2} stations were linked to a water body by the official linkage table. Besides, {3} were located within {4} meters of a water body carrying the name of the station's location.".format(
                str(waterbodyType),
                len(link),
                len(df),
                len(jClosest),
                str(radius),
            )
            # print(msg)  # print number of stations in Python
            arcpy.AddMessage(msg)  # return number of stations in ArcGIS

            return allVP, dfVP

        except:
            # Report severe error messages
            tb = sys.exc_info()[2]  # get traceback object for Python errors
            tbinfo = traceback.format_tb(tb)[0]
            pymsg = "Python errors while assigning stations to streams:\nTraceback info:\n{0}Error Info:\n{1}".format(
                tbinfo, str(sys.exc_info()[1])
            )
            arcmsg = "ArcPy errors while assigning stations to streams:\n{0}".format(
                arcpy.GetMessages(severity=2)
            )
            print(pymsg)  # print Python error message in Python
            print(arcmsg)  # print ArcPy error message in Python
            arcpy.AddError(pymsg)  # return Python error message in ArcGIS
            arcpy.AddError(arcmsg)  # return ArcPy error message in ArcGIS
            sys.exit(1)

        finally:
            # Clean up
            if self.keep_gdb != "true":
                # Delete feature classes
                for fc in [fcStations, fcJoined]:
                    if arcpy.Exists(fc):
                        arcpy.Delete_management(fc)

    def longitudinal(self, waterbodyType, fileName, parameterType):
        """Set up a longitudinal DataFrame based on the type of water body.

        Streams: For a given year, finds DVFI for a station with multiple
                 observations by taking the median and rounding down."""
        try:
            # Set up a Pandas DataFrame for the chosen type of water body
            if waterbodyType == "streams":
                df = self.frame(
                    waterbodyType,
                    fileName,
                    parameterType,
                    parameterCol="Indekstype",
                    valueCol="Indeks",
                )

                # Drop obs with unknown index value and save index as integers
                df = df[df.ind != "U"]
                df["ind"] = df["ind"].astype(int)

            # Set up a longitudinal df with every station and its latest records
            long = (
                df[["station", "location", "x", "y"]]
                .groupby(["station"], as_index=False)
                .last()
            )

            # Add a column for each year
            for i in self.years:
                # Subset to relevant year and drop location, year, and coordinates
                dfYear = df[df["year"] == i].drop(
                    ["location", "year", "x", "y"], axis=1
                )

                if waterbodyType == "streams":
                    # Group multiple obs for a station: Take the median and round down
                    dfYear = (
                        dfYear.groupby(["station"]).median().apply(np.floor).astype(int)
                    )

                # Merge into longitudinal df
                dfYear.columns = [i]
                long = long.merge(dfYear, how="left", on="station")

            return long

        except:
            ## Report severe error messages
            tb = sys.exc_info()[2]  # get traceback object for Python errors
            tbinfo = traceback.format_tb(tb)[0]
            msg = "Could not set up DataFrame for {0}:\nTraceback info:\n{1}Error Info:\n{2}".format(
                waterbodyType, tbinfo, str(sys.exc_info()[1])
            )
            print(msg)  # print error message in Python
            arcpy.AddError(msg)  # return error message in ArcGIS
            sys.exit(1)

    def frame(self, waterbodyType, fileName, parameterType, parameterCol, valueCol):
        """Function to set up a Pandas DataFrame for a given type of water body"""
        try:
            # Read the data
            df = pd.read_excel("data\\" + fileName)  # 1987-2020

            # Create 'Year' column from the date integer
            df["year"] = df["Dato"].astype(str).str.slice(0, 4).astype(int)

            # Subset the data to only contain the relevant parameter
            df = df[df[parameterCol] == parameterType]

            # Subset the data to relevant variables and sort by year
            df = df[
                [
                    "ObservationsStedNr",
                    "Lokalitetsnavn",
                    "year",
                    valueCol,
                    "Xutm_Euref89_Zone32",
                    "Yutm_Euref89_Zone32",
                ]
            ].sort_values("year")

            # Shorten column names
            df.columns = ["station", "location", "year", "ind", "x", "y"]

            # Capitalize location names
            df["location"] = df["location"].str.upper()

            return df

        except:
            ## Report severe error messages
            tb = sys.exc_info()[2]  # get traceback object for Python errors
            tbinfo = traceback.format_tb(tb)[0]
            msg = "Could not set up DataFrame for {0}:\nTraceback info:\n{1}Error Info:\n{2}".format(
                waterbodyType, tbinfo, str(sys.exc_info()[1])
            )
            print(msg)  # print error message in Python
            arcpy.AddError(msg)  # return error message in ArcGIS
            sys.exit(1)

    def ecological_status(self, waterbodyType, dfIndicator, dfVP, suffix="obs"):
        """Based on the type of water body, convert the longitudinal DataFrame to the EU index of ecological status, i.e. from 0-4 for Bad, Poor, Moderate, Good, and High water quality respectively.

        Create a table of statistics and export it as an html table.

        Print the length and share of water bodies observed at least once."""
        try:
            # Convert index of indicators to index of ecological status
            dfStatus = self.indicator_to_status(waterbodyType, dfIndicator)

            # Merge df for observed ecological status with df for characteristics
            df = dfVP[["length"]].merge(dfStatus, how="inner", on="wb")

            # Calculate total length of all water bodies in current water body plan (VP2)
            totalLength = df["length"].sum()

            # Create an empty df for statistics
            stats = pd.DataFrame(
                index=self.years,
                columns=[
                    "Status known (%)",
                    "Share of known is High (%)",
                    "Share of known is Good (%)",
                    "Share of known is Moderate (%)",
                    "Share of known is Poor (%)",
                    "Share of known is Bad (%)",
                ],
            )

            # Calculate the above statistics for each year
            for i in self.years:
                y = df[[i, "length"]].reset_index(drop=True)
                y["known"] = np.select([y[i].notna()], [y["length"]])
                y["high"] = np.select([y[i] == 4], [y["length"]])
                y["good"] = np.select([y[i] == 3], [y["length"]])
                y["moderate"] = np.select([y[i] == 2], [y["length"]])
                y["poor"] = np.select([y[i] == 1], [y["length"]])
                y["bad"] = np.select([y[i] == 0], [y["length"]])

                # Add shares of total length to stats
                knownLength = y["known"].sum()
                stats.loc[i] = [
                    100 * knownLength / totalLength,
                    100 * y["high"].sum() / knownLength,
                    100 * y["good"].sum() / knownLength,
                    100 * y["moderate"].sum() / knownLength,
                    100 * y["poor"].sum() / knownLength,
                    100 * y["bad"].sum() / knownLength,
                ]

            # Save statistics to html for online presentation
            stats.astype(int).to_html("output\\" + waterbodyType + "_eco_obs_stats.md")

            # Shorten column names of statistics
            stats.columns = ["known", "high", "good", "moderate", "poor", "bad"]

            # Save statistics and water bodies to CSV
            dfStatus.to_csv("output\\" + waterbodyType + "_eco_" + suffix + ".csv")
            stats.to_csv("output\\" + waterbodyType + "_eco_" + suffix + "_stats.csv")

            # Create missing values graph (heatmap of missing observations by year):
            self.missing_values_graph(waterbodyType, df, suffix)

            # Create df limited to water bodies that are observed at least once
            observed = dfVP[["length"]].merge(
                dfStatus.dropna(how="all"),
                how="inner",
                on="wb",
            )

            # Report length and share of water bodies observed at least once.
            msg = "{0} km is the total shore length of {1} included in VP3, of which {2}% of {1} representing {3} km ({4}% of total shore length of {1}) have been assessed at least once. On average, {5}% of {1} representing {6} km ({7}% of total shore length of {1}) are assessed each year.\n".format(
                round(totalLength * 10 ** (-3)),
                waterbodyType,
                round(100 * len(observed) / len(df)),
                round(observed["length"].sum() * 10 ** (-3)),
                round(100 * observed["length"].sum() / totalLength),
                round(100 * np.mean(dfStatus.count() / len(df))),
                round(stats["known"].mean() / 100 * totalLength * 10 ** (-3)),
                round(stats["known"].mean()),
            )
            # print(msg)  # print statistics in Python
            arcpy.AddMessage(msg)  # return statistics in ArcGIS

            return dfStatus, stats

        except:
            # Report severe error messages
            tb = sys.exc_info()[2]  # get traceback object for Python errors
            tbinfo = traceback.format_tb(tb)[0]
            msg = "Could not create df with observed ecological status for {0}:\nTraceback info:\n{1}Error Info:\n{2}".format(
                waterbodyType, tbinfo, str(sys.exc_info()[1])
            )
            print(msg)  # print error message in Python
            arcpy.AddError(msg)  # return error message in ArcGIS
            sys.exit(1)

    def indicator_to_status(self, waterbodyType, dfIndicator):
        """Convert biophysical indicators to ecological status."""
        try:
            # Copy DataFrame for the biophysical indicator
            df = dfIndicator.copy()

            if waterbodyType == "streams":
                # Convert DVFI fauna index for streams to index of ecological status
                for i in self.years:
                    # Categorical variable for ecological status: Bad, Poor, Moderate, Good, High
                    conditions = [
                        df[i] < 1.5,
                        (df[i] >= 1.5) & (df[i] < 3.5),
                        (df[i] >= 3.5) & (df[i] < 4.5),
                        (df[i] >= 4.5) & (df[i] < 6.5),
                        df[i] >= 6.5,
                    ]
                    df[i] = np.select(conditions, [0, 1, 2, 3, 4], default=np.nan)

            return df

        except:
            ## Report severe error messages
            tb = sys.exc_info()[2]  # get traceback object for Python errors
            tbinfo = traceback.format_tb(tb)[0]
            msg = "Could not convert biophysical indicator to ecological status for {0}:\nTraceback info:\n{1}Error Info:\n{2}".format(
                waterbodyType, tbinfo, str(sys.exc_info()[1])
            )
            print(msg)  # print error message in Python
            arcpy.AddError(msg)  # return error message in ArcGIS
            sys.exit(1)

    def missing_values_graph(self, waterbodyType, frame, suffix):
        """Heatmap visualizing observations of ecological status as either missing or using the EU index of ecological status, i.e., from 0-4 for Bad, Poor, Moderate, Good, and High water quality respectively.
        Saves a figure of the heatmap."""
        try:
            # Sort by number of missing values
            df = frame.copy()
            df["nan"] = df.shape[1] - df.count(axis=1)
            df = df.sort_values(["nan"], ascending=False)[self.years]

            # Plot heatmap
            df.fillna(0, inplace=True)
            cm = sns.xkcd_palette(["grey", "red", "orange", "yellow", "green", "blue"])
            plt.figure(figsize=(12, 7.4))
            ax = sns.heatmap(
                df,
                cmap=cm,
                cbar=False,
                cbar_kws={"ticks": [-1, 0, 1, 2, 3, 4, 5]},
            )
            ax.set(yticklabels=[])
            plt.ylabel(waterbodyType + " (N=" + str(len(df)) + ")", fontsize=14)
            plt.xlabel("")
            plt.title(
                (
                    "Ecological status of "
                    + waterbodyType
                    + ":"
                    + "\nMissing value (grey), Bad (red), Poor (orange), Moderate (yellow), Good (green), High (blue)"
                ),
                fontsize=14,
            )
            plt.tight_layout()
            plt.savefig(
                "output\\" + waterbodyType + "_eco_" + suffix + ".png",
                bbox_inches="tight",
            )

        except:
            # Report severe error messages
            tb = sys.exc_info()[2]  # get traceback object for Python errors
            tbinfo = traceback.format_tb(tb)[0]
            msg = "Could not create missing values graph (heatmap) for ecological status of {1}:\nTraceback info:\n{2}Error Info:\n{2}".format(
                waterbodyType, tbinfo, str(sys.exc_info()[1])
            )
            print(msg)  # print error message in Python
            arcpy.AddError(msg)  # return error message in ArcGIS
            sys.exit(1)

    def map_book(self, fc, df):
        """Create a pdf map book with a page for each year using the fc created with get_fc_from_WFS()"""
        try:
            # Add an integer field (column) for storing the ecological status
            arcpy.AddField_management(fc, "status", "INTEGER")

            # Create a map book to contain a pdf page for each year
            bookPath = self.path + "\\output\\" + fc + ".pdf"
            if os.path.exists(bookPath):
                os.remove(bookPath)

            # Initiate the map book
            book = arcpy.mp.PDFDocumentCreate(bookPath)

            ##### Make a feature class, layer and pdf map for each year
            for i in self.years[::-1]:
                try:
                    # Copy feature class from template
                    fcYear = fc + str(i) + "fc"
                    if arcpy.Exists(fcYear):
                        arcpy.Delete_management(fcYear)
                    arcpy.CopyFeatures_management(fc, fcYear)

                    # Create update cursor for feature layer
                    with arcpy.da.UpdateCursor(fcYear, ["ov_id", "status"]) as cursor:
                        try:
                            # For each row/object in feature class add ecological status from df
                            for row in cursor:
                                if row[0] in df.index:
                                    row[1] = df.loc[row[0], i]
                                    cursor.updateRow(row)

                        except:
                            # Report severe error messages from Python or ArcPy
                            tb = sys.exc_info()[
                                2
                            ]  # get traceback object for Python errors
                            tbinfo = traceback.format_tb(tb)[0]
                            pymsg = "Python errors while updating {0} in {1}:\nTraceback info:\n{2}Error Info:\n{3}".format(
                                row[0], fc, tbinfo, str(sys.exc_info()[1])
                            )
                            arcmsg = (
                                "ArcPy errors while updating {0} in {1}:\n{2}".format(
                                    row[0], fc, arcpy.GetMessages(severity=2)
                                )
                            )
                            print(pymsg)  # print Python error message in Python
                            print(arcmsg)  # print ArcPy error message in Python
                            arcpy.AddError(
                                pymsg
                            )  # return Python error message in ArcGIS
                            arcpy.AddError(
                                arcmsg
                            )  # return ArcPy error message in ArcGIS
                            sys.exit(1)

                        finally:
                            # Clean up for next iteration
                            del cursor, row

                    # Create a feature layer for the feature class (name of layer applies to the legend)
                    layerYear = fc + str(i)
                    arcpy.MakeFeatureLayer_management(fcYear, layerYear)

                    # Apply symbology for ecological status from layer
                    arcpy.ApplySymbologyFromLayer_management(
                        layerYear, self.path + "\\" + fc + "_symbology.lyrx"
                    )

                    # Save layer file
                    if arcpy.Exists(layerYear + ".lyrx"):
                        arcpy.Delete_management(layerYear + ".lyrx")
                    arcpy.SaveToLayerFile_management(layerYear, layerYear + ".lyrx")

                    # Reference the layer file
                    lyrFile = arcpy.mp.LayerFile(layerYear + ".lyrx")

                    # Reference the ArcGIS Pro project, its map, and the layout to export
                    aprx = arcpy.mp.ArcGISProject(self.path + "\\gis.aprx")
                    m = aprx.listMaps("Map")[0]
                    lyt = aprx.listLayouts("Layout")[0]

                    # Add layer to map
                    m.addLayer(lyrFile, "TOP")

                    # Export layout to a temporary PDF
                    lyt.exportToPDF("temp.pdf", resolution=300)

                    # Append the page to the map book
                    book.appendPages(self.path + "\\temp.pdf")

                except:
                    # Report severe error messages from Python or ArcPy
                    tb = sys.exc_info()[2]  # get traceback object for Python errors
                    tbinfo = traceback.format_tb(tb)[0]
                    pymsg = "Python errors while creating a PDF for {0}, {1}:\nTraceback info:\n{2}Error Info:\n{3}".format(
                        fc, i, tbinfo, str(sys.exc_info()[1])
                    )
                    arcmsg = (
                        "ArcPy errors while creating a PDF for {0}, {1}:\n{2}".format(
                            fc, i, arcpy.GetMessages(severity=2)
                        )
                    )
                    print(pymsg)  # print Python error message in Python
                    print(arcmsg)  # print ArcPy error message in Python
                    arcpy.AddError(pymsg)  # return Python error message in ArcGIS
                    arcpy.AddError(arcmsg)  # return ArcPy error message in ArcGIS
                    sys.exit(1)

                finally:
                    # Clean up after each iteration of loop
                    if self.keep_gdb != "true":
                        # Delete feature class and layer file
                        for fc in [fcYear, layerYear + ".lyrx"]:
                            if arcpy.Exists(fc):
                                arcpy.Delete_management(fc)
                    del fcYear, layerYear, lyrFile, aprx, m, lyt

            # Commit changes and save the map book
            book.saveAndClose()

        except:
            # Report severe error messages from Python or ArcPy
            tb = sys.exc_info()[2]  # get traceback object for Python errors
            tbinfo = traceback.format_tb(tb)[0]
            pymsg = "Python errors while creating a map book for {0}:\nTraceback info:\n{1}Error Info:\n{2}".format(
                fc, tbinfo, str(sys.exc_info()[1])
            )
            arcmsg = "ArcPy errors while creating a map book for {0}:\n{1}".format(
                fc, arcpy.GetMessages(severity=2)
            )
            print(pymsg)  # print Python error message in Python
            print(arcmsg)  # print ArcPy error message in Python
            arcpy.AddError(pymsg)  # return Python error message in ArcGIS
            arcpy.AddError(arcmsg)  # return ArcPy error message in ArcGIS
            sys.exit(1)

        finally:
            # Clean up
            if os.path.exists("temp.pdf"):
                os.remove("temp.pdf")
            del book
            if self.keep_gdb != "true":
                # Delete the entire geodatabase (all FCs must be deleted first)
                if arcpy.Exists(self.arcPath):
                    arcpy.Delete_management(self.arcPath)
