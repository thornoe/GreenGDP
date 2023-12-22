"""
Name:       script_module.py

Label:      Construct and map longitudinal data of ecological status of streams.

Summary:    ThorNoe.GitHub.io/GreenGDP explains the overall approach and methodology.

Rqmts:      ArcGIS Pro must be installed on the system and be up to date.

Usage:      This module supports script.py and WaterbodiesScriptTool in gis.tbx.
            See GitHub.com/ThorNoe/GreenGDP for instructions to run or update it all.

Functions:  The class in this module contains 8 functions of which some are nested:
            - get_data() and get_fc_from_WFS() are both standalone functions.
            - observed_indicator() calls:
                - stations_to_streams(), which calls:
                    - longitudinal(), which again calls:
                        - frame()
            - ecological_status() calls:
              - DVFI_to_status()
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
        keepGeodatabase,
    ):
        self.years = list(range(yearFirst, yearLast + 1))
        self.data = dataFilenames
        self.linkage = linkageFilenames
        self.wfs_service = WFS_ServiceURL
        self.wfs_fc = WFS_featureClassNames
        self.keep_gdb = keepGeodatabase
        self.path = os.getcwd()
        self.arcPath = self.path + "\\gis.gdb"

        # Set the ArcPy workspace
        arcpy.env.workspace = self.arcPath

        # Create an empty geodatabase

        # Check that the folders with data and linkage files exist or create them
        self.get_data()

        # Get feature class for coastal catchment area from the WFS service
        self.get_fc_from_WFS("catch")

        # Create the output folder if it doesn't exist already
        os.makedirs(self.path + "\\output", exist_ok=True)

    def get_data(self):
        """Function to check that the folders and their files exist.
        Otherwise creates the folder and downloads the files from GitHub.
        """
        try:
            # Dictionary for all data and linkage files
            allFiles = {
                "data": [a for a in list(self.data.values())],
                "linkage": [a for a in list(self.linkage.values())],
            }
            allFiles["data"].append("demographics.csv")
            allFiles["data"].append("SR486_VandkvalitetsBenefitTransferRedskab.xlsx")

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
                    folderName, tbinfo, str(sys.exc_info()[1])
                )
                print(OSmsg)  # print system error message in Python
                arcpy.AddError(OSmsg)  # return system error message in ArcGIS
                sys.exit(1)

            finally:
                # Change the directory back to the original working folder
                os.chdir(self.path)

    def get_gdb_from_WFS(self):
        """Create a feature class from a WFS service given the type of water body.
        Also create a template with only the most necessary fields.
        """
        try:
            # Set names of the feature class for the given type of water body
            WFS_FeatureType = self.wfs_fc[fc]

            # Set the names of the fields (columns) in fc that contain the ID (and typology)
            if fc == "catch":
                fields = ["op_id"]
            else:
                fields = ["ov_id", "ov_typ"]

            if arcpy.Exists(fc):
                arcpy.Delete_management(fc)  #  if making changes to the fc template
            if not arcpy.Exists(fc):
                # Execute the WFSToFeatureClass tool to download the feature class.
                arcpy.conversion.WFSToFeatureClass(
                    self.wfs_service,
                    WFS_FeatureType,
                    self.arcPath,
                    fc,
                    max_features=15000,
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

    def frame(self, waterbodyType, parameterCol, parameterType, valueCol):
        """Function to set up a Pandas DataFrame for a given type of water body"""
        try:
            # Obtain the filenames from the initialization of this class
            filenames = self.data[waterbodyType]

            # Read the data
            df = pd.read_excel("data\\" + filenames)  # 1989-2020

            # Create 'Year' column from the date integer
            df["Year"] = df["Dato"].astype(str).str.slice(0, 4).astype(int)

            # Subset the data to only contain the relevant parameter
            df = df[df[parameterCol] == parameterType]

            # Subset the data to relevant variables and sort by year
            df = df[
                [
                    "ObservationsStedNr",
                    "Lokalitetsnavn",
                    "Year",
                    valueCol,
                    "Xutm_Euref89_Zone32",
                    "Yutm_Euref89_Zone32",
                ]
            ].sort_values("Year")

            # Shorten column names
            df.columns = ["Station", "Location", "Year", valueCol, "X", "Y"]

            # Capitalize location names
            df["Location"] = df["Location"].str.upper()

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

    def longitudinal(self, waterbodyType, parameterType):
        """Set up a longitudinal dataframe based on the type of water body

        Streams: For a given year, finds DVFI for a station with multiple
                 observations by taking the median and rounding down.
        """
        try:
            # Set up a Pandas DataFrame for the chosen type of water body
            if waterbodyType == "streams":
                df = self.frame(waterbodyType, "Indekstype", parameterType, "Indeks")

                # Drop obs with unknown index value and save index as integers
                df = df[df.Indeks != "U"]
                df["Indeks"] = df["Indeks"].astype(int)

            # Set up a longitudinal df with every station and its latest records
            long = (
                df[["Station", "Location", "X", "Y"]]
                .groupby(["Station"], as_index=False)
                .last()
            )

            # Add a column for each year
            for i in self.years:
                # Subset to relevant year
                dfYear = df[df["Year"] == i]

                # Drop year and coordinates
                dfYear = dfYear.drop(["Year", "X", "Y"], axis=1)

                if waterbodyType == "streams":
                    # Group multiple obs for a station: Take the median and round down
                    dfYear = (
                        dfYear.groupby(["Station"]).median().apply(np.floor).astype(int)
                    )

                # Merge into longitudinal df
                dfYear.columns = [i]
                long = long.merge(dfYear, how="left", on="Station")

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

    def stations_to_streams(
        self,
        waterbodyType,
        stationID,
        waterbodyID,
        waterbodyName,
        booleanVar,
        radius=10,
    ):
        """Streams: Assign monitoring stations to water bodies via linkage table.

        For unmatched stations: Assign to stream within a radius of 10 meters
        where the location names match the name of the stream.

        For a given year, finds DVFI for a stream with multiple stations
        by taking the median and rounding down.

        Finally, extends to all streams in the current water body plan (VP2) and
        adds the ID, catchment area, and length of each stream in km using the
        feature classes collected via get_fc_from_WFS().
        """
        try:
            # Create longitudinal DataFrame for stations in streams by monitoring approach
            DVFI = self.longitudinal(waterbodyType, "DVFI")
            DVFI_MIB = self.longitudinal(waterbodyType, "DVFI, MIB")
            DVFI_felt = self.longitudinal(waterbodyType, "Faunaklasse, felt")

            # Group by station and keep the first non-null entry each year DVFI>MIB>felt
            long = (
                pd.concat([DVFI, DVFI_MIB, DVFI_felt])
                .groupby(["Station"], as_index=False)
                .first()
            )

            # Read the linkage table
            link = pd.read_excel("linkage\\" + self.linkage[waterbodyType])

            # Merge longitudinal DataFrame with linkage table for water body IDs in VP2
            df = long.merge(
                link[[stationID, waterbodyID, booleanVar]],
                how="left",
                left_on="Station",
                right_on=stationID,
            )

            # Drop water bodies that were only present in the draft for VP2
            df = df[df[booleanVar] != 0].drop(columns=[booleanVar])

            # Make station-ID and coordinates into integers
            df[["Station", "X", "Y"]] = df[["Station", "X", "Y"]].astype(int)

            # Stations in current water body plan (VP2)
            link = df.dropna(subset=[stationID]).drop(columns=[stationID])

            # Stations not covered by the linkage table
            noLink = df[df[stationID].isna()].drop(columns=[stationID, waterbodyID])

            # Create a spatial reference object with the same geoprachical coordinate system
            spatialRef = arcpy.SpatialReference("ETRS 1989 UTM Zone 32N")

            # Create a new feature class shapefile (will overwrite if it already exists)
            fcStations = waterbodyType + "_stations"
            arcpy.CreateFeatureclass_management(
                self.arcPath, fcStations, "POINT", spatial_reference=spatialRef
            )

            # Create fields for 'Station' and 'Location'
            arcpy.AddField_management(fcStations, "Station", "INTEGER")
            arcpy.AddField_management(fcStations, "Location", "TEXT")

            # Create cursor to insert stations that were not in the linkage table
            try:
                with arcpy.da.InsertCursor(
                    fcStations, ["SHAPE@XY", "Station", "Location"]
                ) as cursor:
                    # Loop over each station-ID in df:
                    for index, row in noLink.iterrows():
                        try:
                            # Use cursor to insert new row in feature class
                            cursor.insertRow(
                                [(row["X"], row["Y"]), row["Station"], row["Location"]]
                            )

                        except:
                            # Report other severe error messages from Python or ArcPy
                            tb = sys.exc_info()[
                                2
                            ]  # get traceback object for Python errors
                            tbinfo = traceback.format_tb(tb)[0]
                            print(
                                "Python errors while inserting station {0} in {1}:\nTraceback info:{2}\nError Info:\n{3}\n".format(
                                    str(row["Station"]),
                                    fcStations,
                                    tbinfo,
                                    str(sys.exc_info()[1]),
                                )
                            )
                            print(
                                "ArcPy errors while inserting station {0} in {1}:\n{2}".format(
                                    str(row["Station"]),
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

            # Specify joined feature classes (will overwrite if it already exists)
            fcJoined = fcStations + "_joined"

            # Spatial Join unmatched stations with streams within given radius
            arcpy.SpatialJoin_analysis(
                waterbodyType,
                fcStations,
                fcJoined,
                "JOIN_ONE_TO_MANY",
                "KEEP_COMMON",
                match_option="CLOSEST",
                search_radius=radius,
                distance_field_name="Distance",
            )

            # Specify the field (column) in fc that contain the water body IDs for VP2
            vpID = self.wfs_vpID[waterbodyType]

            # Fields of interest
            fieldsStations = ["Station", "Distance", "Location", vpID]

            # Use SeachCursor to read columns to pd.DataFrame
            stations = [row for row in arcpy.da.SearchCursor(fcJoined, fieldsStations)]
            dfStations = pd.DataFrame(stations, columns=fieldsStations)

            # Add water body names from linkage table and sort by distance (ascending)
            j = (
                dfStations.merge(
                    link[[waterbodyName, waterbodyID]],
                    how="inner",
                    left_on=vpID,
                    right_on=waterbodyID,
                )
                .drop([vpID], axis=1)
                .sort_values("Distance")
            )

            # Capitalize water body names
            j[waterbodyName] = j[waterbodyName].str.upper()

            # Indicate matching water body names
            j["Match"] = np.select([j["Location"] == j[waterbodyName]], [True])

            # Subset to unique stations with their closest matching water body
            jClosest = (
                j[j["Match"] == True].groupby(["Station"], as_index=False).first()
            )

            # Inner merge of noLink stations and jClosest water body with matching name
            jMatches = noLink.merge(
                jClosest[["Station", waterbodyID]], how="inner", on="Station"
            )

            # Concatenate the dfs of all stations that have been matched to a water body
            allMatches = pd.concat([link, jMatches])

            # Group multiple stations in a water body: Take the median and round down
            waterBodies = (
                allMatches.groupby([waterbodyID])
                .median()
                .apply(np.floor)
                .drop(columns=["Station", "X", "Y"])
            )

            # Specify the fields in fc that contain the main catchment area and typology
            main = self.wfs_main[waterbodyType]
            typo = self.wfs_typo[waterbodyType]

            # Fields (columns) for ID, length, main catchment area, and typology
            fields = [vpID, "Shape_Length", main, typo]

            # Use SearchCursor to create df with characteristics of all water bodies
            dataCharacteristics = [
                row for row in arcpy.da.SearchCursor(waterbodyType, fields)
            ]
            dfCharacteristics = pd.DataFrame(dataCharacteristics, columns=fields)

            # Shore length is counted on both sides of the stream
            dfCharacteristics[["length"]] = dfCharacteristics[["Shape_Length"]] * 2
            dfCharacteristics = dfCharacteristics.drop(["Shape_Length"], axis=1)

            # Merge dfCharacteristics with df for all water bodies in VP2
            allVP = dfCharacteristics.merge(
                waterBodies, how="outer", left_on=vpID, right_index=True
            ).set_index(vpID)

            # print('\nallMatches:', len(allMatches),
            #       '\nwaterBodies:', len(waterBodies),
            #       '\ndfCharacteristics:', len(dfCharacteristics),
            #       '\nallVP:', len(allVP))
            # Report the number of stations matched by linkage table and ArcPy
            msg = "Streams:\n{0} stations were linked to a water body by the official linkage table. Besides, {1} were located within 10 meters of a water body carrying the name of the station's location.".format(
                len(link), len(jClosest)
            )
            # print(msg)  # print number of stations in Python
            arcpy.AddMessage(msg)  # return number of stations in ArcGIS

            return allVP

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

        # finally:
        #     # Clean up
        #     for fc in [waterbodyType, fcStations, fcJoined]:
        #         if arcpy.Exists(fc):
        #             arcpy.Delete_management(fc)

    def observed_indicator(self, waterbodyType):
        """Based on the type of water body, set up a longitudinal dataframe
        with the observed indicators for all water bodies.
        """
        try:
            # Create an output folder if it doesn't exist
            os.makedirs(self.path + "\\output", exist_ok=True)

            if waterbodyType == "streams":
                # Create longitudinal df and use linkage table to assign stations to water bodies
                df = self.stations_to_streams(
                    waterbodyType,
                    stationID="DCE_stationsnr",
                    waterbodyID="VP2_g_del_cd",
                    waterbodyName="Navn",
                    booleanVar="VP2_gældende",
                )

            # Save to CSV for later statistical work
            df.to_csv("output\\" + waterbodyType + "_ind_obs.csv")

            return df

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

    def indicator_to_status(self, waterbodyType, df):
        """Streams: Convert DVFI fauna index to index of ecological status."""
        try:
            if waterbodyType == "streams":
                # Convert DVFI fauna index for streams to index of ecological status
                for i in self.years:
                    # Categorical variable for ecological status: Bad, Poor, Moderate, Good, High
                    conditions = [
                        df[i] == 1,
                        (df[i] == 2) | (df[i] == 3),
                        df[i] == 4,
                        (df[i] == 5) | (df[i] == 6),
                        df[i] == 7,
                    ]
                    df[i] = np.select(conditions, [1, 2, 3, 4, 5], default=np.nan)

            return df

        except:
            ## Report severe error messages
            tb = sys.exc_info()[2]  # get traceback object for Python errors
            tbinfo = traceback.format_tb(tb)[0]
            msg = "Could not convert DVFI to ecological status:\nTraceback info:\n{0}Error Info:\n{1}".format(
                tbinfo, str(sys.exc_info()[1])
            )
            print(msg)  # print error message in Python
            arcpy.AddError(msg)  # return error message in ArcGIS
            sys.exit(1)

    def ecological_status(self, waterbodyType, dfIndicator):
        """Based on the type of water body, convert the longitudinal dataframe
        to the EU index of ecological status, i.e. from 1-5 for bad, poor,
        moderate, good, and high water quality respectively.

        Create a table of statistics and export it as an html table.

        Print the length and share of water bodies observed at least once.
        """
        try:
            # Convert index of indicators to index of ecological status
            df = self.indicator_to_status(waterbodyType, dfIndicator)

            # Calculate total length of all water bodies in current water body plan (VP2)
            totalLength = df["length"].sum()

            # Create an empty df for statistics
            stats = pd.DataFrame(
                index=self.years,
                columns=[
                    "Status known (%)",
                    "Share of known is high (%)",
                    "Share of known is good (%)",
                    "Share of known is moderate (%)",
                    "Share of known is poor (%)",
                    "Share of known is bad (%)",
                ],
            )

            # Calculate the above statistics for each year
            for i in self.years:
                y = df[[i, "length"]].reset_index(drop=True)
                y["Known"] = np.select([y[i].notna()], [y["length"]])
                y["High"] = np.select([y[i] == 5], [y["length"]])
                y["Good"] = np.select([y[i] == 4], [y["length"]])
                y["Moderate"] = np.select([y[i] == 3], [y["length"]])
                y["Poor"] = np.select([y[i] == 2], [y["length"]])
                y["Bad"] = np.select([y[i] == 1], [y["length"]])

                # Add shares of total length to stats
                knownLength = y["Known"].sum()
                stats.loc[i] = [
                    100 * knownLength / totalLength,
                    100 * y["High"].sum() / knownLength,
                    100 * y["Good"].sum() / knownLength,
                    100 * y["Moderate"].sum() / knownLength,
                    100 * y["Poor"].sum() / knownLength,
                    100 * y["Bad"].sum() / knownLength,
                ]

            # Save statistics to html for online presentation
            stats.astype(int).to_html("output\\" + waterbodyType + "_eco_obs_stats.md")

            # Shorten column names of statistics
            stats.columns = ["known", "high", "good", "moderate", "poor", "bad"]

            # Save statistics and waterbodies to CSV for later statistical work
            df.to_csv("output\\" + waterbodyType + "_eco_obs.csv")
            stats.to_csv("output\\" + waterbodyType + "_eco_obs_stats.csv")

            # Create missing values graph (heatmap of missing observations by year):
            self.missing_values_graph(df, waterbodyType, "missing")

            # Calculate water bodies that are observed at least once
            observed = df[["length", "typo"]].merge(
                df[self.years].dropna(how="all"),
                how="inner",
                left_index=True,
                right_index=True,
            )

            # Report length and share of water bodies observed at least once.
            msg = "{0} km is the total shore length of {1} included in VP2, of which {2}% of {1} representing {3} km ({4}% of total shore length) have been assessed at least once. On average, {5}% of {1} representing {6} km ({7}% of total shore length) are assessed each year.\n".format(
                int(totalLength) * 10 ** (-3),
                waterbodyType,
                int(100 * len(observed) / len(df)),
                int(observed["length"].sum()) * 10 ** (-3),
                int(100 * observed["length"].sum() / totalLength),
                int(100 * np.mean(df[self.years].count() / len(df))),
                int(stats["known"].mean() * totalLength / 100) * 10 ** (-3),
                int(stats["known"].mean()),
            )
            # print(msg)  # print statistics in Python
            arcpy.AddMessage(msg)  # return statistics in ArcGIS

            return df, stats

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

    def map_book(self, fc, df):
        """Create a pdf map book with a page for each year
        using the fc created with get_fc_from_WFS()
        """
        try:
            # Set the name of the field (column) containing the water body plan IDs
            vpID = self.wfs_vpID[fc]

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
                    with arcpy.da.UpdateCursor(fcYear, [vpID, "status"]) as cursor:
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
                        if arcpy.Exists(fcYear):
                            arcpy.Delete_management(fcYear)
                        if arcpy.Exists(layerYear + ".lyrx"):
                            arcpy.Delete_management(layerYear + ".lyrx")
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

    def missing_values_graph(self, frame, waterbodyType, suffix):
        """Heatmap visualizing observations of ecological status as either
        missing or using the EU index of ecological status, i.e. from 1-5
        for bad, poor, moderate, good, and high water quality respectively.

        Saves a figure of the heatmap.
        """
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
                df, cmap=cm, cbar=False, cbar_kws={"ticks": [0, 1, 2, 3, 4, 5, 6]}
            )
            ax.set(yticklabels=[])
            plt.ylabel(waterbodyType + " (N=" + str(len(df)) + ")", fontsize=14)
            plt.xlabel("")
            plt.title(
                (
                    "Ecological status of "
                    + waterbodyType
                    + ":"
                    + "\nmissing value (grey), bad (red), poor (orange), moderate (yellow), good (green), high (blue)"
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
