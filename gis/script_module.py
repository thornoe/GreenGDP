"""
Name:       script_module.py

Label:      Set up longitudinal data, impute missing values, and apply valuation.

Summary:    ThorNoe.GitHub.io/GreenGDP explains the overall approach and methodology.

Rqmts:      ArcGIS Pro must be installed on the system and be up to date.

Usage:      This module supports script.py and WaterbodiesScriptTool in gis.tbx.
            See GitHub.com/ThorNoe/GreenGDP for instructions to run or update it all.

Functions:  The class in this module contains 10 functions of which some are nested:
            - get_data(), get_fc_from_WFS(), map_book(), and values_by_catchment_area() are standalone functions.
            - observed_indicator() calls:
                - longitudinal()
            - impute_missing() calls:
                - ecological_status(), which calls:
                    - indicator_to_status()
                    - missing_values_graph()
            - valuation() calls:
                - BT()
            Descriptions can be seen under each function.

License:    MIT Copyright (c) 2024
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
from cycler import cycler
from scipy import interpolate
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer

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
        WFS_fieldsInFeatureClass,
        WFS_replaceFeatureClasses,
        keepGeodatabase,
    ):
        self.years = list(range(yearFirst, yearLast + 1))
        self.data = dataFilenames
        self.linkage = linkageFilenames
        self.wfs_service = WFS_ServiceURL
        self.wfs_fc = WFS_featureClassNames
        self.wfs_fields = WFS_fieldsInFeatureClass
        self.wfs_replace = WFS_replaceFeatureClasses
        self.keep_gdb = keepGeodatabase
        self.path = os.getcwd()
        self.arcPath = self.path + "\\gis.gdb"

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
        plt.rc("figure", figsize=[10, 6.2])  #  golden ratio

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
            # Specify name of the feature class (fc) for the given type of water body
            WFS_FeatureType = self.wfs_fc[fc]

            # Specify names of the fields (columns) in fc that contain relevant variables
            fields = self.wfs_fields[fc]

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

    def observed_indicator(self, j, radius=15):
        """Set up a longitudinal DataFrame for all water bodies of category j by year t.

        Assign monitoring stations to water bodies in water body plan via linkage table.

        For monitoring stations not included in the linkage table: Assign a station to a waterbody if the station's coordinates are located within said waterbody. For streams, if the station is within a radius of 15 meters of a stream where the name of the stream matches the location name attached to the monitoring station).

        Finally, construct the longitudinal DataFrame of observed biophysical indicator by year for all water bodies in the current water body plan. Separately, save the water body ID, typology, district ID, and shore length of each water body in VP3 using the feature classes collected via the get_fc_from_WFS() function.
        """
        try:
            if j == "streams":
                # Create longitudinal df for stations in streams by monitoring version
                kwargs = dict(
                    f=self.data[j][1],
                    d="Dato",
                    x="Xutm_Euref89_Zone32",
                    y="Yutm_Euref89_Zone32",
                    valueCol="Indeks",
                    parameterCol="Indekstype",
                )
                DVFI_F = self.longitudinal(j, parameter="Faunaklasse, felt", **kwargs)
                DVFI_M = self.longitudinal(j, parameter="DVFI, MIB", **kwargs)
                DVFI = self.longitudinal(j, parameter="DVFI", **kwargs)

                # Observations after 2020 (publiced after ODA database update Jan 2024)
                DVFI2 = self.longitudinal(
                    j,
                    f=self.data[j][0],
                    d="Dato",
                    x="X-UTM",
                    y="Y-UTM",
                    valueCol="Indeks",
                )

                # Group by station; keep last non-missing entry each year, DVFI>MIB>felt
                long = (
                    pd.concat([DVFI_F, DVFI_M, DVFI, DVFI2]).groupby("station").last()
                )

            else:  #  lakes and coastal waters
                # Create longitudinal df for stations
                long = self.longitudinal(
                    j,
                    f=self.data[j][0],
                    d="Startdato",
                    x="X_UTM32",
                    y="Y_UTM32",
                    valueCol="Resultat",
                )
                if j == "lakes":
                    # Obtain the few missing coordinates
                    stations = pd.read_csv("linkage\\" + self.linkage[j][1]).astype(int)
                    stations.columns = ["station", "x", "y"]
                    stations.set_index("station", inplace=True)
                    long[["x", "y"]] = long[["x", "y"]].combine_first(stations)

            # Read the linkage table
            dfLinkage = pd.read_csv("linkage\\" + self.linkage[j][0])

            # Convert station ID to integers
            dfLinkage = dfLinkage.copy()  #  to avoid SettingWithCopyWarning
            dfLinkage.loc[:, "station"] = (
                dfLinkage["station_id"].str.slice(7).astype(int)
            )

            # Merge longitudinal DataFrame with linkage table for water bodies in VP3
            df = long.merge(dfLinkage[["station", "ov_id"]], how="left", on="station")

            # Stations covered by the linkage tabel for the third water body plan VP3
            link = df.dropna(subset=["ov_id"])

            # Convert water body ID (wb) to integers
            link = link.copy()  #  to avoid SettingWithCopyWarning
            if j == "lakes":
                link.loc[:, "wb"] = link["ov_id"].str.slice(6).astype(int)
            else:
                link.loc[:, "wb"] = link["ov_id"].str.slice(7).astype(int)

            # Stations not covered by the linkage table for VP3
            noLink = df[df["ov_id"].isna()].drop(columns=["ov_id"])

            # Create a spatial reference object with same geographical coordinate system
            spatialRef = arcpy.SpatialReference("ETRS 1989 UTM Zone 32N")

            # Specify name of feature class for stations (points)
            fcStations = j + "_stations"

            # Create new feature class shapefile (will overwrite if it already exists)
            arcpy.CreateFeatureclass_management(
                self.arcPath, fcStations, "POINT", spatial_reference=spatialRef
            )

            # Create field for 'station' and list fields
            arcpy.AddField_management(fcStations, "station", "INTEGER")
            fieldsStations = ["SHAPE@XY", "station"]
            if j == "streams":
                # Create field for 'location' and append to list of fields
                arcpy.AddField_management(fcStations, "location", "TEXT")
                fieldsStations.append("location")

            # Create cursor to insert stations that were not in the linkage table
            try:
                with arcpy.da.InsertCursor(fcStations, fieldsStations) as cursor:
                    # Loop over each station-ID in df:
                    for index, row in noLink.iterrows():
                        try:
                            # Use cursor to insert new row in feature class
                            if j == "streams":
                                cursor.insertRow(
                                    [
                                        (row["x"], row["y"]),
                                        row["station"],
                                        row["location"],
                                    ]
                                )
                            else:
                                cursor.insertRow([(row["x"], row["y"]), row["station"]])

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

            # Specify name of feature class for streams in VP3 (polylines)
            fc = j

            # Specify name of joined feature class (polylines)
            fcJoined = fcStations + "_joined"

            # Spatial Join unmatched stations with streams within given radius
            arcpy.SpatialJoin_analysis(
                target_features=fc,
                join_features=fcStations,
                out_feature_class=fcJoined,  #  will overwrite if it already exists
                join_operation="JOIN_ONE_TO_MANY",
                join_type="KEEP_COMMON",
                match_option="CLOSEST",  #  if more than one stream is within radius
                search_radius=radius,  #  match to stream withing radius of station
                distance_field_name="Distance",
            )

            # Specify fields of interest of fcJoined
            if j == "streams":
                fieldsJ = ["station", "ov_id", "ov_navn", "location", "Distance"]
            else:  #  lakes and coastal waters
                fieldsJ = ["station", "ov_id"]

            # Create DataFrame from fcJoined and sort by distance (ascending)
            stations = [row for row in arcpy.da.SearchCursor(fcJoined, fieldsJ)]
            join = pd.DataFrame(stations, columns=fieldsJ)

            # Convert water body ID (wb) to integers
            join = join.copy()  #  to avoid SettingWithCopyWarning
            if j == "lakes":
                join.loc[:, "wb"] = join["ov_id"].str.slice(6).astype(int)
            else:
                join.loc[:, "wb"] = join["ov_id"].str.slice(7).astype(int)

            if j == "streams":
                # Capitalize water body names
                join["ov_navn"] = join["ov_navn"].str.upper()

                # Rename unnamed water bodies to distinguish from named water bodies
                join["location"].mask(
                    join["location"] == "[IKKE NAVNGIVET]", "UDEN NAVN", inplace=True
                )

                # a Indicate that station and stream has the same water body name
                join["match"] = np.select([join["location"] == join["ov_navn"]], [True])

                # Subset to unique stations with their closest matching water body
                jClosest = (
                    join[join["match"] == True]
                    .sort_values("Distance")
                    .groupby("station", as_index=False)
                    .first()
                )

                # Inner merge of noLink stations and jClosest water body with matching name
                noLinkClosest = noLink.merge(jClosest[["station", "wb"]], on="station")

                # df containing all stations that have been matched to a water body
                allMatches = pd.concat([link, noLinkClosest]).drop(
                    columns=["station", "location", "x", "y", "ov_id"]
                )

            else:  #  for lakes and coastal waters
                # df containing all stations that have been matched to a water body
                allMatches = pd.concat([link, join]).drop(
                    columns=["station", "x", "y", "ov_id"]
                )

            # Group multiple stations in a water body: Take the median and round down
            waterbodies = allMatches.groupby("wb").median().apply(np.floor)

            # Fields in fc that contain water body ID, typology, district, and length
            fields = ["ov_id", "ov_typ", "distr_id"]

            # Append field for ecological status as assessed in basis analysis for VP3
            if j == "streams":
                fields.append("til_oko_bb")  #  based on DVFI bottom fauna class
            else:  #  lakes and coastal waters
                fields.append("til_oko_fy")  #  based on chlorophyll concentration

            # Append field for shape length
            if j != "coastal":
                fields.append("Shape_Length")  #  lakes circumference; streams polyline

            # Append field for natural/artificial/strongly modified streams
            if j == "streams":
                fields.append("na_kun_stm")

            # Create df from fc with characteristics of all waterbodies in VP
            dataVP = [row for row in arcpy.da.SearchCursor(fc, fields)]
            dfVP = pd.DataFrame(dataVP, columns=fields)

            if j == "coastal":
                # Drop coastal waterbody without catchment area (Hesselø is uninhabited)
                dfVP = dfVP[dfVP["ov_id"] != "DKCOAST205"]

            # Convert water body ID (wb) to integers
            dfVP = dfVP.copy()  #  to avoid SettingWithCopyWarning
            if j == "lakes":
                dfVP.loc[:, "wb"] = dfVP["ov_id"].str.slice(6).astype(int)
            else:
                dfVP.loc[:, "wb"] = dfVP["ov_id"].str.slice(7).astype(int)

            # Sort by water body ID (wb, ascending)
            dfVP = dfVP.set_index("wb").sort_index()

            # Specify shore length for each category j
            if j == "streams":
                # Shore length is counted on both sides of the stream; convert to km
                dfVP[["length"]] = dfVP[["Shape_Length"]] * 2 / 1000

            elif j == "lakes":
                # Shore length is the circumference, i.e. Shape_Length; convert to km
                dfVP[["length"]] = dfVP[["Shape_Length"]] / 1000

            else:  #  coastal waters
                # Coastline by Zandersen et al.(2022) based on Corine Land Cover 2018
                Geo = pd.read_excel("data\\" + self.data["shared"][2], index_col=0)
                Geo.index.name = "wb"
                # Merge with df for all water bodies in VP3
                dfVP[["length"]] = Geo[["shore coastal"]]

            # Subset columns to relevant water body characteristics
            dfVP = dfVP.drop(columns="ov_id")

            # Save characteristics of water bodies to CSV for later work
            dfVP.to_csv("output\\" + j + "_VP.csv")

            # Merge df for all water bodies in VP3 with df for observed status
            allVP = dfVP[[]].merge(waterbodies, how="left", on="wb")

            # Save observations to CSV for later statistical work
            allVP.to_csv("output\\" + j + "_ind_obs.csv")

            # Report stations matched by linkage table and distance+name respectively
            if j == "streams":
                msg = "{0}:\n{1} out of {2} stations were linked to a water body by the official linkage table. Besides, {3} stations were located within {4} meters of a water body carrying the name of the station's location.".format(
                    str(j),
                    len(link),
                    len(df),
                    len(jClosest),
                    str(radius),
                )
            else:
                msg = "{0}:\n{1} out of {2} stations were linked to a water body by the official linkage table for VP3. Besides, {3} stations were located inside the polygon shape of a water body present in VP3.".format(
                    str(j),
                    len(link),
                    len(df),
                    len(join),
                )
            # print(msg)  # print number of stations in Python
            arcpy.AddMessage(msg)  # return number of stations in ArcGIS

            return allVP, dfVP

        except:
            # Report severe error messages
            tb = sys.exc_info()[2]  # get traceback object for Python errors
            tbinfo = traceback.format_tb(tb)[0]
            pymsg = "Python errors while assigning stations to water bodies:\nTraceback info:\n{0}Error Info:\n{1}".format(
                tbinfo, str(sys.exc_info()[1])
            )
            arcmsg = (
                "ArcPy errors while assigning stations to water bodies:\n{0}".format(
                    arcpy.GetMessages(severity=2)
                )
            )
            print(pymsg)  # print Python error message in Python
            print(arcmsg)  # print ArcPy error message in Python
            arcpy.AddError(pymsg)  # return Python error message in ArcGIS
            arcpy.AddError(arcmsg)  # return ArcPy error message in ArcGIS
            sys.exit(1)

        finally:  # Clean up
            for fc in [fcStations, fcJoined]:  # Delete feature classes
                if arcpy.Exists(fc):
                    arcpy.Delete_management(fc)
            del fcStations, fcJoined

    def longitudinal(self, j, f, d, x, y, valueCol, parameterCol=0, parameter=0):
        """Set up a longitudinal DataFrame for all stations in category j by year t.

        Streams: For a given year, find the DVFI index value of bottom fauna for a station with multiple observations by taking the median and rounding down.

        Lakes and coastal waters: For a given year, estimate the chlorophyll summer average for every station monitored at least four times during May-September by linear interpolating of daily data from May 1 to September 30 (or extrapolate by inserting the first/last observation from May/September if there exist no observations outside of said period that are no more than 6 weeks away from the first/last observation in May/September).
        """
        try:
            # Read the data for biophysical indicator (source: https://ODAforalle.au.dk)
            df = pd.read_excel("data\\" + f)
            # Rename the station ID column and make it the index of df
            df = df.set_index("ObservationsStedNr").rename_axis("station")
            # Create 'Year' column from the date column
            df = df.copy()  #  to avoid SettingWithCopyWarning
            df.loc[:, "year"] = df[d].astype(str).str.slice(0, 4).astype(int)
            if parameterCol != 0:
                # Subset the data to only contain the relevant parameter
                df = df[df[parameterCol] == parameter]
            # Drop missing values and sort by year
            df = df.dropna(subset=valueCol).sort_values("year")
            # Column names for the final longitudinal DataFrame besides the indicator
            cols = ["x", "y"]
            if j == "streams":
                cols.append("location")  #  add location name for final DataFrame
                df = df[[x, y, "Lokalitetsnavn", "year", valueCol]]  #  subset columns
                df.columns = cols + ["year", "ind"]  #  shorten column names
                df["location"] = df["location"].str.upper()  # capitalize location names
                df = df[df["ind"] != "U"]  #  drop obs with unknown indicator value "U"
                df["ind"] = df["ind"].astype(int)  #  convert indicator to integer
            else:  # Lakes and coastal waters
                # Convert date column to datetime format
                df[d] = pd.to_datetime(df[d].astype(str), format="%Y%m%d")  #  convert
                df = df[[x, y, d, "year", valueCol]]  #  subset to relevant columns
                df.columns = cols + ["date", "year", "ind"]  #  shorten column names
                df.set_index("date", append=True, inplace=True)  #  add 'date' to index

            # Replace 0-values with missing in 'x' and 'y' columns
            df[["x", "y"]] = df[["x", "y"]].replace(0, np.nan)

            # Set up a longitudinal df with every station and its last non-null entry
            long = df[cols].groupby(level="station").last()

            # For each year t, add a column with observations for the indicator
            for t in df["year"].unique():
                # Subset to year t
                dft = df[df["year"] == t]
                # Subset to station and indicator columns only
                dft = dft[["ind"]]
                if j == "streams":
                    # Group multiple obs for a station: Take the median and round down
                    dfYear = dft.groupby("station").median().apply(np.floor).astype(int)
                    # Rename the indicator column to year t
                    dfYear.columns = [t]
                else:
                    # Generate date range 6 weeks before and after May 1 and September 30
                    dates = pd.date_range(str(t) + "-03-20", str(t) + "-11-11")
                    summer = pd.date_range(str(t) + "-05-01", str(t) + "-09-30")
                    # Subset to dates in the date range
                    dft = dft.loc[
                        (dft.index.get_level_values("date") >= dates.min())
                        & (dft.index.get_level_values("date") <= dates.max())
                    ]
                    # Take the mean of multiple obs for any station-date combination
                    dft = dft.groupby(level=["station", "date"]).mean()
                    # Pivot dft with dates as index and stations as columns
                    dft = dft.reset_index().pivot(
                        index="date", columns="station", values="ind"
                    )
                    # Subset dft to rows that are within the summer date range
                    dftSummer = dft.loc[dft.index.isin(summer), :]
                    # Drop columns (stations) with less than 4 values
                    dftSummer = dftSummer.dropna(axis=1, thresh=4)
                    # Create empty DataFrame with dates as index and stations as columns
                    dfd = pd.DataFrame(index=dates, columns=dftSummer.columns)
                    # Update the empty dfd with the chlorophyll observations in dft
                    dfd.update(dft)
                    # Convert to numeric, errors='coerce' will set non-numeric values to NaN
                    dfd = dfd.apply(pd.to_numeric, errors="coerce")
                    # Linear Interpolation of missing values with consecutive gap < 6 weeks
                    dfd = dfd.interpolate(limit=41, limit_direction="both")
                    # Linear Interpolation for May-September without limit
                    dfd = dfd.loc[dfd.index.isin(summer), :].interpolate(
                        limit_direction="both"
                    )
                    # Drop any column that might somehow still contain missing values
                    dfd = dfd.dropna(axis=1)
                    # Take the summer average of chlorophyll for each station in year t
                    dfYear = dfd.groupby(dfd.index.year).mean().T
                # Merge into longitudinal df
                long = long.merge(dfYear, how="left", on="station")

            return long

        except:
            ## Report severe error messages
            tb = sys.exc_info()[2]  # get traceback object for Python errors
            tbinfo = traceback.format_tb(tb)[0]
            msg = "Could not set up DataFrame for {0}:\nTraceback info:\n{1}Error Info:\n{2}".format(
                j, tbinfo, str(sys.exc_info()[1])
            )
            print(msg)  # print error message in Python
            arcpy.AddError(msg)  # return error message in ArcGIS
            sys.exit(1)

    def impute_missing(self, j, dfEcoObs, dfVP, index):
        """Impute ecological status for all water bodies from the observed indicator."""
        try:
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
                cols = ["Soft bottom", "Natural", "Small"]

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
                cols = ["Saline"]

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
                cols = ["Sediment", "Deep"]

            # Merge DataFrame for observed values with DataFrame for dummies
            dfObsSelected = dfEcoObs.merge(typ[cols], on="wb")  #  selected predictors

            # Iterative imputer using BayesianRidge() estimator with increased tolerance
            imputer = IterativeImputer(tol=1e-1, max_iter=100, random_state=0)

            # Fit imputer, transform data iteratively, and limit to years of interest
            dfImp = pd.DataFrame(
                imputer.fit_transform(np.array(dfObsSelected)),
                index=dfObsSelected.index,
                columns=dfObsSelected.columns,
            )[dfEcoObs.columns]

            # Calculate a 5-year moving average (MA) for each water body to reduce noise
            MA = dfImp.T.rolling(window=5, min_periods=3, center=True).mean().T

            # Merge imputed ecological status each year with basis analysis for VP3
            dfImpMA = MA.merge(dfEcoObs["basis"], on="wb")

            # Convert the imputed ecological status to categorical scale {0, 1, 2, 3, 4}
            dfImp2, impStats = self.ecological_status(
                j, dfImp[self.years], dfVP, "imp", index
            )

            # Convert moving average of the imputed eco status to the categorical scale
            dfImpMA2, impStatsMA = self.ecological_status(
                j, dfImpMA, dfVP, "imp_MA", index
            )

            return dfImp2[self.years], dfImpMA2[self.years], impStats, impStatsMA

        except:
            ## Report severe error messages
            tb = sys.exc_info()[2]  # get traceback object for Python errors
            tbinfo = traceback.format_tb(tb)[0]
            msg = "Could not impute biophysical indicator to ecological status for {0}:\nTraceback info:\n{1}Error Info:\n{2}".format(
                j, tbinfo, str(sys.exc_info()[1])
            )
            print(msg)  # print error message in Python
            arcpy.AddError(msg)  # return error message in ArcGIS
            sys.exit(1)

    def ecological_status(self, j, dfIndicator, dfVP, suffix="obs", index=None):
        """Call indicator_to_status() to convert the longitudinal DataFrame to the EU index of ecological status, i.e., from 0-4 for Bad, Poor, Moderate, Good, and High water quality based on the category and typology of each water body.

        Also call missing_values_graph() to map missing observations by year.

        Create a table of statistics and export it as an html table.

        Print the shore length and share of water bodies observed at least once."""
        try:
            # Index for statistics by year and each ecological status
            indexStats = self.years

            if suffix == "obs":
                # Convert observed biophysical indicator to ecological status
                dfEcoObs = self.indicator_to_status(j, dfIndicator, dfVP)

                # Specify the biophysical indicator for the current category
                if j == "streams":
                    indicator = "til_oko_bb"  #  bottom fauna measured as DVFI index
                else:  #  lakes and coastal waters
                    indicator = "til_oko_fy"  #  phytoplankton measured as chlorophyll

                # Include ecological status as assessed in basis analysis for VP3
                basis = dfVP[[indicator]].copy()

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

                # Replace Danish strings in the df with the corresponding ordinal values
                basis.replace(status_dict, inplace=True)
                basis.columns = ["basis"]

                # Merge observed ecological status each year with basis analysis for VP3
                dfEco = dfEcoObs.merge(basis, on="wb")

                # Add the basis analysis to the index for statistics by year and status
                indexStats.append("basis")

            else:
                # Imputed ecological status using a continuous scale
                dfEco = dfIndicator.copy()

            # Save CSV of data on mean ecological status by water body and year
            dfEco.to_csv("output\\" + j + "_eco_" + suffix + ".csv")

            if suffix != "obs":
                # Prepare for statistics and missing values graph
                for t in dfEco.columns:
                    # Precautionary conversion of imputed status to categorical scale
                    conditions = [
                        dfEco[t] < 1,  # Bad
                        (dfEco[t] >= 1) & (dfEco[t] < 2),  #  Poor
                        (dfEco[t] >= 2) & (dfEco[t] < 3),  #  Moderate
                        (dfEco[t] >= 3) & (dfEco[t] < 4),  #  Good
                        dfEco[t] >= 4,  #  High
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
                index=indexStats,
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

            # Calculate the above statistics for each year
            for t in indexStats:
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
                observed = dfVP[["length"]].merge(
                    dfEco.drop(columns="basis").dropna(how="all"),
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
                    round(stats["known"].mean() / 100 * totalLength),
                    round(stats["known"].mean()),
                )
                # print(msg)  # print statistics in Python
                arcpy.AddMessage(msg)  # return statistics in ArcGIS

                return dfEco, stats["not good"], indexSorted

            return dfEco, stats["not good"]

        except:
            # Report severe error messages
            tb = sys.exc_info()[2]  # get traceback object for Python errors
            tbinfo = traceback.format_tb(tb)[0]
            msg = "Could not create df with observed ecological status for {0}:\nTraceback info:\n{1}Error Info:\n{2}".format(
                j, tbinfo, str(sys.exc_info()[1])
            )
            print(msg)  # print error message in Python
            arcpy.AddError(msg)  # return error message in ArcGIS
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
            arcpy.AddError(msg)  # return error message in ArcGIS
            sys.exit(1)

    def missing_values_graph(self, j, frame, suffix="obs", index=None):
        """Heatmap visualizing observations of ecological status as either missing or using the EU index of ecological status, i.e., from 0-4 for Bad, Poor, Moderate, Good, and High water quality respectively.
        Saves a figure of the heatmap."""
        try:
            if suffix == "obs":
                # Sort by eco status in basis analysis then number of missing values
                df = frame.copy()
                df["nan"] = df.shape[1] - df.count(axis=1)
                df = df.sort_values(["basis", "nan"], ascending=False)[self.years]

                # Save index to reuse the order after imputing the missing values
                index = df.index

                # Specify heatmap to show missing values as gray
                colors = ["grey", "red", "orange", "yellow", "green", "blue"]
                uniqueValues = [-1, 0, 1, 2, 3, 4]
                description = "Missing value (gray), Bad (red), Poor (orange), Moderate (yellow), Good (green), High (blue)"

            else:
                # Sort water bodies by number of missing values prior to imputation
                df = frame.copy().reindex(index)[self.years]

                # Specify heatmap without missing values
                colors = ["red", "orange", "yellow", "green", "blue"]
                uniqueValues = [0, 1, 2, 3, 4]
                description = "Bad (red), Poor (orange), Moderate (yellow), Good (green), High (blue)"

            # Plot heatmap
            df.fillna(-1, inplace=True)
            cm = sns.xkcd_palette(colors)
            plt.figure(figsize=(12, 12))
            ax = sns.heatmap(
                df,
                cmap=cm,
                cbar=False,
                cbar_kws={"ticks": uniqueValues},
            )
            ax.set(yticklabels=[])
            plt.ylabel(
                str(len(df)) + " " + j + " ordered by number of missing values",
                fontsize=14,
            )
            plt.xlabel("")
            plt.title(
                ("Ecological status of " + j + "\n" + description),
                fontsize=14,
            )
            plt.tight_layout()
            plt.savefig(
                "output\\" + j + "_eco_" + suffix + ".pdf",
                bbox_inches="tight",
            )

            return index

        except:
            # Report severe error messages
            tb = sys.exc_info()[2]  # get traceback object for Python errors
            tbinfo = traceback.format_tb(tb)[0]
            msg = "Could not create missing values graph (heatmap) for ecological status of {0}:\nTraceback info:\n{1}Error Info:\n{2}".format(
                j, tbinfo, str(sys.exc_info()[1])
            )
            print(msg)  # print error message in Python
            arcpy.AddError(msg)  # return error message in ArcGIS
            sys.exit(1)

    def values_by_catchment_area(self, j, dfEcoImp, dfVP):
        """Assign water bodies to coastal catchment areas and calculate the weighted arithmetic mean of ecological status after truncating from above at Good status.
        For each year t, set up df with variables for the Benefit Transfer equation."""
        try:
            if j == "coastal":
                dfEcoImpCatch = dfEcoImp.copy()

                # ID is shared between coastal waters and coastal catchment areas v
                dfEcoImpCatch["v"] = dfEcoImpCatch.index

            else:  #  streams and lakes to coastal catchment areas
                # Specify name of joined feature class (polygons)
                jCatch = j + "_catch"

                # Join water bodies with the catchment area they have their center in
                arcpy.SpatialJoin_analysis(
                    target_features=j,
                    join_features="catch",
                    out_feature_class=jCatch,  #  will overwrite if it already exists
                    join_operation="JOIN_ONE_TO_MANY",
                    match_option="HAVE_THEIR_CENTER_IN",
                )

                # Fields in fc that contain coastal catchment area ID and water body ID
                fields = ["op_id", "ov_id"]

                # Create DataFrame from jCatch of water bodies in each catchment area
                dataCatch = [row for row in arcpy.da.SearchCursor(jCatch, fields)]
                dfCatch = pd.DataFrame(dataCatch, columns=fields)

                # Convert water body ID (wb) and coastal catchment area ID to integers
                dfCatch = dfCatch.copy()  #  to avoid SettingWithCopyWarning
                if j == "lakes":
                    dfCatch.loc[:, "wb"] = dfCatch["ov_id"].str.slice(6).astype(int)
                else:
                    dfCatch.loc[:, "wb"] = dfCatch["ov_id"].str.slice(7).astype(int)
                dfCatch["v"] = dfCatch["op_id"]

                # Subset to columns; water body ID as index; sort by catchment area ID
                dfCatch = dfCatch[["wb", "v"]].set_index("wb").sort_values(by="v")

                # Assign unjoined water bodies to their relevant coastal catchment area
                if j == "streams":
                    dfCatch.loc[3024, "v"] = "113"  #  Kruså to Inner Flensborg Fjord
                    dfCatch.loc[8504, "v"] = "233"  #  outlet from Kilen to Venø Bugt
                elif j == "lakes":
                    dfCatch.loc[342, "v"] = "233"  #  Nørskov Vig to Venø Bugt
                    dfCatch.loc[11206, "v"] = "80"  #  Gamborg Nor to Gamborg Fjord
                    dfCatch.loc[11506, "v"] = "136"  #  Lille Langesø to Indre Randers F

                # Merge df for imputed ecological status w. coastal catchment area
                dfEcoImpCatch = dfEcoImp.merge(dfCatch.astype(int), on="wb")

            # Merge df for imputed ecological status w. shore length
            dfEco = dfEcoImpCatch.merge(dfVP[["length"]], on="wb")  #  length

            # List of coastal catchment areas where category j is present
            j_present = list(dfEco["v"].unique())

            # Total length of water bodies of category j by coastal catchment area v
            shores_v = dfEco[["v", "length"]].groupby("v").sum().iloc[:, 0]

            # Demographics by coastal catchment area v and year t (1990-2018)
            dem = pd.read_csv(
                "data\\" + self.data["shared"][1], index_col=[0, 1]
            ).sort_index()

            # Years used for interpolation of demographics
            t_old = np.arange(1990, 2018 + 1)
            t_new = np.arange(1990, 2020 + 1)

            # For each coastal catchment area v, extrapolate demographics to 2019-2020
            frames_v = {}  #  dictionary to store df for each coastal catchment area v
            for v in dem.index.get_level_values("v").unique():
                df = pd.DataFrame(
                    index=t_new
                )  #  empty df to store values for each year t
                for col in dem.columns:
                    # Function for linear extrapolation
                    f = interpolate.interp1d(
                        t_old, dem.loc[v, col], fill_value="extrapolate"
                    )
                    df[col] = f(t_new)
                frames_v[v] = df  #  store df in dictionary of DataFrames
            dfDem = pd.concat(frames_v).sort_index()
            dfDem.index.names = ["v", "t"]

            # Consumer Price Index by year t (1990-2020)
            CPI = pd.read_excel("data\\" + self.data["shared"][0], index_col=0)

            # Merge CPI with demographics by v and t (households, age, and hh income)
            Dem = dfDem[["N"]].merge(
                CPI["CPI"], "left", left_index=True, right_index=True
            )

            # Dummy for mean age > 45 in catchment area
            Dem["D age"] = np.select([dfDem["age"] > 45], [1])

            # Mean gross real household income (100,000 DKK, 2018 prices) by v and t
            Dem["y"] = dfDem["income"] * CPI.loc[2018, "CPI"] / Dem["CPI"] / 100000
            Dem["ln y"] = np.log(Dem["y"])  #  log mean gross real household income
            Dem = Dem.loc[j_present].reorder_levels([1, 0]).sort_index()

            # Geographical data by coastal catchment area v (assumed time-invariant)
            Geo = pd.read_excel("data\\" + self.data["shared"][2], index_col=0)
            Geo.index.name = "v"
            Geo = Geo.loc[j_present].sort_index()

            # For each year t, create a df of variables needed for benefit transfer
            frames_t = {}  #  create empty dictionary to store a df for each year t

            # Truncate DataFrame for ecological status of water bodies from above/below
            Q = dfEco.copy()
            Q[self.years] = Q[self.years].mask(Q[self.years] > 3, 3)  #  above at Good
            Q[self.years] = Q[self.years].mask(Q[self.years] < 0, 0)  #  below at Bad

            # DataFrame with dummy for less than good ecological status
            SL = Q.copy()
            SL[self.years] = (
                SL[self.years].mask(SL[self.years] < 3, 1).mask(SL[self.years] >= 3, 0)
            )

            # For each year t, create df by v for variables needed for benefit transfer
            for t in self.years:
                df = pd.DataFrame()  #  empty df for values by coastal catchment area

                # Q is mean ecological status of water bodies weighted by shore length
                Q[t] = Q[t] * Q["length"]  #  ecological status × shore length
                df["Q"] = Q[["v", t]].groupby("v").sum()[t] / shores_v

                if t > 1989:
                    df["ln y"] = Dem.loc[t, "ln y"]  #  ln mean gross real household inc
                    df["D age"] = Dem.loc[t, "D age"]  #  dummy for mean age > 45 years
                    SL[t] = SL[t] * SL["length"]  #  shore length if status < good
                    SL_not_good = SL[["v", t]].groupby("v").sum()  #  if status < good
                    df["ln PSL"] = SL_not_good[t] / Geo["shores all j"]  #  proportional
                    ln_PSL = np.log(df.loc[df["ln PSL"] > 0, "ln PSL"])  #  log PSL
                    ln_PSL_full = pd.Series(index=df.index)  #  empty series with index
                    ln_PSL_full[df["ln PSL"] != 0] = ln_PSL  #  fill with ln_PSL if > 0
                    df["ln PSL"] = df["ln PSL"].mask(df["ln PSL"] > 0, ln_PSL_full)
                    df["ln PAL"] = Geo["ln PAL"]  #  proportion arable land
                    df["SL"] = SL_not_good / 1000  #  SL in 1,000 km
                    if j == "lakes":
                        df["D lake"] = 1
                    else:
                        df["D lake"] = 0
                    df["N"] = Dem.loc[t, "N"]  #  number of households

                frames_t[t] = df  #  store df in dictionary of DataFrames

            dfBT = pd.concat(frames_t)
            dfBT.index.names = ["t", "v"]

            return dfBT, shores_v

        except:
            # Report severe error messages
            tb = sys.exc_info()[2]  # get traceback object for Python errors
            tbinfo = traceback.format_tb(tb)[0]
            msg = "Could not set up df with variables by coastal catchment area for {0}:\nTraceback info:\n{1}Error Info:\n{2}".format(
                j, tbinfo, str(sys.exc_info()[1])
            )
            print(msg)  # print error message in Python
            arcpy.AddError(msg)  # return error message in ArcGIS
            sys.exit(1)

    def valuation(self, dfBT, real=True, investment=False, factor=False):
        """Valuation as either Cost of Water Pollution (CWP) or Investment Value (IV).
        If not set to return real values (2018 prices), instead returns values in the prices of both the current year and the preceding year (for chain linking).
        """
        try:
            # Copy DataFrame with the variables needed for the benefit transfer equation
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
            df[["D age", "D lake", "N"]] = df[["D age", "D lake", "N"]].astype(int)

            # Consumer Price Index by year t (1990-2020)
            CPI_NPV = pd.read_excel("data\\" + self.data["shared"][0], index_col=0)

            # Merge data with CPI to correct for assumption of unitary income elasticity
            kwargs = dict(how="left", left_index=True, right_index=True)
            df1 = df.merge(CPI_NPV, **kwargs)
            df1["unityMWTP"] = self.BT(df1)  #  MWTP assuming unitary income elasticity

            if factor is False:
                # Calculate factor that MWTP is increased by if using estimated income ε
                df2018 = df1[df1.index.get_level_values("t") == 2018].copy()
                df2018["elastMWTP"] = self.BT(df2018, elast=1.453)  #  meta reg income ε
                df2018["factor"] = df2018["elastMWTP"] / df2018["unityMWTP"]
                df2018 = df2018.droplevel("t")
                df2 = df1.merge(df2018[["factor"]], **kwargs)
                df2 = df2.reorder_levels(["j", "t", "v"]).sort_index()

            else:
                df2 = df1.copy()

            # Adjust with factor of actual ε over unitary ε; set MWTP to 0 for certain Q
            df2["MWTP"] = df2["unityMWTP"] * df2["factor"] * df2["nonzero"]

            # Aggregate real MWTP per hh over households in coastal catchment area
            df2["CWP"] = df2["MWTP"] * df2["N"] / 1e06  #  million DKK (2018 prices)

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
                    df2["CWP"] = df2["CWP"] * CPI_NPV.loc[2018, "NPV"]

                    # Rename CWP to IV (investment value of water quality improvements)
                    df2["IV"] = df2["CWP"]  #  million DKK (2018 prices)

                    return df2[["IV"]]  #  return real investment value by j, t, v

            if real is True:
                if factor is True:
                    return df2[["CWP"]]  #  real cost of water pollution by j, t, v

                else:
                    return df2  #  return full df to use df2["factor"] for decomposition

            # Aggregate nominal MWTP per hh over households in coastal catchment area
            df2["CWPn"] = df2["CWP"] * df2["CPI"] / CPI_NPV.loc[2018, "CPI"]

            # CWP in the prices of the preceding year (for year-by-year chain linking)
            df2["D"] = df2["CWPn"] * df2["CPI t-1"] / df2["CPI"]

            # Aggregate over coastal catchment areas
            grouped = (
                df2[["CWPn", "D"]]
                .groupby(["j", "t"])
                .sum()
                .unstack(level=0)
                .rename_axis(None)
                .rename_axis([None, None], axis=1)
            )

            if investment is False:
                # Rename CWP in prices of current year, and preceding year respectively
                grouped.columns = grouped.columns.set_levels(
                    [
                        "Cost (current year's prices, million DKK)",
                        "Cost (preceding year's prices, million DKK)",
                    ],
                    level=0,
                )

            else:
                # Rename IV in prices of current year, and preceding year respectively
                grouped.columns = grouped.columns.set_levels(
                    [
                        "Investment value (current year's prices, million DKK)",
                        "Investment value (preceding year's prices, million DKK)",
                    ],
                    level=0,
                )

            return grouped  #  in prices of current year and preceding year respectively

        except:
            # Report severe error messages
            tb = sys.exc_info()[2]  # get traceback object for Python errors
            tbinfo = traceback.format_tb(tb)[0]
            msg = "Could not apply valuation to df {0}:\nTraceback info:\n{1}Error Info:\n{2}".format(
                d, tbinfo, str(sys.exc_info()[1])
            )
            print(msg)  # print error message in Python
            arcpy.AddError(msg)  # return error message in ArcGIS
            sys.exit(1)

    def BT(self, df, elast=1):
        """Apply Benefit Transfer equation from meta study (Zandersen et al., 2022)"""
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
                - 0.378 * df["D lake"]
            )

            # Real MWTP per household (DKK, 2018 prices) using the meta study variance
            MWTP = np.exp(lnMWTP + (0.136 + 0.098) / 2)  #  variance components

            return MWTP

        except:
            # Report severe error messages
            tb = sys.exc_info()[2]  # get traceback object for Python errors
            tbinfo = traceback.format_tb(tb)[0]
            msg = "Could not apply benefit transfer equation to df {0} with elasticity {1}:\nTraceback info:\n{1}Error Info:\n{2}".format(
                df, tbinfo, str(sys.exc_info()[1])
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
            for t in self.years[::-1]:
                try:
                    # Copy feature class from template
                    fcYear = fc + str(t) + "fc"
                    if arcpy.Exists(fcYear):
                        arcpy.Delete_management(fcYear)
                    arcpy.CopyFeatures_management(fc, fcYear)

                    # Create update cursor for feature layer
                    with arcpy.da.UpdateCursor(fcYear, ["ov_id", "status"]) as cursor:
                        try:
                            # For each row/object in feature class add ecological status from df
                            for row in cursor:
                                if row[0] in df.index:
                                    row[1] = df.loc[row[0], t]
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
                    layerYear = fc + str(t)
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
                        fc, t, tbinfo, str(sys.exc_info()[1])
                    )
                    arcmsg = (
                        "ArcPy errors while creating a PDF for {0}, {1}:\n{2}".format(
                            fc, t, arcpy.GetMessages(severity=2)
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
