"""
Name:       script.py

Label:      Construct and map longitudinal data of ecological status of streams.

Summary:    ThorNoe.GitHub.io/GreenGDP explains the overall approach and methodology.

Rqmts:      ArcGIS Pro must be installed on the system and be up to date.

Usage:      This sandbox is line-by-line implementation of the script supporting 
            WaterbodiesScriptTool in the gis.tbx toolbox, however, for streams only.
            See GitHub.com/ThorNoe/GreenGDP for instructions to run or update it all.

License:    MIT Copyright (c) 2020-2024
Author:     Thor Donsby Noe
"""

###############################################################################
#   0. Imports                                                                #
###############################################################################
import arcpy
import pandas as pd

###############################################################################
#   1. Additional setup - after running parts 0-1 of script.py                #
###############################################################################
arcPath = path + "\\gis.gdb"

###############################################################################
#   2. Additional specifications - after running parts 0-2 of script.py       #
###############################################################################
# Specify the years of interest
years = list(range(year_first, year_last + 1))

# Specify that the fc in question is streams (rather than looping over each fc)
waterbodyType = "streams"
parameterType = "DVFI"
parameterCol = "Indekstype"
valueCol = "Indeks"
fcStations = waterbodyType + "_stations"
fc = wfs_fc[waterbodyType] + ".shp"
radius = 10

###############################################################################
#   3. Run the functions line-by-line                                         #
###############################################################################
arcpy.ListFeatureClasses()
arcpy.Exists(fc)
arcpy.Exists(fcStations)
for field in arcpy.ListFields(fcJoined):
    field.name, field.type
arcpy.Delete_management(fcStations)

# def frame(self, waterbodyType, parameterCol, parameterType, valueCol):
"""Function to set up a Pandas DataFrame for a given type of water body"""
filenames = data[waterbodyType]

# def longitudinal(self, waterbodyType, parameterType):
"""Set up a longitudinal dataframe based on the type of water body.
Streams: For a given year, finds DVFI for a station with multiple
         observations by taking the median and rounding down."""
df = c.frame(waterbodyType, "Indekstype", parameterType, "Indeks")
i = 2001

# def stations_to_streams(self, waterbodyType, radius=15)
"""Streams: Assign monitoring stations to water bodies via linkage table.

        For unmatched stations: Assign to stream within a radius of 15 meters
        where the location names match the name of the stream.

        For a given year, finds DVFI for a stream with multiple stations
        by taking the median and rounding down.

        Finally, extends to all streams in the current water body plan (VP3) and
        adds the ID, catchment area, and length of each stream in km using the
        feature classes collected via get_fc_from_WFS()."""
# Create longitudinal DataFrame for stations in streams by monitoring approach
DVFI = c.longitudinal(waterbodyType, "DVFI")
DVFI_MIB = c.longitudinal(waterbodyType, "DVFI, MIB")
DVFI_felt = c.longitudinal(waterbodyType, "Faunaklasse, felt")
# Read the linkage table
linkage = pd.read_csv("linkage\\" + linkage[waterbodyType])
# Stations covered by the linkage tabel for the current water body plan (VP3)
link.tail(2)
# Stations not covered by the linkage table for VP3
noLink.tail(2)
# Create a new feature class shapefile (will overwrite if it already exists)
arcpy.CreateFeatureclass_management(
    arcPath, fcStations, "POINT", spatial_reference=spatialRef
)
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
                tb = sys.exc_info()[2]  # get traceback object for Python errors
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
dfStations.tail(10)
link.tail(10)
j.tail(2)
for d in range(2,10,2):
    d, len(dfStations[dfStations["Distance"] <= d]), len(j[j["Distance"] <= d])
len(dfStations), len(link), len(j)


# def observed_indicator(self, waterbodyType):
"""Based on the type of water body, set up a longitudinal dataframe
with the observed indicators for all water bodies."""
# Create longitudinal df and use linkage table to assign stations to water bodies
# df = self.stations_to_streams(waterbodyType)
