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
# Import Operation System (os) and ArcPy package (requires ArcGIS Pro installed)
import os

import arcpy

###############################################################################
#   1. Setup                                                                  #
###############################################################################
# Specify the parent folder as the working directory of the operating system
# os.chdir(arcpy.GetParameterAsText(0))
root = r"C:\Users\au687527\GitHub\GreenGDP"
path = root + "\\gis"
arcPath = path + "\\gis.gdb"
arcpy.env.workspace = path
os.chdir(path)

# Set overwrite option
arcpy.env.overwriteOutput = True

# Specify whether to keep the geodatabase when the script finishes
# keep_gdb = arcpy.GetParameterAsText(1)
keep_gdb = 1

###############################################################################
#   2. Specifications                                                         #
###############################################################################
# Specify the years of interest
year_first = 1987
year_last = 2020
years = list(range(year_first, year_last + 1))

# Specify the names of each type of water body and its data files
data = {"streams": "streams_DVFI.xlsx"}

# Specify the names of the corresponding linkage files
linkage = {"streams": "streams_stations_VP3.xlsx"}

# WFS service URL for the current water body plan (VP3 covers 2021-2027)
wfs_service = "https://wfs2-miljoegis.mim.dk/vp3endelig2022/ows?service=WFS&request=Getcapabilities"

# Specify the name of the feature class (fc) for each type of water body
wfs_fc = {
    "catch": "vp3e2022_kystvand_opland_afg",
    "coastal": "vp3e2022_marin_samlet_1mil",
    "lakes": "vp3e2022_soe_samlet",
    "streams": "vp3e2022_vandloeb_samlet",
}

# Specify the name of the field (column) in fc that contains the ID of the water body
# wfs_vpID = {
#     "coastal": "ov_id",
#     "lakes": "ov_id",
#     "streams": "ov_id",
# }

# Specify the name of the field (column) in fc that contains the main catchment area
# wfs_main = {"catch": "op_id", "coastal": "mst_id"}

# Specify the name of the field (column) in fc that contains the typology of the water body
# wfs_typo = {"coastal": "ov_typ", "lakes": "ov_typ", "streams": "ov_typ"}

###############################################################################
#   3. Run the functions line-by-line                                         #
###############################################################################
fc = "catch"
# fc = "streams"

# def get_fc_from_WFS(self, fc):
"""Create a feature class from a WFS service given the type of water body.
Also create a template with only the most necessary fields.
"""
# Set names of the feature class for the given type of water body
WFS_FeatureType = wfs_fc[fc]
# Set the names of the fields (columns) in fc that contain the ID (and typology)
# if fc == "catch":
fields = ["op_id"]
# else:
# fields = ["ov_id", "ov_typ"]
if arcpy.Exists(fc):
    "exists"
    # arcpy.Delete_management(fc)  #  if making changes to the fc template
if not arcpy.Exists(fc):
    "doesn't exist"
# Execute the WFSToFeatureClass tool to download the feature class.
arcpy.conversion.WFSToFeatureClass(
    wfs_service,
    WFS_FeatureType,
    path,
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
