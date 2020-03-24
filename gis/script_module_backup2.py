import numpy as np
import pandas as pd
import arcpy, sys, traceback, os, urllib
arcpy.env.overwriteOutput = True    # set overwrite option



class Water_Quality:
    """ Class for all data processing and mapping functions
    """
    def __init__(self, dataFilenames, linkageFilenames, WFS_featureClassNames,
                 WFS_fieldNamesForWaterBodyID, WFS_fieldNamesForWaterBodySize):
        self.data = dataFilenames
        self.linkage = linkageFilenames
        self.wfs_fc = WFS_featureClassNames
        self.wfs_vpID = WFS_fieldNamesForWaterBodyID
        self.wfs_size = WFS_fieldNamesForWaterBodySize
        self.path = os.getcwd()
        self.arcPath = self.path + '\\gis.gdb'
        arcpy.env.workspace = self.arcPath  # Set the ArcPy workspace



    def get_data(self):
        """ Function to check that the folders and their files exist.
            Otherwise creates the folder and downloads the files from GitHub.
        """
        try:
            # Dictionary for all data and linkage files
            allFiles = {'data': [b for a in list(self.data.values()) for b in a],
                        'linkage': [a for a in list(self.linkage.values())]}
        except:
            # Report severe error messages
            tb = sys.exc_info()[2]  # get traceback object for Python errors
            tbinfo = traceback.format_tb(tb)[0]
            msg = 'Could not set up dictionary with all files:\nTraceback info:\n{0}Error Info:\n{1}'\
                  .format(tbinfo, str(sys.exc_info()[1]))
            print(msg)            # print error message in Python
            arcpy.AddError(msg)   # return error message in ArcGIS
            sys.exit(1)

        for key, filenames in allFiles.items():
            try:
                # Create the folder if it doesn't exist
                newPath = self.path + '\\' + key
                os.makedirs(newPath, exist_ok=True)
                os.chdir(newPath)

                for f in filenames:
                    # Download the files if they don't exist
                    if not os.path.exists(f):
                        try:
                            url = 'https://github.com/thornoe/GNNP/raw/master/gis/' + key + '/' + f
                            urllib.request.urlretrieve(url, f)
                        except urllib.error.URLError as e:
                            ## Report URL error messages
                            urlmsg = 'URL error for {0}:\n{1}'.format(f, e.reason)
                            print(urlmsg)            # print URL error message in Python
                            arcpy.AddError(urlmsg)   # return URL error message in ArcGIS
                            sys.exit(1)
                        except:
                            ## Report other severe error messages
                            tb = sys.exc_info()[2]  # get traceback object for Python errors
                            tbinfo = traceback.format_tb(tb)[0]
                            msg = 'Could not download {0}:\nTraceback info:\n{1}Error Info:\n{2}'\
                                  .format(f, tbinfo, str(sys.exc_info()[1]))
                            print(msg)            # print error message in Python
                            arcpy.AddError(msg)   # return error message in ArcGIS
                            sys.exit(1)

            except OSError as e:
                # Report system errors
                tb = sys.exc_info()[2]  # get traceback object for Python errors
                tbinfo = traceback.format_tb(tb)[0]
                OSmsg = 'System error for {0} folder:\nTraceback info:\n{1}Error Info:\n{2}'\
                        .format(folderName, tbinfo, str(sys.exc_info()[1]))
                print(OSmsg)            # print system error message in Python
                arcpy.AddError(OSmsg)   # return system error message in ArcGIS
                sys.exit(1)

            finally:
                # Change the directory back to the original working folder
                os.chdir(self.path)



    def get_fc_from_WFS(self, fc, WFS_Service):
        """ Create a feature class from a WFS service given the type of water body.
            Also create a template with only the most necessary fields.
        """
        try:
            # Set names of the feature class for the given type of water body
            WFS_FeatureType = self.wfs_fc[fc]

            # Set the names of the fields containing the water body plan IDs and names
            vpID = self.wfs_vpID[fc]

            if not arcpy.Exists(fc):
                # Execute the WFSToFeatureClass tool to download the feature class.
                arcpy.WFSToFeatureClass_conversion(WFS_Service, WFS_FeatureType,
                                                   self.arcPath, fc, max_features=15000)

            # Make a template as a copy of the input to preserve the original
            fcTemplate = fc + 'Template'
            arcpy.CopyFeatures_management(fc, fcTemplate)

            # Create a list of unnecessary fields
            fieldNameList = []
            fieldObjList = arcpy.ListFields(fcTemplate)
            for field in fieldObjList:
                if not field.required:
                    if not field.name == vpID:
                        fieldNameList.append(field.name)

            # Remove unnecessary fields (columns) to reduce the size of the feature class
            arcpy.DeleteField_management(fcTemplate, fieldNameList)

        except:
            # Report severe error messages from Python or ArcPy
            tb = sys.exc_info()[2]  # get traceback object for Python errors
            tbinfo = traceback.format_tb(tb)[0]
            pymsg = 'Python errors while using WFS for {0}:\nTraceback info:\n{1}Error Info:\n{2}'\
                    .format(fc, tbinfo, str(sys.exc_info()[1]))
            arcmsg = 'ArcPy errors while using WFS for {0}:\n{1}'\
                     .format(fc, arcpy.GetMessages(severity=2))
            print(pymsg)            # print Python error message in Python
            print(arcmsg)           # print ArcPy error message in Python
            arcpy.AddError(pymsg)   # return Python error message in ArcGIS
            arcpy.AddError(arcmsg)  # return ArcPy error message in ArcGIS
            sys.exit(1)



    def frame(self, waterbodyType, parameterCol, parameterType, valueCol):
        """ Function to set up a Pandas DataFrame for a given type of water body
        """
        try:
            # Obtain the filenames from the initialization of this class
            filenames = self.data[waterbodyType]

            # Read and concatenate the data for the two time periods
            df1 = pd.read_excel('data\\' + filenames[0]) # 1990-2018
            df2 = pd.read_excel('data\\' + filenames[1]) # 2019-
            df = pd.concat([df1, df2], join="inner", ignore_index=True)

            # Create 'Year' column from the date integer
            df['Year'] = df['Dato'].astype(str).str.slice(0, 4).astype(int)

            # Subset the data to only contain the relevant parameter
            df = df[df[parameterCol].str.contains(parameterType)]

            # Subset the data to only contain the relevant parameter
            df = df[df[parameterCol].str.contains(parameterType)]

            # Subset the data to relevant variables and sort by year
            df = df[['ObservationsStedNr', 'Lokalitetsnavn', 'Year', valueCol,
                     'Xutm_Euref89_Zone32', 'Yutm_Euref89_Zone32']]\
                     .sort_values('Year')

            # Shorten column names
            df.columns = ['Station', 'Location', 'Year', valueCol, 'X', 'Y']

            # Capitalize location names
            df['Location'] = df['Location'].str.upper()

            return df

        except:
            ## Report severe error messages
            tb = sys.exc_info()[2]  # get traceback object for Python errors
            tbinfo = traceback.format_tb(tb)[0]
            msg = 'Could not set up DataFrame for {0}:\nTraceback info:\n{1}Error Info:\n{2}'\
                  .format(waterbodyType, tbinfo, str(sys.exc_info()[1]))
            print(msg)            # print error message in Python
            arcpy.AddError(msg)   # return error message in ArcGIS
            sys.exit(1)



    def longitudinal(self, waterbodyType):
        """ Set up a longitudinal dataframe based on the type of water body

            Streams: For a given year, finds DVFI for a station with multiple
                     observations by taking the median and rounding down.
        """
        try:
            # Set up a Pandas DataFrame for the chosen type of waterbody
            if waterbodyType == 'streams':
                df = self.frame(waterbodyType, 'Indekstype', 'DVFI', 'Indeks')

                # Drop obs with unknown index value and save index as integers
                df = df[df.Indeks != 'U']
                df['Indeks'] = df['Indeks'].astype(int)

            # Set up a longitudinal df with very station and its latest records
            long = df[['Station', 'Location', 'X', 'Y']]\
                     .groupby(['Station'], as_index=False).last()

            # Sorted list of the years covered by the data
            years = list(df.Year.unique())
            years.sort()

            # Add a column for each year
            for i in years:
                # Subset to relevant year
                dfYear = df[df['Year']==i]

                # Drop year and coordinates
                dfYear = dfYear.drop(['Year', 'X', 'Y'], axis=1)

                if waterbodyType == 'streams':
                    # Group multiple obs for a station: Take the median and round down
                    dfYear = dfYear.groupby(['Station']).median().apply(np.floor).astype(int)

                # Merge into longitudinal df
                dfYear.columns = [i]
                long = long.merge(dfYear, how="left", on='Station')

            return long, years

        except:
            ## Report severe error messages
            tb = sys.exc_info()[2]  # get traceback object for Python errors
            tbinfo = traceback.format_tb(tb)[0]
            msg = 'Could not set up DataFrame for {0}:\nTraceback info:\n{1}Error Info:\n{2}'\
                  .format(waterbodyType, tbinfo, str(sys.exc_info()[1]))
            print(msg)            # print error message in Python
            arcpy.AddError(msg)   # return error message in ArcGIS
            sys.exit(1)



    def stations_to_streams(self, waterbodyType, stationID, waterBodyID,
                            waterBodyName, booleanVar):
        """ Streams: Assign monitoring stations to water bodies via linkage table.

            For unmatched stations: Assign to stream within a radius of 15 meters
            where the locaton names match the name of the stream.

            For a given year, finds DVFI for a stream with multiple stations
            by taking the median and rounding down.

            Finally, extends to all streams in the current water body plan and
            adds the length of each stream in km.
        """
        try:
            # Create a longitudinal DataFrame for stations in streams
            long, years = self.longitudinal(waterbodyType)

            # Read the linkage table
            link = pd.read_excel('linkage\\' + self.linkage[waterbodyType])

            df = long.merge(link[[stationID, waterBodyID, booleanVar]], how="left",
                            left_on='Station', right_on=stationID)

            # Drop water bodies that were only present in the draft for VP2
            df = df[df[booleanVar]!=0].drop(columns=[booleanVar])

            # Make station-ID and coordinates into integers
            df[['Station', 'X', 'Y']] = df[['Station', 'X', 'Y']].astype(int)

            # Stations in current water body plan (VP2)
            match = df.dropna(subset=[stationID]).drop(columns=[stationID])

            # Stations not covered by the linkage table
            noMatch = df[df[stationID].isna()].drop(columns=[stationID, waterBodyID])

            # Create a spatial reference object with the same geoprachical coordinate system
            spatialRef = arcpy.SpatialReference("ETRS 1989 UTM Zone 32N")

            # Create a new feature class shapefile (will overwrite if it already exists)
            fcStations = waterbodyType + '_stations'
            arcpy.CreateFeatureclass_management(self.arcPath, fcStations, "POINT",
                                                spatial_reference=spatialRef)

            # Create fields for 'Station' and 'Location'
            arcpy.AddField_management(fcStations, 'Station', "INTEGER")
            arcpy.AddField_management(fcStations, 'Location', "TEXT")

            # Create cursor to insert stations that were not in the linkage table
            try:
                with arcpy.da.InsertCursor(fcStations, ['SHAPE@XY', 'Station', 'Location']) as cursor:
                    # Loop over each station-ID in df:
                    for index, row in noMatch.iterrows():
                        try:
                            # Use cursor to insert new row in feature class
                            cursor.insertRow([(row['X'], row['Y']), row['Station'], row['Location']])
    
                        except:
                            # Report other severe error messages from Python or ArcPy
                            tb = sys.exc_info()[2]  # get traceback object for Python errors
                            tbinfo = traceback.format_tb(tb)[0]
                            print('Python errors while inserting station {0} in {1}:\nTraceback info:{2}\nError Info:\n{3}\n'\
                                  .format(str(row['Station']), fcStations, tbinfo, str(sys.exc_info()[1])))
                            print('ArcPy errors while inserting station {0} in {1}:\n{2}'\
                                  .format(str(row['Station']), fcStations, tbinfo, str(arcpy.GetMessages(severity=2))))
                            sys.exit(1)
    
                        finally:
                            # Clean up for next iteration
                            del index, row

            finally:
                del cursor

            # Refer to the template created with get_fc_from_WFS()
            fcTemplate = waterbodyType + 'Template'

            # Specify joined feature class (will overwrite if it already exists)
            joinedFC = fcStations + '_joined'

            # Spatial Join unmatched stations with streams within 15 meters
            arcpy.SpatialJoin_analysis(fcTemplate, fcStations, joinedFC,
                                       "JOIN_ONE_TO_MANY", "KEEP_COMMON",
                                       match_option="CLOSEST",
                                       search_radius=50,
                                       distance_field_name='Distance')

            # Set the names of the field containing the water body plan IDs
            vpID = self.wfs_vpID[waterbodyType]

            # Fields of interest
            fieldNames = ['Station', 'Distance', 'Location', vpID]

            # Use SeachCursor to read columns to pd.DataFrame
            data = [row for row in arcpy.da.SearchCursor(joinedFC, fieldNames)]
            joined = pd.DataFrame(data, columns=fieldNames)

            # Add water body names from linkage table and sort by distance (ascending)
            j = joined.merge(link[[waterBodyName, waterBodyID]], how="inner",
                             left_on=vpID, right_on=waterBodyID)\
                             .drop([vpID], axis=1)\
                             .sort_values('Distance')

            # Capitalize water body names
            j[waterBodyName] = j[waterBodyName].str.upper()

            # Indicate matching water body names
            j['Match'] = np.select([j['Location']==j[waterBodyName]], [True])

            # Subset to unique stations with their closest matching water body
            jMatches = j[j['Match']==True].groupby(['Station'], as_index=False).first()

            # Inner merge of noMatch and j_matches stations with their closest matching water body
            jMatches = noMatch.merge(jMatches[['Station', waterBodyID]],
                                     how='inner', on='Station')

            # Concatenate the dfs of stations that have been matched to a water body
            allMatches = pd.concat([match, jMatches])

            # Group multiple stations in a water body: Take the median and round down
            waterBodies = allMatches.groupby([waterBodyID]).median().apply(np.floor)\
                                    .drop(columns=['Station', 'X', 'Y'])

            # Field/column names
            ID = self.wfs_vpID[waterbodyType]
            length = self.wfs_size[waterbodyType]

            # Use SeachCursor to create df with length (km) of all water bodies
            dataLength = [row for row in arcpy.da.SearchCursor(waterbodyType, [ID, length])]
            dfLength = pd.DataFrame(dataLength, columns=[ID, length])

            # Merge length-df with df for water bodies in the current water body plan (VP2)
            allVP = dfLength.merge(waterBodies, how="outer", left_on=ID,
                                   right_index=True).set_index(vpID)

            # Save to CSV for later statistical work
            allVP.to_csv('data\\streams_DVFI_longitudinal.csv')

            # Report the number of stations matched by linkage table and ArcPy
            msg = "{0} stations were matched to a water body by the official linkage table. Besides, {1} were located within 15 meters of a water body carrying the name of the station's location."\
                  .format(len(match), len(jMatches))
            print(msg)            # print number of stations in Python
            arcpy.AddMessage(msg) # return number of stations in ArcGIS

            return allVP, years

        except:
            # Report severe error messages
            tb = sys.exc_info()[2]  # get traceback object for Python errors
            tbinfo = traceback.format_tb(tb)[0]
            pymsg = 'Python errors while assigning stations to streams:\nTraceback info:\n{0}Error Info:\n{1}'\
                    .format(tbinfo, str(sys.exc_info()[1]))
            arcmsg = 'ArcPy errors while assigning stations to streams:\n{0}'\
                     .format(arcpy.GetMessages(severity=2))
            print(pymsg)            # print Python error message in Python
            print(arcmsg)           # print ArcPy error message in Python
            arcpy.AddError(pymsg)   # return Python error message in ArcGIS
            arcpy.AddError(arcmsg)  # return ArcPy error message in ArcGIS
            sys.exit(1)



    def DVFI_to_status(self, df, years):
        """ Convert DVFI fauna index for streams to index of ecological status.
        """
        try:
            for i in years:
                # Categorical variable for ecological status: Bad, Poor, Moderate, Good, High
                conditions = [df[i]==1,
                              (df[i]==2) | (df[i]==3),
                              df[i]==4,
                              (df[i]==5) | (df[i]==6),
                              df[i]==7]
                df[i] = np.select(conditions, [1, 2, 3, 4, 5], default=np.nan)

            return df

        except:
            ## Report severe error messages
            tb = sys.exc_info()[2]  # get traceback object for Python errors
            tbinfo = traceback.format_tb(tb)[0]
            msg = 'Could not convert DVFI to ecological status:\nTraceback info:\n{0}Error Info:\n{1}'\
                  .format(tbinfo, str(sys.exc_info()[1]))
            print(msg)            # print error message in Python
            arcpy.AddError(msg)   # return error message in ArcGIS
            sys.exit(1)



    def ecological_status(self, waterbodyType):
        """ Based on the type of water body, set up a longitudinal dataframe
            and convert it to the EU index of ecological status, i.e. from 1-5
            for bad, poor, moderate, good, and high water quality respectively.
            
            Create a table of statistics and export it as an html table.
            
            Print the size and share of water bodies observed at least once.
        """
        try:
            if waterbodyType == 'streams':
                # Create longitudinal df and use linkage table to assign stations to water bodies
                allVP, years = self.stations_to_streams(waterbodyType,
                                                        stationID='DCE_stationsnr',
                                                        waterBodyID='VP2_g_del_cd',
                                                        waterBodyName='Navn',
                                                        booleanVar='VP2_gældende')

                # Convert DVFI fauna index for streams to index of ecological status
                df = self.DVFI_to_status(allVP, years)

            # Save to CSV for later statistical work
            df.to_csv('data\\' + waterbodyType + '_ecological_status.csv')
   
            # Specify name of size-variable
            size = self.wfs_size[waterbodyType]
            
            # Calculate total size of all water bodies in current water body plan (VP2)
            totalSize = df[size].sum()
            
            # Create an empty df for statistics
            stats = pd.DataFrame(index=['Status known (%)',
                                        'Share of known is high (%)',
                                        'Share of known is good (%)',
                                        'Share of known is moderate (%)',
                                        'Share of known is poor (%)',
                                        'Share of known is bad (%)'])
            
            # Calculate the above statistics for each year
            for i in years:
                y = df[[size, i]].reset_index(drop=True)
                y['Known'] = np.select([y[i].notna()], [y[size]])
                y['High'] = np.select([y[i]==5], [y[size]])
                y['Good'] = np.select([y[i]==4], [y[size]])
                y['Moderate'] = np.select([y[i]==3], [y[size]])
                y['Poor'] = np.select([y[i]==2], [y[size]])
                y['Bad'] = np.select([y[i]==1], [y[size]])
                
                # Add shares of total size to stats
                knownSize = y['Known'].sum()
                stats[i] = [100*knownSize/totalSize,
                            100*y['High'].sum()/knownSize,
                            100*y['Good'].sum()/knownSize,
                            100*y['Moderate'].sum()/knownSize,
                            100*y['Poor'].sum()/knownSize,
                            100*y['Bad'].sum()/knownSize]
            
            # Convert statistics to integers
            stats = stats.astype(int)
            
            # Save to html for online presentation
            stats.to_html('data\\' + waterbodyType + '_stats.md')

            # Water bodies observed at least once
            observed = df[[size]].merge(df.drop(columns=[size]).dropna(how="all"),
                                        how="inner", left_index=True, right_index=True)

            # Unit of measurement
            if waterbodyType == 'streams':
                unit = 'km'
            else:
                unit = 'sq. km'
            
            # Report size and share of water bodies observed at least once.
            msg = 'The current water body plan covers {0} {1} of {2}, of which {2} representing {3} {1} ({4}%) have been assessed at least once. On average {2} representing {5} {1} ({6}%) are assessed each year.'\
                  .format(int(totalSize), unit, waterbodyType, 
                          int(observed[size].sum()),
                          int(100*observed[size].sum()/totalSize),
                          int(stats.iloc[0].mean()*totalSize/100),
                          int(stats.iloc[0].mean()))
            print(msg)            # print statistics in Python
            arcpy.AddMessage(msg) # return statistics in ArcGIS
            
            return df, years

        except:
            # Report severe error messages
            tb = sys.exc_info()[2]  # get traceback object for Python errors
            tbinfo = traceback.format_tb(tb)[0]
            msg = 'Could not create df with ecological status for {0}:\nTraceback info:\n{1}Error Info:\n{2}'\
                  .format(waterbodyType, tbinfo, str(sys.exc_info()[1]))
            print(msg)            # print error message in Python
            arcpy.AddError(msg)   # return error message in ArcGIS
            sys.exit(1)


    def map_book(self, fc, df, yearsList, numberOfRetries=0, sleepError=0):
        """ Create a pdf map book with a page for each year
        """
        import time

        try:
            # Set the name of the field (column) containing the water body plan IDs
            vpID = self.wfs_vpID[fc]

            # Refer to the template
            fcTemplate = fc + 'Template'

            # Add an integer field (column) for storing the ecological status
            arcpy.AddField_management(fcTemplate, 'status', 'INTEGER')

            # Create a map book to contain a pdf page for each year
            bookPath = self.path + '\\' + fc + '.pdf'
            if os.path.exists(bookPath):
                os.remove(bookPath)

            # Initiate the map book
            book = arcpy.mp.PDFDocumentCreate(bookPath)

            ##### Make a feature class, layer and pdf map for each year
            for i in yearsList:
                try:
                    errlist = []
                    for iteration in range(1+numberOfRetries):
                        try:
                            # Copy feature class from template
                            fcYear = fc + str(i) + 'fc'
                            if arcpy.Exists(fcYear):
                                arcpy.Delete_management(fcYear)
                            arcpy.CopyFeatures_management(fcTemplate, fcYear)

                            # Create update cursor for feature layer
                            with arcpy.da.UpdateCursor(fcYear, [vpID, 'status']) as cursor:
                                try:
                                    # For each row/object in feature class add ecological status from df
                                    for row in cursor:
                                        if row[0] in df.index:
                                            row[1] = df.loc[row[0], i]
                                            cursor.updateRow(row)
        
                                except:
                                    # Report severe error messages from Python or ArcPy
                                    tb = sys.exc_info()[2]  # get traceback object for Python errors
                                    tbinfo = traceback.format_tb(tb)[0]
                                    pymsg = 'Python errors while updating {0} in {1}:\nTraceback info:\n{2}Error Info:\n{3}'\
                                            .format(row[0], fc, tbinfo, str(sys.exc_info()[1]))
                                    arcmsg = 'ArcPy errors while updating {0} in {1}:\n{2}'\
                                             .format(row[0], fc, arcpy.GetMessages(severity=2))
                                    print(pymsg)            # print Python error message in Python
                                    print(arcmsg)           # print ArcPy error message in Python
                                    arcpy.AddError(pymsg)   # return Python error message in ArcGIS
                                    arcpy.AddError(arcmsg)  # return ArcPy error message in ArcGIS
        
                                finally:
                                    # Clean up for next iteration
                                    del cursor, row
        
                            # Create a feature layer for the feature class (name of layer applies to the legend)
                            layerYear = fc + str(i)
                            arcpy.MakeFeatureLayer_management(fcYear, layerYear)
        
                            # Apply symbology for ecological status from layer
                            arcpy.ApplySymbologyFromLayer_management(layerYear,
                                            self.path + '\\' + fc + '_symbology.lyrx')

                            # Save temporary layer file
                            arcpy.SaveToLayerFile_management(layerYear,
                                                             layerYear + '.lyrx')
        
                            # Reference the temporary layer file
                            lyrFile = arcpy.mp.LayerFile(layerYear + '.lyrx')
        
                            # Reference the ArcGIS Pro project, its map, and the layout to export
                            aprx = arcpy.mp.ArcGISProject(self.path + '\\gis.aprx')
                            m = aprx.listMaps("Map")[0]
                            lyt = aprx.listLayouts("Layout")[0]
        
                            # Add layer to map
                            m.addLayer(lyrFile, "TOP")
        
                            # Export layout to a temporary PDF
                            lyt.exportToPDF('temp.pdf', resolution = 300)
        
                            # Append the page to the map book
                            book.appendPages(self.path + '\\' + 'temp.pdf')
                            
                            # Skip retrying and jump to next year
                            break
        
                        except:
                            # Save severe error messages from Python or ArcPy
                            tb = sys.exc_info()[2]  # get traceback object for Python errors
                            tbinfo = traceback.format_tb(tb)[0]
                            pymsg = 'Python errors while creating a PDF for {0}, {1}:\nTraceback info:\n{2}Error Info:\n{3}'\
                                    .format(fc, i, tbinfo, str(sys.exc_info()[1]))
                            arcmsg = 'ArcPy errors while creating a PDF for {0}, {1}:\n{2}'\
                                    .format(fc, i, arcpy.GetMessages(severity=2))
                            errlist.append(pymsg)   # Save Python error message
                            errlist.append(arcmsg)  # Save ArcPy error message
                            
                            # Sleep before trying again in case of error
                            time.sleep(sleepError)

                        finally:
                            # Clean up after each iteration of loop
                            if arcpy.Exists(fcYear):
                                arcpy.Delete_management(fcYear)
                            if arcpy.Exists(fc + str(i) + '.lyrx'):
                                arcpy.Delete_management(fc + str(i) + '.lyrx')
                            del fcYear, layerYear, lyrFile, aprx, m, lyt

                except:
                    # Report severe error messages from Python or ArcPy
                    for e in errlist:
                        print(e)                # print error messages in Python
                        arcpy.AddError(arcmsg)  # return error messages in ArcGIS
                        sys.exit(1)

            # Commit changes and save the map book
            book.saveAndClose()

        except:
            # Report severe error messages from Python or ArcPy
            tb = sys.exc_info()[2]  # get traceback object for Python errors
            tbinfo = traceback.format_tb(tb)[0]
            pymsg = 'Python errors while creating a map book for {0}:\nTraceback info:\n{1}Error Info:\n{2}'\
                    .format(fc, tbinfo, str(sys.exc_info()[1]))
            arcmsg = 'ArcPy errors while creating a map book for {0}:\n{1}'\
                     .format(fc, arcpy.GetMessages(severity=2))
            print(pymsg)            # print Python error message in Python
            print(arcmsg)           # print ArcPy error message in Python
            arcpy.AddError(pymsg)   # return Python error message in ArcGIS
            arcpy.AddError(arcmsg)  # return ArcPy error message in ArcGIS
            sys.exit(1)

        finally:
            # Clean up
            if arcpy.Exists(fcTemplate):
                arcpy.Delete_management(fcTemplate)
            if os.path.exists('temp.pdf'):
                    os.remove('temp.pdf')
            del book
