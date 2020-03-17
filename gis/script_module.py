import numpy as np
import pandas as pd
# import ArcPy
import sys, traceback

class dataClass:
    """ Class for all data processing and mapping functions
    """
    def __init__(self, dataFilenames, linkageFilenames):
        self.data = dataFilenames
        self.linkage = linkageFilenames


    def get_data(self):
        """ Function to check that the folders and their files exist.
            Otherwise creates the folder and downloads the files from GitHub.
        """
        import os, urllib

        try:
            ### Dictionary for all data and linkage files
            allFiles = {'data': [b for a in list(self.data.values()) for b in a],
                        'linkage': [a for a in list(self.linkage.values())]}
        except:
            ## Report severe error messages
            tb = sys.exc_info()[2]  # get traceback object for Python errors
            tbinfo = traceback.format_tb(tb)[0]
            msg = 'Could not set up dictionary with all files:\nTraceback info:\n{0}Error Info:\n{1}'\
                  .format(tbinfo, str(sys.exc_info()[1]))
            print(msg)            # print error message in Python
            # arcpy.AddError(msg)   # return error message in ArcGIS
            # sys.exit(1)

        for key, filenames in allFiles.items():
            try:
                ### Create the folder if it doesn't exist
                parentFolder = os.getcwd()
                newPath = parentFolder + '\\' + key
                os.makedirs(newPath, exist_ok=True)
                os.chdir(newPath)

                for f in filenames:
                    ### Download the files if they don't exist
                    if not os.path.exists(f):
                        try:
                            url = 'https://github.com/thornoe/GNNP/raw/master/gis/' + key + '/' + f
                            urllib.request.urlretrieve(url, f)
                        except urllib.error.URLError as e:
                            ## Report URL error messages
                            urlmsg = 'URL error for {0}:\n{1}'.format(f, e.reason)
                            print(urlmsg)            # print URL error message in Python
                            # arcpy.AddError(urlmsg)   # return URL error message in ArcGIS
                            # sys.exit(1)
                        except:
                            ## Report other severe error messages
                            tb = sys.exc_info()[2]  # get traceback object for Python errors
                            tbinfo = traceback.format_tb(tb)[0]
                            msg = 'Could not download {0}:\nTraceback info:\n{1}Error Info:\n{2}'\
                                  .format(f, tbinfo, str(sys.exc_info()[1]))
                            print(msg)            # print error message in Python
                            # arcpy.AddError(msg)   # return error message in ArcGIS
                            # sys.exit(1)

            except OSError as e:
                ## Report system errors
                tb = sys.exc_info()[2]  # get traceback object for Python errors
                tbinfo = traceback.format_tb(tb)[0]
                OSmsg = 'System error for {0} folder:\nTraceback info:\n{1}Error Info:\n{2}'\
                        .format(folderName, tbinfo, str(sys.exc_info()[1]))
                print(OSmsg)            # print system error message in Python
                # arcpy.AddError(OSmsg)   # return system error message in ArcGIS
                # sys.exit(1)

            finally:
                ## Change the directory back to the original working folder
                os.chdir(parentFolder)


    def frame(self, waterbodyType, parameterCol, parameterType, valueCol):
        """ Function to set up a Pandas DataFrame for a given type of water body
        """
        try:
            ### Obtain the filenames from the initialization of this class
            filenames = self.data[waterbodyType]

            ### Read and concatenate the data for the two time periods
            df1 = pd.read_excel('data\\' + filenames[0]) # 1990-2018
            df2 = pd.read_excel('data\\' + filenames[1]) # 2019-
            df1 = pd.concat([df1, df2], join="inner", ignore_index=True)

            ### Create 'Year' column from the date integer
            df1['Year'] = df1['Dato'].astype(str).str.slice(0, 4).astype(int)

            ### Subset the data to only contain the relevant parameter
            df1 = df1[df1[parameterCol].str.contains(parameterType)]

            ### Subset the data to only contain the relevant parameter
            df1 = df1[df1[parameterCol].str.contains(parameterType)]

            ### Subset the data to only contain the relevant parameter
            df1 = df1[['ObservationsStedNr', 'Year', valueCol, 'Xutm_Euref89_Zone32',
                       'Yutm_Euref89_Zone32']]

            ### Shorten column names
            df1.columns = ['Station', 'Year', valueCol, 'X', 'Y']

            return df1

        except:
            ## Report severe error messages
            tb = sys.exc_info()[2]  # get traceback object for Python errors
            tbinfo = traceback.format_tb(tb)[0]
            msg = 'Could not set up DataFrame for {0}:\nTraceback info:\n{1}Error Info:\n{2}'\
                  .format(waterbodyType, tbinfo, str(sys.exc_info()[1]))
            print(msg)            # print error message in Python
            # arcpy.AddError(msg)   # return error message in ArcGIS
            # sys.exit(1)


    def longitudinal(self, waterbodyType):
        """ Set up a longitudinal dataframe based on the type of water body
        """
        try:
            ### Set up a Pandas DataFrame for the chosen type of waterbody
            if waterbodyType == 'streams':
                df = self.frame(waterbodyType, 'Indekstype', 'DVFI', 'Indeks')

                ### Drop obs with unknown index value and turn the df into integers
                df = df[df.Indeks != 'U'].astype(int)

            ### Set up a longitudinal df with all the unique stations and their coordinates
            long = df[['Station', 'X', 'Y']].groupby(['Station']).median().astype(int)

            ### List the years covered by the data
            yearsList = list(df.Year.unique())
            yearsList.sort()

            ### Add a column for each year
            for i in yearsList:
                ### Subset to relevant year
                dfYear = df[df['Year']==i]

                ### Drop year and coordinates
                dfYear = dfYear.drop(['Year', 'X', 'Y'], axis=1)

                if waterbodyType == 'streams':
                    ### Group multiple obs for a station by taking the median and round down
                    dfYear = dfYear.groupby(['Station']).median().apply(np.floor).astype(int)

                ### Merge into longitudinal df
                dfYear.columns = [i]
                long = long.merge(dfYear, how='left', on='Station')

            return long, yearsList

        except:
            ## Report severe error messages
            tb = sys.exc_info()[2]  # get traceback object for Python errors
            tbinfo = traceback.format_tb(tb)[0]
            msg = 'Could not set up DataFrame for {0}:\nTraceback info:\n{1}Error Info:\n{2}'\
                  .format(waterbodyType, tbinfo, str(sys.exc_info()[1]))
            print(msg)            # print error message in Python
            # arcpy.AddError(msg)   # return error message in ArcGIS
            # sys.exit(1)
