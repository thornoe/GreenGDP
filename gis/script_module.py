

def get_data(folderName, fileNames):
    """ Function to check a folder and its files exist.
        Otherwise creates the folder and downloads the files from GitHub.
        Parameters:
        folderName : The name of the subfolder which should be nested in the working folder.
        fileNames : The list of full filenames in the subfolder including the file formats.
    """
    import os, urllib, sys, traceback

    ### Create the folder if it doesn't exist
    try:
        parentFolder = os.getcwd()
        newPath = parentFolder + '\\' + folderName
        os.makedirs(newPath, exist_ok=True)
        os.chdir(newPath)

        for f in fileNames:
            ### Download the files if they don't exist
            if not os.path.exists(f):
                try:
                    url = 'https://github.com/thornoe/GNNP/raw/master/gis/' + folderName + '/' + f
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
                    msg = 'Could not download {0}:\nTraceback info:\n{1}Error Info:\n{2}'.format(f, tbinfo, str(sys.exc_info()[1]))
                    print(msg)            # print error message in Python
                    # arcpy.AddError(msg)   # return error message in ArcGIS
                    # sys.exit(1)

    except OSError as e:
        ## Report system errors
        tb = sys.exc_info()[2]  # get traceback object for Python errors
        tbinfo = traceback.format_tb(tb)[0]
        OSmsg = 'System error for {0} folder:\nTraceback info:\n{1}Error Info:\n{2}'.format(folderName, tbinfo, str(sys.exc_info()[1]))
        print(OSmsg)            # print system error message in Python
        # arcpy.AddError(OSmsg)   # return system error message in ArcGIS
        # sys.exit(1)

    finally:
        ## Change the directory back to the original working folder
        os.chdir(parentFolder)
