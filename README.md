# gnnp




### Make the Anaconda Spyder environment compatible with the ArcPy package
The version of Python used in ArcGIS is usually older than the one installed in the [Anaconda platform](https://www.anaconda.com/distribution/), making [Spyder](https://www.spyder-ide.org/) incompatible with the ArcPy package as it depends on the ArcGIS installation. Thus, you need to:
1. [Clone the Python environment](https://support.esri.com/en/technical-article/000020560) and make it the default for the ArcGIS Pro application and the *Python Command Prompt* by typing `proswap <new enviroment name>`.
2. Within the *Python Command Prompt*, install Python and Spyder `conda install python=<version> spyder` based on the [version of Python used in ArcGIS](https://support.esri.com/en/technical-article/000013224), e.g. for ArcGIS 2.5.0: `conda install python=3.6.9 spyder`
3. To use ArcPy commands, open Spyder by typing `spyder` within the *Python Command Prompt* and `import arcpy`.
