# The Green Net National Product: *Subproject regarding pollution of the water environment*

This subproject is a part of a larger research project on [UCPH](https://www.ku.dk/english/) that aims to estimate the Green Net National Product (GNNP) for Denmark since 1990.

The first stage of this subproject is to map the so-called ecological water quality of Danish water bodies since 1990.

For more information and preliminary results, see: [thornoe.github.io/GNNP](https://thornoe.github.io/GNNP/)


### How to run the script tool in ArcGIS Pro

1. Click the green `Clone or download` button at the top right corner of this GitHub page ^^
2. Within the `gis` folder, open `gis.aprx` with ArcGIS Pro.
3. Open the script tool `Process and map data` under the `Catalog Pane` > `Folders` > `gis` > `gis.tbx`


### Update with the most recent data
The most recent data on indicators can be downloaded from the Danish surface water database by opening [odaforalle.au.dk](https://odaforalle.au.dk/) in Internet Explorer. To access the data, it is only required to supply your e-mail address.

For streams:

- Go to `Hent data` > `Vandløb` > `Bundfauna` > `DVFI-indeks`
  - In `Fra` write `01-01-2019`
  - Under `Data` choose `Fravælg alle` (optional as the scipt can also handle excessive columns)
  - In the bottom right corner, click `Excel (<no. rows> rækker)` and save to the `gis/data` folder as `streams_DVFI_2019-.xlsx` (overwrite the existing file)

Run the script tool as described above to update the time series and recreate all the illustrations with the most recent data.


### Run ArcPy commands in the Anaconda Spyder environment
The version of Python used in ArcGIS is usually older than the one installed in the [Anaconda platform](https://www.anaconda.com/distribution/), making [Spyder](https://www.spyder-ide.org/) incompatible with the ArcPy package as it depends on the ArcGIS installation. Thus, to be able to run ArcPy commands via Spyder, you need to:
1. [Clone the Python environment](https://support.esri.com/en/technical-article/000020560) and make it the default for the ArcGIS Pro application and the *Python Command Prompt* by typing `proswap <new enviroment name>`.
2. Within the *Python Command Prompt*, install Python and Spyder `conda install python=<version> spyder` based on the [version of Python used in ArcGIS](https://support.esri.com/en/technical-article/000013224), e.g. for ArcGIS 2.5.0: `conda install python=3.6.9 spyder`
3. To use ArcPy commands, open Spyder by typing `spyder` within the *Python Command Prompt* and `import arcpy`.
