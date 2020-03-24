# The Green Net National Product: *Subproject regarding pollution of the water environment*

This subproject is a part of a larger research project on [UCPH](https://www.ku.dk/english/) that aims to estimate the Green Net National Product (GNNP) for Denmark since 1990. The first stage of this subproject constructs longitudinal datasets and maps the ecological status of Danish water bodies.

For more information and preliminary results for streams, see: [ThorNoe.github.io/GNNP](https://thornoe.github.io/GNNP/)


### How to run the script tool in ArcGIS Pro

One will need to have [ArcGIS Pro](https://www.esri.com/en-us/arcgis/products/arcgis-pro/overview) installed and up to date.

1. Click the green `Clone or download` button at the top right corner of this GitHub page ^^
2. Within the `gnnp\gis` folder, open `gis.aprx` with ArcGIS Pro.
3. In the `Catalog Pane`, open `Toolboxes` > `gis.tbx` > `WaterbodiesScriptTool`. Select the `gnnp\gis` folder and `Run` the script.


### Update with the most recent data

The most recent data on indicators can be downloaded from the Danish surface water database by opening [odaforalle.au.dk](https://odaforalle.au.dk/) in Internet Explorer. To access the data, it is required that you supply your e-mail address.

For streams:

- Go to `Hent data` > `Vandløb` > `Bundfauna` > `DVFI-indeks`
  - Set the *from* date: In `Fra` write `01-01-2019`
  - Set the *to* date: In `Til` (optional)
  - In the bottom right corner, click `Excel (<no. rows> rækker)` and save to the `gis/data` folder as `streams_DVFI_2019-.xlsx` *(overwrite the existing file with the exact same name!)*

Run the script tool as described above to update the time series and recreate all the illustrations with the most recent data.

Gitdown table v. 0.2

{"gitdown": "include", "file": "./gis/data/streams_stats.md"}

html:

{"gitdown": "include", "file": "./gis/data/streams_stats.html"}

### Run ArcPy commands in the Anaconda Spyder environment
The version of Python used in ArcGIS is usually older than the one installed in the [Anaconda platform](https://www.anaconda.com/distribution/), making [Spyder](https://www.spyder-ide.org/) incompatible with the ArcPy package as it depends on the ArcGIS installation. Thus, to be able to run ArcPy commands via Spyder, you need to:
1. Click the **Start** icon.
2. Navigate to the **ArcGIS** folder.
3. Click **Python Command Prompt** within which you
   - Add Spyder by typing `conda install spyder`
   - Open Spyder by typing `spyder` and `import arcpy`. You need to open Spyder this way whenever you are to use ArcPy commands.
