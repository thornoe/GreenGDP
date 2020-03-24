# The Green Net National Product: *Subproject regarding pollution of the water environment*

This subproject is a part of a larger research project on [UCPH](https://www.ku.dk/english/) that aims to estimate the Green Net National Product (GNNP) for Denmark since 1990. The first stage of this subproject constructs longitudinal datasets and maps the ecological status of Danish water bodies.

For more information and preliminary results for streams, see: [ThorNoe.github.io/GNNP](https://thornoe.github.io/GNNP/)


### How to run the script tool in ArcGIS Pro

[ArcGIS Pro](https://www.esri.com/en-us/arcgis/products/arcgis-pro/overview) must be installed and up to date.

1. Click the green `Clone or download` button at the top right corner of this GitHub page ^^
2. Within the downloaded `gnnp` folder, go to the `gis` folder and open `waterbodies.aprx` with ArcGIS Pro.
3. In the `Catalog Pane`, open `Toolboxes` > `waterbodies.tbx` > `WaterbodiesScriptTool`. Select the `gis` folder and `Run` the script.

### Output

Besides the map book [streams.pdf], the script tool creates two longitudinal datasets for streams. [streams_DVFI_longitudinal.csv] is for DVFI index and [streams_DVFI_longitudinal.csv] is for the ecological status. They should be opened with `pandas.read_csv()` in Python rather than with Excel.


### Update with the most recent data

The most recent data on indicators can be downloaded from the Danish surface water database by opening [odaforalle.au.dk](https://odaforalle.au.dk/) in Internet Explorer. To access the data, it is required that you supply your e-mail address.

For streams:

- Go to `Hent data` > `Vandløb` > `Bundfauna` > `DVFI-indeks`
  - Set the *from* date: In `Fra` write `01-01-2019`
  - Set the *to* date: In `Til` (optional)
  - In the bottom right corner, click `Excel (<no. rows> rækker)` and save to the `gis/data` folder as `streams_DVFI_2019-.xlsx` *(overwrite the existing file with the exact same name!)*

Run the script tool as described above to update the longitudinal datasets and recreate all the illustrations with the most recent data.


### Run ArcPy commands in the Anaconda Spyder environment
The version of Python used in ArcGIS Pro is usually older than the one installed in the [Anaconda platform](https://www.anaconda.com/distribution/), making [Spyder](https://www.spyder-ide.org/) incompatible with the ArcPy package which depends on the ArcGIS Pro installation. Thus, to be able to run ArcPy commands within the Spyder editor, you need to:
1. Click the **Start** icon.
2. Navigate to the **ArcGIS** folder.
3. Click **Python Command Prompt** within which you
   - Add Spyder by typing `conda install spyder`
   - Open Spyder by typing `spyder` and `import arcpy`. You need to open Spyder this way whenever you are to use ArcPy commands.
