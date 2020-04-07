# The Green Net National Product: *Subproject regarding pollution of the water environment*

This subproject is a part of a larger research project on [UCPH](https://www.ku.dk/english/) that aims to estimate the Green Net National Product (GNNP) for Denmark since 1990. The first stage of this subproject constructs longitudinal datasets and maps the ecological status of Danish water bodies.

For more information and preliminary results for streams, see: [ThorNoe.GitHub.io/GNNP](https://thornoe.github.io/GNNP/). Similar work for lakes and coastal waters is forthcoming.


### How to run the script tool in ArcGIS Pro

[ArcGIS Pro](https://www.esri.com/en-us/arcgis/products/arcgis-pro/overview) must be installed and up to date.

1. Click the green `Clone or download` button at the top right corner of this GitHub page ^^
2. Within the downloaded `gnnp` folder, go to the `gis` folder and open the ArcGIS Project File `gis.aprx` with ArcGIS Pro.
3. In the `Catalog Pane`, open `Toolboxes` > `gis.tbx` > `WaterbodiesScriptTool`. Select the `gis` folder and `Run` the script.


### Output

Besides the [map book](https://github.com/thornoe/GNNP/raw/master/gis/output/streams.pdf) with a map for each year, the script tool creates two longitudinal datasets for further statistical work:
- [streams_DVFI_longitudinal.csv](https://github.com/thornoe/GNNP/raw/master/gis/output/streams_DVFI_longitudinal.csv) contains the DVFI values of all streams
- [streams_ecological_status.csv](https://github.com/thornoe/GNNP/raw/master/gis/output/streams_ecological_status.csv) is for the ecological status of all streams


### Update with the most recent data

The most recent data on indicators can be downloaded from the Danish surface water database by opening [odaforalle.au.dk](https://odaforalle.au.dk/) in Internet Explorer. To access the data, it is required that you supply your e-mail address.

For streams:

- Go to `Hent data` > `Vandløb` > `Bundfauna` > `DVFI-indeks`
  - Set the *from* date: In `Fra` write `01-01-2019`
  - Set the *to* date: In `Til` (optional)
  - In the bottom right corner, click `Excel (<no. rows> rækker)` and save to the `gis/data` folder as `streams_DVFI_2019-.xlsx` *(overwrite the existing file with the exact same name!)*

Run the script tool as described above to update the longitudinal datasets and recreate the map book with the most recent data.


### Update with new identification of water bodies in 2022

VP3, the third waterbody plan covering 2021-2027 should be passed by ultimo 2021. Thereafter, an updated [MiljøGIS](https://mst.dk/service/miljoegis/) will be published and the Danish Environmental Protection Agency will produce updated linkage tables to report the new identification of water bodies to the EU EPA.

If interested in updating the script for VP3 by then, update the specifications in `script.py` with names of the new linkage tables as well as the specifications for the updated MiljøGIS map (first, open it with the ArcGIS Pro Geoprocessing tool `WFS To Feature Class`). In `script_module.py`, edit the function `ecological_status()` where it specifies the parameters for calling `self.stations_to_streams()`. If there is no boolean variable in the new linkage table, modify the function `stations_to_streams()` accordingly.
