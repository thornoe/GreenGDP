# Green GDP: Pollution of water ecosystem services
This research project estimates the investment in natural capital and cost of pollution of water ecosystem services for the [Green National Accounts](https://susy.ku.dk/research/developing-and-implementing-green-national-accounts-and-the-green-gdp) of Denmark for 1990-2020. 

For more information, see presentation of the ecological status of streams at [ThorNoe.GitHub.io/GreenGDP](https://thornoe.github.io/GreenGDP).

## Output
Besides the [map book](https://github.com/thornoe/GreenGDP/raw/master/gis/output/streams.pdf) with a map for each year, the script tool creates two longitudinal datasets for further statistical work:
- [streams_DVFI_longitudinal.csv](https://github.com/thornoe/GreenGDP/raw/master/gis/output/streams_DVFI_longitudinal.csv) contains the DVFI values of all streams
- [streams_ecological_status.csv](https://github.com/thornoe/GreenGDP/raw/master/gis/output/streams_ecological_status.csv) is for the ecological status of all streams

## How to run the script tool in ArcGIS Pro
[ArcGIS Pro](https://www.esri.com/en-us/arcgis/products/arcgis-pro/overview) must be installed and up to date.
1. Click the green `Clone or download` button at the top right corner of this GitHub page ^^
2. Within the downloaded `gnni` folder, go to the `gis` folder and open the ArcGIS Project File `gis.aprx` with ArcGIS Pro
3. In the `Catalog Pane`, open `Toolboxes` > `gis.tbx` > `WaterbodiesScriptTool`. Select the `gis` folder and `Run` the script

### Public data for surface water
The most recent data on indicators can be downloaded from the Danish surface water database by opening [odaforalle.au.dk](https://odaforalle.au.dk/) in Microsoft Edge. To access the data, it is required that you supply your e-mail address.

For streams:
- Go to `Hent data` > `Vandløb` > `Bundfauna` > `DVFI-indeks`
  - Set the *from* date: In `Fra` write `01-01-1989`
  - Set the *to* date: In `Til` (optional)
  - In the bottom right corner, click `Excel (<no. rows> rækker)` and save to the `gis/data` folder as `streams_DVFI.xlsx` *(overwrite the existing file)*

Run the script tool as described above to update the longitudinal datasets and recreate the map book with the most recent data.

### Update with new identification of water bodies in 2024
VP3, the third waterbody plan covering 2021-2027 should be passed by ultimo 2021. Thereafter, an updated [MiljøGIS](https://mst.dk/service/miljoegis/) will be published and the Danish Environmental Protection Agency will produce updated linkage tables to report the new identification of water bodies to the EU EPA.

If interested in updating the script for VP3 by then, update the specifications in [script.py](https://github.com/thornoe/GreenGDP/blob/master/gis/script.py) with names of the new linkage tables as well as the specifications for the updated MiljøGIS map (first, open it with the ArcGIS Pro Geoprocessing tool `WFS To Feature Class`). In [script_module.py](https://github.com/thornoe/GreenGDP/blob/master/gis/script_module.py), edit the function `ecological_status()` where it specifies the parameters for calling `self.stations_to_streams()`. If there is no boolean variable in the new linkage table, modify the function `stations_to_streams()` accordingly.

### Run ArcPy commands in the Anaconda Spyder environment
For a mayor revision of the script tool, one will want to be able to run ArcPy commands within an IDE or text editor.

The version of Python used in ArcGIS Pro is systematically older than the most recent version, which causes several incompatibility issues for Python scripts. Thus, the simplest IDE solution is to [download](https://sourceforge.net/projects/pyscripter) and set up the [PyScripter](https://github.com/pyscripter/pyscripter/wiki) editor as explained [here](https://www.e-education.psu.edu/geog485/node/213).
1. Within **ArcGIS Pro**, navigate to the **Package Manager**:
   1. In the **Environment Manager**, *Clone* the default Python environment and *Activate* arcgispro-py3-clone as your new environment.
   2. Under **Add Packages**, *Search* for and *Install* [scikit-learn](https://scikit-learn.org/stable/index.html) for imputation of missing observations and [xlsxwriter](https://xlsxwriter.readthedocs.io/index.html) to set up results in Excel.
   3. Under **Updates**, remember to *Update All* each time you update ArcGIS Pro to a new version.
2. Within **PyScripter**:
   1. Under **Python Versions**, *Add* and *Activate* the cloned environment, e.g. `C:\Users\%USERNAME%\AppData\Local\ESRI\conda\envs\arcgispro-py3-clone`.
   2. Under **Tools** > **Options** > **IDE Shortcuts**, you can *Assign* the command **Run: Execute selection** to `Shift+Enter` (or whatever you prefer) instead of `Ctrl+F7`.

## License
This project is released under the [MIT License](https://github.com/thornoe/GreenGDP/blob/master/LICENSE), that is, you can basically do anything with my code as long as you give appropriate credit and don’t hold me liable.
