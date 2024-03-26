# Green GDP: Pollution of water ecosystem services
This research project estimates the investment in natural capital and cost of pollution of water ecosystem services for the [Green National Accounts](https://susy.ku.dk/research/developing-and-implementing-green-national-accounts-and-the-green-gdp) of Denmark for 1990-2020. 

For a brief overview of the approach and the development in the ecological status of coastal waters, lakes, and streams, see [ThorNoe.GitHub.io/GreenGDP](https://thornoe.github.io/GreenGDP).

## Output

Main results include:
- **Cost of pollution**: [Figure](https://github.com/thornoe/GreenGDP/blob/master/gis/output/all_cost.pdf) and [table](https://github.com/thornoe/GreenGDP/blob/master/gis/output/all_cost.csv).
- **Investment in water quality improvement**: [Figure](https://github.com/thornoe/GreenGDP/blob/master/gis/output/all_investment.pdf) and [table](https://github.com/thornoe/GreenGDP/blob/master/gis/output/all_investment.csv).

Observed and imputed ecological status are displayed at [ThorNoe.GitHub.io/GreenGDP](https://thornoe.github.io/GreenGDP), based on:
- [Table](https://github.com/thornoe/GreenGDP/raw/master/gis/output/all_eco_LessThanGood.xlsx) of **ecological status** as observed, imputed, and imputed with moving average respectively. 
- [Table](https://github.com/thornoe/GreenGDP/blob/master/gis/output/all_eco_imp.csv) for each of the **108 catchment areas** where mean ecological status by category (coastal waters, lakes, and streams respectively) is weighted by shore length.
- **Longitudinal tables** by waterbody for:
  - **Observed biological indicators** of [coastal waters](https://github.com/thornoe/GreenGDP/blob/master/gis/output/coastal_ind_obs.csv), [lakes](https://github.com/thornoe/GreenGDP/blob/master/gis/output/lakes_ind_obs.csv), and [streams](https://github.com/thornoe/GreenGDP/blob/master/gis/output/streams_ind_obs.csv) respectively.
  - **Observed ecological status** of [coastal waters](https://github.com/thornoe/GreenGDP/blob/master/gis/output/coastal_eco_obs.csv), [lakes](https://github.com/thornoe/GreenGDP/blob/master/gis/output/lakes_eco_obs.csv), and [streams](https://github.com/thornoe/GreenGDP/blob/master/gis/output/streams_eco_obs.csv) respectively.
  - **Imputed ecological status** of [coastal waters](https://github.com/thornoe/GreenGDP/blob/master/gis/output/coastal_eco_imp.csv), [lakes](https://github.com/thornoe/GreenGDP/blob/master/gis/output/lakes_eco_imp.csv), and [streams](https://github.com/thornoe/GreenGDP/blob/master/gis/output/streams_eco_imp.csv) respectively.
  - **Imputed ecological status using a 5-year moving average** of [coastal waters](https://github.com/thornoe/GreenGDP/blob/master/gis/output/coastal_eco_imp_MA.csv), [lakes](https://github.com/thornoe/GreenGDP/blob/master/gis/output/lakes_eco_imp_MA.csv), and [streams](https://github.com/thornoe/GreenGDP/blob/master/gis/output/streams_eco_imp_MA.csv) respectively.

## How to run the script tool in ArcGIS Pro
All figures and tables can be reproduced by running the **script tool** or the underlying **Python** [script](https://github.com/thornoe/GreenGDP/blob/master/gis/script.py) using the Python class in the [script module](https://github.com/thornoe/GreenGDP/blob/master/gis/script_module.py) (based on [ArcPy](https://developers.arcgis.com/documentation/arcgis-add-ins-and-automation/arcpy), an [ArcGIS Pro](https://www.esri.com/en-us/arcgis/products/arcgis-pro/overview) license is as prerequisite).

Using [ArcGIS Pro 3.2](https://pro.arcgis.com/en/pro-app/3.2/get-started/get-started.htm), you can run the script tool as follows:
1. Click the green `Code` button at the top right corner of this GitHub page ^^ to download the repository.
2. Within the downloaded `GreenGDP` (or `GreenGDP-master`) folder, go to the `gis` folder and open the ArcGIS Project File `gis.aprx` with ArcGIS Pro.
3. In the `Catalog Pane`, open `Toolboxes` > `gis.tbx` > `WaterbodiesScriptTool`. Select the `gis` folder and `Run` the script

### Public data for surface water
The most recent data on biological indicators of ecological status can be downloaded from the Danish surface water database [odaforalle.au.dk](https://odaforalle.au.dk). To access the data, you are required to submit your e-mail address.

For **coastal waters** (or **lakes**):
- Go to `Hent data` > `Hav` (or `Sø` for lakes).
  - Select `Vandkemi` > `Næringsstof og Miljøfarligt stof` (or `Næringsstof m.m.` for lakes).
  - Set the *from* date: In `Fra` write `01-01-1987`.
  - Set the *to* date: In `Til` (optional).
  - In `Parameter` select `Chlorophyl (ukorrigeret)` and `Chlorophyl A`.
  - In the bottom right corner, click `Excel (<no. rows> rækker)` and save to the `gis/data` folder as `coastal_chlorophyll.xlsx` (or `lakes_chlorophyll.xlsx`) *(overwrite the existing files)*.

For **streams**:
- Go to `Hent data` > `Vandløb` > `Bundfauna` > `DVFI-indeks`.
  - Set the *from* date: In `Fra` write `01-01-2021` ([historical data](https://github.com/thornoe/GreenGDP/blob/master/gis/data/streams_1987-2020.xlsx) is no longer at the site).
  - Set the *to* date: In `Til` (optional).
  - In the bottom right corner, click `Excel (<no. rows> rækker)` and save to the `gis/data` folder as `streams_DVFI.xlsx` *(overwrite the existing file)*.

To extend the green national account beyond 2020, specify a new `year_last` when you run the script tool (as described above) to update the figures and tables with more recent data.

### Update with new identification of waterbodies after 2027
When a fourth waterbody plan (VP4) is passed to cover the period after 2027, an updated [MiljøGIS](https://mst.dk/service/miljoegis) will be published and the Danish Environmental Protection Agency ([Miljøstyrelsen](https://mst.dk/)) will produce updated linkage tables.

If interested in updating the script for VP4 by then, update the specifications in [script.py](https://github.com/thornoe/GreenGDP/blob/master/gis/script.py) with the new `year_last` and file names of the new `linkage` tables as well as the **WFS** specifications for the updated MiljøGIS map (first, open it with the ArcGIS Pro Geoprocessing tool `WFS To Feature Class`).

### Run ArcPy commands in the PyScripter editor
For a mayor revision of the script tool (e.g., updating to VP4), one will want to be able to run ArcPy commands within an IDE or text editor.

The version of Python used in ArcGIS Pro is systematically older than the most recent version, which causes several incompatibility issues for Python scripts. Thus, the simplest IDE solution is to [download](https://sourceforge.net/projects/pyscripter) and set up the [PyScripter](https://github.com/pyscripter/pyscripter/wiki) editor as explained [here](https://www.e-education.psu.edu/geog485/node/213).
1. Within **ArcGIS Pro**, navigate to the **Package Manager**:
   1. In the **Environment Manager**, *Clone* the default Python environment and *Activate* arcgispro-py3-clone as your new environment.
   2. Under **Add Packages**, *Search* for and *Install* [scikit-learn](https://scikit-learn.org/stable/index.html) for imputation of missing observations.
   3. Under **Updates**, remember to *Update All* each time you update ArcGIS Pro to a new version.
2. Within **PyScripter**:
   1. Under **Python Versions**, *Add* and *Activate* the cloned environment, e.g. `C:\Users\%USERNAME%\AppData\Local\ESRI\conda\envs\arcgispro-py3-clone`.
   2. Under **Tools** > **Options** > **IDE Shortcuts**, you can *Assign* the command **Run: Execute selection** to `Shift+Enter` (or whatever you prefer) instead of `Ctrl+F7`.

## License
This project is released under the [MIT License](https://github.com/thornoe/GreenGDP/blob/master/LICENSE), that is, you can basically do anything with my code as long as you give appropriate credit and don’t hold me liable.
