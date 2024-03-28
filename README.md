# Green GDP: Pollution of water ecosystem services
This research project estimates the investment in natural capital and cost of pollution of water ecosystem services for the [Green National Accounts](https://susy.ku.dk/research/developing-and-implementing-green-national-accounts-and-the-green-gdp) of Denmark for 1990-2020. 

For a brief overview of the approach and the development in the ecological status of coastal waters, lakes, and streams, see [ThorNoe.GitHub.io/GreenGDP](https://thornoe.github.io/GreenGDP).

## Using Byn Lake as an example
The marginal willingness to pay for improving the ecological status of Byn Lake from Moderate to Good is estimated to be 92 DKK per household per year, amounting to a total cost of 238,997 DKK per year, using 2018 prices and demographics for the 2,575 households in the catchment area of outer Nissum Fjord.

Thus, the investment value of improving the ecological status of Byn Lake from Moderate to Good is estimated to be 22,751 DKK per household, amounting to a total investment value of 58,586,158 DKK, using 2018 prices and demographics for the 2,575 households in the catchment area of outer Nissum Fjord and the net present value implied by the declining discount rate prescribed by the Ministry of Finance ([2021](https://fm.dk/media/18371/dokumentationsnotat-for-den-samfundsoekonomiske-diskonteringsrente_7-januar-2021.pdf)) during 2014-2020.

This value does not capture the willingness to pay for reestablishing a habitat for endangered species. Arguably, households from all parts of Denmark would assign a significant existence and bequest value to preventing the extinction of native species.

## Code for Byn Lake
[Byn_Lake_variables.csv](https://github.com/thornoe/GreenGDP/blob/Byn-Lake/gis/output/Byn_Lake_variables.csv) is created by running [Byn_Lake_data.py](https://github.com/thornoe/GreenGDP/blob/Byn-Lake/gis/Byn_Lake_data.py), using [ArcPy](https://developers.arcgis.com/documentation/arcgis-add-ins-and-automation/arcpy) functions in [script_module.py](https://github.com/thornoe/GreenGDP/blob/Byn-Lake/gis/script_module.py), which require an [ArcGIS Pro](https://www.esri.com/en-us/arcgis/products/arcgis-pro/overview) license (as explained in the [master-branch](https://github.com/ThorNoe/GreenGDP)).

However, [Byn_Lake_valuation.py](https://github.com/thornoe/GreenGDP/blob/Byn-Lake/gis/Byn_Lake_valuation.py) can be run and modifier without prerequisites. If extrapolating the demographics in [Byn_Lake_variables.csv](https://github.com/thornoe/GreenGDP/blob/Byn-Lake/gis/output/Byn_Lake_variables.csv) beyond 2018, note the implied increase in uncertainty, since the special delivery from Statistics Denmark (using a 100m × 100m grid) only covered 1990-2018. Likewise, the nonlinear benefit transfer function ([Zandersen et al, 2022](https://dce.au.dk/udgivelser/vr/nr-451-500/abstracts/no-486-socio-economic-benefits-of-improved-water-quality-development-and-use-of-meta-analysis-function-for-benefit-transfer)) was estimated using DKK in 2018-prices.

## License
This project is released under the [MIT License](https://github.com/thornoe/GreenGDP/blob/master/LICENSE), that is, you can basically do anything with my code as long as you give appropriate credit and don’t hold me liable.
