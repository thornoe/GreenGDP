# Ecological status of surface waters
The first part of this research project is to construct a longitudinal dataset of ecological status for each Danish stream, lake and coastal water from 1990-2020. The data processing is outlined below. Ecological status is observed through biological indicators of eutrophication, which is the process of nutrient over-enrichment of waterbodies due to organic pollution from agriculture and point sources. Such pollution can lead to high growth of phytoplankton and thus oxygen depletion which leads to worse ecological conditions and loss of ecosystem services.

The second part consists of imputing missing values in the dataset. The imputed ecological status is presented in the first figure below.

The third part applies valuation studies to be able to estimate the development in the Green Net National Income (GNNI) for Denmark. For surface water, we apply the Benefit Transfer equation estimated by a meta analysis function ([Zandersen et al, 2022](https://dce.au.dk/udgivelser/vr/nr-451-500/abstracts/no-486-socio-economic-benefits-of-improved-water-quality-development-and-use-of-meta-analysis-function-for-benefit-transfer)) to help answer the overall research question, to what extent economic growth can be considered sustainable or at the cost of the environment?

## Imputed ecological status

#### Surface water: Less than good ecological status (share of shore length)
![Ecological status](https://github.com/thornoe/GreenGDP/blob/master/gis/output/all_eco_imp_LessThanGood.png)

## Identification of waterbodies
This project follows the selection and demarcation given by **VP2** *([Vandområdeplan 2 - Danish](https://mst.dk/natur-vand/vandmiljoe/vandomraadeplaner/vandomraadeplaner-2015-2021/))* which is the Danish waterbody plan for 2015-2021 that implements phase two of the **EU-WFD** ([the European Union's Water Framework Directive - *English*](https://ec.europa.eu/environment/water/water-framework/)). The surface waterbodies in VP2 are constituted by a total of 6979 streams, 857 lakes and 119 coastal waters.


### Assigning monitoring stations to waterbodies
Geographically, monitoring stations are recorded as point observations, streams are identified as vectors, while lakes and coastal waters constitute polygons. Official linkage tables link all waterbodies in VP2 with stations that have been monitored at some point after 2007. [MiljøGIS (Danish)](http://miljoegis.mim.dk/spatialmap?profile=vandrammedirektiv2-bek-2019) interactively shows the demarcation and the result of the basis analysis for VP2 which was carried out during 2008-2012 for streams or during 2007-2013 for lakes and coastal waters. It also encompassed several new indicators in a very thorough assessment of both ecological and chemical status.

For **streams**, 11,289 stations were matched to a waterbody by the official linkage table.


## Ecological status of streams by year
35,867 km is the he total shore length of streams included in VP2, of which 76% of streams representing 31,204 km (87% of total shore length) have been assessed at least once. On average, 15% of streams representing 8,923 km (24% of total shore length) are assessed each year. The composition as well as the extent of stations being monitored within a year is ever-changing, peaking at 46% of total shore length in 2002 as seen in the first row of the table below.

While the monitoring of water quality in Danish streams does cover the range of different pollution sources and physical conditions, it must be underlined that there is an overrepresentation of larger streams which generally have higher water quality. Moreover, a moderate to bad ecological status can in some cases be caused by influence of ochre or poor physical conditions due to stream straightening or intensive dredging and cutting of water weeds ([DCE, 2021 - *Danish*](https://dce.au.dk/udgivelser/vr/nr-451-500/abstracts/no-473-streams-2020)).

Thus, the statistics shown in the table should be taken with a grain of salt but shows the same positive trend that can be identified for many individual streams in the map book below.

#### Streams: Ecological status 1990-2020
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Share of known is High (%)</th>
      <th>Share of known is Good (%)</th>
      <th>Share of known is Moderate (%)</th>
      <th>Share of known is Poor (%)</th>
      <th>Share of known is Bad (%)</th>
      <th>Share of known is not Good (%)</th>
      <th>Status known (%)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1989</th>
      <td>2</td>
      <td>36</td>
      <td>46</td>
      <td>12</td>
      <td>2</td>
      <td>61</td>
      <td>10</td>
    </tr>
    <tr>
      <th>1990</th>
      <td>3</td>
      <td>39</td>
      <td>40</td>
      <td>14</td>
      <td>2</td>
      <td>56</td>
      <td>10</td>
    </tr>
    <tr>
      <th>1991</th>
      <td>5</td>
      <td>35</td>
      <td>42</td>
      <td>13</td>
      <td>2</td>
      <td>58</td>
      <td>9</td>
    </tr>
    <tr>
      <th>1992</th>
      <td>2</td>
      <td>36</td>
      <td>44</td>
      <td>14</td>
      <td>2</td>
      <td>61</td>
      <td>21</td>
    </tr>
    <tr>
      <th>1993</th>
      <td>4</td>
      <td>33</td>
      <td>43</td>
      <td>16</td>
      <td>2</td>
      <td>61</td>
      <td>24</td>
    </tr>
    <tr>
      <th>1994</th>
      <td>3</td>
      <td>33</td>
      <td>44</td>
      <td>17</td>
      <td>1</td>
      <td>62</td>
      <td>26</td>
    </tr>
    <tr>
      <th>1995</th>
      <td>4</td>
      <td>37</td>
      <td>44</td>
      <td>13</td>
      <td>1</td>
      <td>58</td>
      <td>27</td>
    </tr>
    <tr>
      <th>1996</th>
      <td>5</td>
      <td>34</td>
      <td>41</td>
      <td>15</td>
      <td>2</td>
      <td>59</td>
      <td>25</td>
    </tr>
    <tr>
      <th>1997</th>
      <td>5</td>
      <td>35</td>
      <td>44</td>
      <td>13</td>
      <td>1</td>
      <td>59</td>
      <td>27</td>
    </tr>
    <tr>
      <th>1998</th>
      <td>5</td>
      <td>34</td>
      <td>46</td>
      <td>12</td>
      <td>1</td>
      <td>60</td>
      <td>29</td>
    </tr>
    <tr>
      <th>1999</th>
      <td>5</td>
      <td>34</td>
      <td>48</td>
      <td>10</td>
      <td>1</td>
      <td>60</td>
      <td>38</td>
    </tr>
    <tr>
      <th>2000</th>
      <td>6</td>
      <td>36</td>
      <td>44</td>
      <td>12</td>
      <td>0</td>
      <td>56</td>
      <td>45</td>
    </tr>
    <tr>
      <th>2001</th>
      <td>7</td>
      <td>39</td>
      <td>42</td>
      <td>10</td>
      <td>0</td>
      <td>53</td>
      <td>44</td>
    </tr>
    <tr>
      <th>2002</th>
      <td>7</td>
      <td>39</td>
      <td>43</td>
      <td>8</td>
      <td>0</td>
      <td>52</td>
      <td>46</td>
    </tr>
    <tr>
      <th>2003</th>
      <td>9</td>
      <td>39</td>
      <td>40</td>
      <td>10</td>
      <td>0</td>
      <td>51</td>
      <td>44</td>
    </tr>
    <tr>
      <th>2004</th>
      <td>11</td>
      <td>34</td>
      <td>42</td>
      <td>10</td>
      <td>0</td>
      <td>54</td>
      <td>36</td>
    </tr>
    <tr>
      <th>2005</th>
      <td>9</td>
      <td>40</td>
      <td>40</td>
      <td>9</td>
      <td>0</td>
      <td>50</td>
      <td>33</td>
    </tr>
    <tr>
      <th>2006</th>
      <td>10</td>
      <td>43</td>
      <td>39</td>
      <td>6</td>
      <td>0</td>
      <td>46</td>
      <td>30</td>
    </tr>
    <tr>
      <th>2007</th>
      <td>9</td>
      <td>41</td>
      <td>40</td>
      <td>8</td>
      <td>0</td>
      <td>49</td>
      <td>28</td>
    </tr>
    <tr>
      <th>2008</th>
      <td>12</td>
      <td>42</td>
      <td>38</td>
      <td>5</td>
      <td>0</td>
      <td>44</td>
      <td>24</td>
    </tr>
    <tr>
      <th>2009</th>
      <td>14</td>
      <td>45</td>
      <td>32</td>
      <td>6</td>
      <td>0</td>
      <td>39</td>
      <td>25</td>
    </tr>
    <tr>
      <th>2010</th>
      <td>15</td>
      <td>42</td>
      <td>31</td>
      <td>9</td>
      <td>0</td>
      <td>41</td>
      <td>25</td>
    </tr>
    <tr>
      <th>2011</th>
      <td>9</td>
      <td>43</td>
      <td>38</td>
      <td>8</td>
      <td>0</td>
      <td>47</td>
      <td>29</td>
    </tr>
    <tr>
      <th>2012</th>
      <td>12</td>
      <td>45</td>
      <td>33</td>
      <td>7</td>
      <td>0</td>
      <td>42</td>
      <td>28</td>
    </tr>
    <tr>
      <th>2013</th>
      <td>10</td>
      <td>46</td>
      <td>34</td>
      <td>7</td>
      <td>0</td>
      <td>43</td>
      <td>33</td>
    </tr>
    <tr>
      <th>2014</th>
      <td>17</td>
      <td>49</td>
      <td>26</td>
      <td>6</td>
      <td>0</td>
      <td>33</td>
      <td>28</td>
    </tr>
    <tr>
      <th>2015</th>
      <td>17</td>
      <td>46</td>
      <td>28</td>
      <td>7</td>
      <td>0</td>
      <td>36</td>
      <td>24</td>
    </tr>
    <tr>
      <th>2016</th>
      <td>21</td>
      <td>44</td>
      <td>29</td>
      <td>4</td>
      <td>0</td>
      <td>34</td>
      <td>28</td>
    </tr>
    <tr>
      <th>2017</th>
      <td>15</td>
      <td>46</td>
      <td>30</td>
      <td>6</td>
      <td>0</td>
      <td>37</td>
      <td>28</td>
    </tr>
    <tr>
      <th>2018</th>
      <td>15</td>
      <td>45</td>
      <td>32</td>
      <td>6</td>
      <td>0</td>
      <td>39</td>
      <td>39</td>
    </tr>
    <tr>
      <th>2019</th>
      <td>20</td>
      <td>44</td>
      <td>28</td>
      <td>6</td>
      <td>0</td>
      <td>35</td>
      <td>28</td>
    </tr>
    <tr>
      <th>2020</th>
      <td>23</td>
      <td>46</td>
      <td>25</td>
      <td>4</td>
      <td>0</td>
      <td>30</td>
      <td>25</td>
    </tr>
  </tbody>
</table>

The map book below shows the ecological status of streams for each year from 1992-2019. As stands out, initially the DVFI index was mostly used in the two [counties](https://en.wikipedia.org/wiki/Counties_of_Denmark) Viborg and Ringkjøbing (in mid-western Jutland).

If your browser does not allow you to navigate the map book, you can [download it here](https://github.com/thornoe/GreenGDP/raw/master/gis/output/streams.pdf) instead.

#### Streams: Ecological status mapped for each year
<iframe src="//www.slideshare.net/slideshow/embed_code/key/vafjNGN9GUGLOm" width="900" height="1200" frameborder="0" marginwidth="0" marginheight="0" scrolling="no" style="border:1px solid #CCC; border-width:1px; margin-bottom:1px; max-width: 100%; max-height: 100%;" allowfullscreen> </iframe>


## Assessing the ecological status of waterbodies

The ecological conditions in waterbodies can be quantified directly in form of the concentration of chlorophyll *a* which are the green pigments found in phytoplankton (for **lakes** and **coastal waters**) or indirectly using biological indicators that are responsive to eutrophication.

For **streams**, the most long-lasting biological indicator for nutrient pollution in Danish streams is the **DVFI index** *(Dansk Vandløbsfaunaindeks)* which measures the occurrence and composition of benthic macroinvertebrates (small bottom-living animals without backbone). The index is one of many modified versions of the Trent Biotic Index (TBI) that was developed in 1964 for the River Trent watershed in England ([Andersen et al, 1984 - *English*](https://www.sciencedirect.com/science/article/abs/pii/0043135484900629)). In 1998 the Danish index was improved as observations before that were less consistent ([Danish Environmental Protection Agency, 1998 - *Danish*](https://www2.mst.dk/udgiv/Publikationer/1998/87-7810-995-7/pdf/87-7810-995-7.PDF)).

### From DVFI observations to ecological status of streams

If a station has multiple observations within a year, take the median and round down. Subsequently, do the same if a stream has observations for multiple stations within a year. Thus, a unique DVFI fauna class is obtained for each stream for every year it has been observed ([SVANA, 2016 *- Danish*](https://mst.dk/media/121345/retningslinjer-vandomraadeplaner-for-anden-planperiode.pdf)).

Finally, the seven fauna classes are converted to the five EU-WFD categories of ecological status: Bad, Poor, Moderate, Good, or High ([ministerial order no. 1001, 2016, Appendix 3 *- Danish*](https://www.retsinformation.dk/Forms/R0710.aspx?id=181970)).

### Data and reproducible code

See [GitHub.com/ThorNoe/GreenGDP](https://github.com/ThorNoe/GreenGDP) for data, code and instructions on how to reproduce or update the data and maps.
