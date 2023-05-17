# phase-o-matic
[![PIP](https://img.shields.io/badge/pip-0.0.9-purple)](https://img.shields.io/badge/pip-0.0.9-purple)
[![LICENSE](https://img.shields.io/badge/license-MIT-orange)](https://img.shields.io/badge/license-MIT-orange)
[![DOI](https://zenodo.org/badge/636333382.svg)](https://zenodo.org/badge/latestdoi/636333382) 
[![COVERAGE](https://img.shields.io/badge/coverage-89%25-green)](https://img.shields.io/badge/coverage-89%25-green) 


Python package for calculating Interferometric Synthetic Aperture Radar phase delays from ERA5 atmospheric models. Utilizes xarray to easily download, processes, and add phase delays to netcdfs of InSAR Phase.

Useful publications for this repo:
 - Doin, M.-P., Lasserre, C., Peltzer, G., Cavalié, O., and Doubre, C.: Corrections of stratified tropospheric delays in SAR interferometry: Validation with global atmospheric models, J Appl Geophys, 69, 35–50, https://doi.org/10.1016/j.jappgeo.2009.03.010, 2009.

 - Jolivet, R., Agram, P. S., Lin, N. Y., Simons, M., Doin, M., Peltzer, G., and Li, Z.: Improving InSAR geodesy using Global Atmospheric Models, J Geophys Res Solid Earth, 119, 2324–2341, https://doi.org/10.1002/2013jb010588, 2014.

 - Hu, Z. and Mallorquí, J. J.: An Accurate Method to Correct Atmospheric Phase Delay for InSAR with the ERA5 Global Atmospheric Model, Remote Sens-basel, 11, 1969, https://doi.org/10.3390/rs11171969, 2019.

This code and ERA data registration instructions are adapted from the awesome atmospheric phase delay repo: https://github.com/insarlab/PyAPS.

<img src="https://github.com/ZachKeskinen/phase-o-matic/blob/main/images/pyaps_phaseo_compare.png">

## Installation

### Pip installation
```bash
pip install phase_o_matic
```

### ERA5 Data Registration

[ERA5](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels) is atmospheric data distributed by the Copernicus Climate Change Service. You must register for an account and save the provided locally where you are downloading.

1. [Create an account](https://cds.climate.copernicus.eu/user/register) with the Copernicus Climate Data Servce.
2. Next create a new file in your home directory called `.cdsapirc` 
```bash
cd ~
nano .cdsapirc
```
with the following text:

```
url: https://cds.climate.copernicus.eu/api/v2
key: 12345*abcdefghij-134-abcdefgadf-82391b9d3f
```

Where you have replaced *12345* with your previous user ID and the part behind the colon (*abcdefghij-134-abcdefgadf-82391b9d3f*) with your personal API key. [More details](https://cds.climate.copernicus.eu/api-how-to)

3. Make sure you have accepted the data licences Terms on the ECMWF website

## Usage

This example usage is also available in `notebooks/usage.ipynb`

```python
import sys
import xarray as xr
import matplotlib.pyplot as plt

from phase_o_matic import presto_phase_delay

# this relative path assumes you are in the notebooks directory
dem = xr.open_dataset('../pyAPS_data/pyaps_geom.nc')['dem']
inc = xr.open_dataset('../pyAPS_data/pyaps_geom.nc')['inc']

work_dir = '../pyAPS_data/example'

t1 = presto_phase_delay(date = '2020-01-03', dem = dem, inc = inc, work_dir = work_dir, wavelength = 0.238403545)
t2 = presto_phase_delay(date = '2020-01-10', dem = dem, inc = inc, work_dir = work_dir, wavelength = 0.238403545)

delay_change = t2.isel(time = 0)['delay'] - t1.isel(time = 0)['delay']

fig, axes = plt.subplots(1, 2, figsize = (12, 9))
delay_change.plot(ax = axes[0], vmax = 0, vmin = -4)
dem.plot(ax = axes[1], vmin = 0, vmax = 2000)
plt.savefig('../images/usage.png')
```

## Coverage instructions

Run the following from the root directory of this project to get a coverage report.

You will need to have the dependencies and `coverage` packages available.

```bash
python -m coverage run -m unittest discover -s ./tests
python -m coverage report
```

## Citations

If you end up finding this repo useful consider citing this repo as:

Keskinen, Z. (2023) Phase-o-matic: InSAR atmospheric delay calculations, https://github.com/ZachKeskinen/phase-o-matic/. DOI: 10.5281/zenodo.7926686

and any use of the ERA5 data should include the ERA5 data citation:

Hersbach, H., Bell, B., Berrisford, P., Biavati, G., Horányi, A., Muñoz Sabater, J., Nicolas, J., Peubey, C., Radu, R., Rozum, I., Schepers, D., Simmons, A., Soci, C., Dee, D., Thépaut, J-N. (2023): ERA5 hourly data on single levels from 1940 to present. Copernicus Climate Change Service (C3S) Climate Data Store (CDS), DOI: 10.24381/cds.adbb2d47
