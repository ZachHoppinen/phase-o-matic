# phase-o-matic
Python package for calculating Interferometric Synthetic Aperture Radar phase delays from ERA5 atmospheric models. Utilizes x-array to easily download, processes, and add phase delays to netcdfs of InSAR Phase. 

Useful publications for this repo:
 - Doin, M.-P., Lasserre, C., Peltzer, G., Cavalié, O., and Doubre, C.: Corrections of stratified tropospheric delays in SAR interferometry: Validation with global atmospheric models, J Appl Geophys, 69, 35–50, https://doi.org/10.1016/j.jappgeo.2009.03.010, 2009.

 - Jolivet, R., Agram, P. S., Lin, N. Y., Simons, M., Doin, M., Peltzer, G., and Li, Z.: Improving InSAR geodesy using Global Atmospheric Models, J Geophys Res Solid Earth, 119, 2324–2341, https://doi.org/10.1002/2013jb010588, 2014.

 - Hu, Z. and Mallorquí, J. J.: An Accurate Method to Correct Atmospheric Phase Delay for InSAR with the ERA5 Global Atmospheric Model, Remote Sens-basel, 11, 1969, https://doi.org/10.3390/rs11171969, 2019.

This code is adapted from the awesome atmospheric phase delay repo: https://github.com/insarlab/PyAPS utilizing xarray.

<img src="https://github.com/ZachKeskinen/phase-o-matic/blob/main/images/pyaps_phaseo_compare.png">

## Usage

```python
import xarray as xr
from phase_o_matic import presto_phase_delay

dem_dataset = xr.open_dataset('../pyAPS_data/pyaps_geom.nc')

t1 = presto_phase_delay(date = '2020-01-03', geometry = dem_dataset, work_dir = '../data/test/', wavelength = 0.238403545)
t2 = presto_phase_delay(date = '2020-01-10', geometry = dem_dataset, work_dir = '../data/test/', wavelength = 0.238403545)

delay_change = t2.isel(time = 0)['delay'] - t1.isel(time = 0)['delay']

delay_change.plot()
```
