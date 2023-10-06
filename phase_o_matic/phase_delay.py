import numpy as np
import xarray as xr
from scipy import integrate

from typing import Union

def calculate_dry_refractivity(dataset: xr.Dataset) -> xr.Dataset:
    """
    Calculate dry/hydro-static refractivity from air pressure, vapor pressure, and temperature
    Following Jolivet et al (2011, 2014).
    
    Args:
    dataset: xarray dataset with air_pressure, vapor_pressure, and temperature

    Returns:
    dataset: xarray dataset with dry refractivity added
    """

    if 'units' in dataset['air_pressure'].attrs:
        assert dataset['air_pressure'].attrs['units'] == 'pascals'
    if 'units' in dataset['temperature'].attrs:
        assert dataset['temperature'].attrs['units'] == 'K'
    
    # constant for dry refractivity
    k1 = 0.776 # K / Pa
    Rd = 287.05 # j / kg K
    g = 9.81 # m / s

    if np.all(np.diff(dataset.height, 1) < 0):
        # reverse heights so we integrate from satellite to ground (high -> low)
        dataset = dataset.reindex(height=list(reversed(dataset.height)))

    # calculate dry refractivity. Reference: https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.453-11-201507-S!!PDF-E.pdf
    # adapted from eq 3 of Jolivet et al (2014) and https://github.com/insarlab/PyAPS/blob/main/src/pyaps3/processor.py#L216
    # https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1002/2013JB010588?src=getftr
    dataset['N_dry'] = 1.0e-6 * k1 * Rd / g * (dataset['air_pressure'] - dataset['air_pressure'].isel(height = -1))

    dataset['N_dry'] = dataset['N_dry'].assign_attrs(units = '', long_name = 'Hydrostatic Refractivity')

    return dataset

def calculate_wet_refractivity(dataset: xr.Dataset) -> xr.Dataset:
    """
    Calculate wet refractivity from vapor pressure, and temperature
    
    Args:
    dataset: xarray dataset with air_pressure, vapor_pressure, and temperature

    Returns:
    dataset: xarray dataset with wet refractivity added
    """

    if 'units' in dataset['vapor_pressure'].attrs:
        assert dataset['vapor_pressure'].attrs['units'] == 'pascals'
    if 'units' in dataset['temperature'].attrs:
        assert dataset['temperature'].attrs['units'] == 'K'

    if not np.all(np.diff(dataset.height, 1) > 0):
        # reverse heights so we integrate from satellite to ground (high -> low)
        dataset = dataset.reindex(height=list(reversed(dataset.height)))

    # constants for wet refractivity
    k1 = 0.776 # K / Pa
    k2 = 0.716 # K / Pa
    k3 = 3750 # K^2 / Pa
    Rd = 287.05 # j / kg K
    Rv = 461.495 # j / kg K

    # calculate wet refractivity
    # adapted from eq 4 of Jolivet et al (2014) and https://github.com/insarlab/PyAPS/blob/main/src/pyaps3/processor.py#L218-L229
    # of eq 5 of Doin et al. 2009
    # https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1002/2013JB010588?src=getftr
    
    # integrate using cumulative trapezoids for vapor pressure over temperature
    s1 = xr.apply_ufunc(integrate.cumtrapz, dataset['vapor_pressure']/dataset['temperature'], kwargs = {'initial': np.nan, 'x' : dataset['height']})
    # we lose the first value so we need to add a value at the end (we use the inital value and then slice it out to avoid xarray throwing an error)
    # calculate end value from linear fit of last two points
    end_val = (2*s1.isel(height = -1) - s1.isel(height = -2)).expand_dims(height = [dataset.isel(height = -1).height + 1])
    # add to end of s1 dataArray
    s1 = xr.concat([s1.isel(height = slice(1, s1.height.size)), end_val], dim = 'height')
    # set s1 height to right value
    s1['height'] = dataset['height']

    # same over vapor pressure / temp **2
    s2 = xr.apply_ufunc(integrate.cumtrapz, dataset['vapor_pressure']/dataset['temperature']**2, kwargs = {'initial': np.nan, 'x' : dataset['height']})
    end_val = (2*s2.isel(height = -1) - s2.isel(height = -2)).expand_dims(height = [dataset.isel(height = -1).height + 1])
    s2 = xr.concat([s2.isel(height = slice(1, s1.height.size)), end_val], dim = 'height')
    s2['height'] = dataset['height']

    # calculate for each point along height axis
    dataset['N_wet'] = -1.0e-6 * (((k2 - (k1 * Rd / Rv)) * s1) + ( k3 * s2))

    # get change from highest elevation
    dataset['N_wet'] = dataset['N_wet'] - dataset['N_wet'].isel(height = -1)

    dataset['N_wet'] = dataset['N_wet'].assign_attrs(units = '', long_name = 'Water Vapor Refractivity')
    
    return dataset

def calculate_refractive_indexes(dataset: xr.Dataset, wavelength: float = 0.23) -> xr.Dataset:
    """
    Calculate dry and wet refractivity from air pressure, vapor pressure, and temperature.
    Integrate refractivity through atmosphere from high to low.
    
    Args:
    dataset: xarray dataset with air_pressure, vapor_pressure, and temperature

    Returns:
    dataset: xarray dataset with dry, wet, combined, and integrated refractivity
    """

    # calculate the refractive index from air pressure
    dataset = calculate_dry_refractivity(dataset)
    # calculate refractivity index from water vapor
    dataset = calculate_wet_refractivity(dataset)
    # combine wet and dry refractivity
    dataset['N'] = dataset['N_dry'] + dataset['N_wet']

    dataset['N'] = dataset['N'].assign_attrs(units = '', long_name = 'Total Refractivity')

    return dataset

def get_delay(dataset: xr.Dataset, dem: xr.DataArray, inc: Union[xr.DataArray, float], wavelength: float = 4*np.pi) -> xr.Dataset:
    """
    Get delay in m (default) or radians (if wavelength provided) for each height
    https://github.com/insarlab/PyAPS/blob/244552cdfcf4e1a55de5f1439be4f08eb45872ec/src/pyaps3/objects.py#L213

    Args:
    ds: xarray dataset of atmosphere with refractive indexes calculated
    dem: xarray dataArray of elevations with coords for "latitude" and "longitude".
    inc: xarray dataArray of incidence angles in radians or a single float to use over the image
    wavelength: either default 4 * pi (returns meters of delay) or radar wavelength in meters (returns radians)

    Returns:
    delay: cos calculated phase delay for each weather station data point interpolated to each meter of DEM
    """

    assert wavelength > 0, "Can not have negative wavelength"

    assert 'N' in dataset.data_vars, "Must have refractivity index calculated to interpolate to DEM"

    assert "latitude" in dem.coords, "Must have latitude as coordinate"
    assert "longitude" in dem.coords, "Must have longitude as coordinate"

    if isinstance(inc, xr.DataArray):

        assert inc.mean() < 2*np.pi, f'Incidence mean over 2 pi. Check if you are inc in radians'

        # check to see if incidence angle and DEM are aligned and align them if not
        try:
            xr.align(dem, inc, join='exact')
        except ValueError:
            inc = inc.interp_like(dem)

            # check if we got rid of all the values due to no overlap
            assert inc.isnull().sum() != dem.size, "No overlap between incidence angle and dem"

    else:
        assert inc < 2*np.pi, f'Incidence mean over 2 pi. Check if you are inc in radians'
    
    # check if atmospheric values cover the dem
    assert dataset.height.min() <= dem.min()
    assert dataset.height.max() >= dem.max()

    # interpolate heights to each meter from the lowest point of the dem to the highest from our current spacing
    dataset = dataset.interp(height = np.round(np.arange(dem.min() - 2, dem.max() + 2)), method = 'linear')

    # interpolate across the dem's lats, longs, and elevations to find the surface delay
    dataset = dataset.interp(latitude = dem.latitude, longitude = dem.longitude, height = dem)

    # add phase delay data_variable
    dataset['delay'] = dataset['N']*np.pi*4.0/(wavelength*np.cos(inc))

    if wavelength == 4 * np.pi:
        dataset['delay'].attrs['units'] = 'meters'
    else:
        dataset['delay'].attrs['units'] = 'radians'

    dataset['delay'].attrs['long_name'] = 'Atmospheric Delay'

    # make sure we have coordinates in the right order for plotting
    dataset = dataset.transpose('time','latitude','longitude')

    return dataset
    