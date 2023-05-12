import numpy as np
import xarray as xr
from scipy import interpolate

from typing import Union

def convert_pressure_to_pascals(dataset: xr.Dataset) -> xr.Dataset:
    """
    Converts pressure levels from millibars to pascals

    Args:
    dataset: xarray dataset with coordinate level in millibars

    Returns:
    dataset: xarray dataset with coordinate level in pascals
    """

    assert 'units' in dataset['level'].attrs, "No assigned units for pressure levels ['level']"

    if dataset['level'].attrs['units'] == 'millibars':
        dataset['level'] = dataset['level'] * 100
        dataset['level'] = dataset['level'].assign_attrs(units = 'pascals', long_name = 'Pressure Level')
    elif dataset['level'].attrs['units'] == 'pascals':
        pass
        # print("Pressure levels already converted to pascals")
    else:
        raise ValueError(f"Unknown units on 'level' coordinate: {dataset['level'].attrs['units']}\
            must be one of 'pascals' or 'millibars'")
    

    return dataset

def cc_era(temperature: xr.DataArray):
    '''Clausius Clayperon law used by ERA Interim.
    https://github.com/insarlab/PyAPS/blob/main/src/pyaps3/era.py#L14

    Args:
        * tmp  (np.ndarray) : Temperature.

    Returns:
        * esat (np.ndarray) : Water vapor saturation partial pressure.'''

    a1w = 611.21
    a3w = 17.502
    a4w = 32.19
    a1i = 611.21
    a3i = 22.587
    a4i = -0.7
    T3  = 273.16
    Ti  = 250.16

    sat_vapor_pressure = xr.zeros_like(temperature)

    esatw = a1w*np.exp(a3w*(temperature-T3)/(temperature-a4w))
    esati = a1i*np.exp(a3i*(temperature-T3)/(temperature-a4i))
    wgt = (temperature-Ti)/(T3-Ti)

    sat_vapor_pressure = sat_vapor_pressure.where(temperature < T3, esatw)
    sat_vapor_pressure = sat_vapor_pressure.where(temperature > Ti, esati)
    sat_vapor_pressure = sat_vapor_pressure.where((temperature >= T3) | (temperature <= Ti), esati + (esatw - esati)*wgt*wgt)

    sat_vapor_pressure = sat_vapor_pressure.rename('vapor_saturation_pressure')

    return sat_vapor_pressure

def get_vapor_partial_pressure(dataset: xr.Dataset) -> xr.Dataset:
    """
    Get vapor partial pressure from humidity for each pressure level

    Args:
    dataset: dataset without vpr pressure as a variable

    Returns:
    dataset: dataset without "vpr" - vapor pressure as a variable
    """
    
    if 'q' in dataset.data_vars:
        # molar gas constant of water vapor
        Rv = 461.495  #(J/Kg/K)
        # molar gas constant of dry air
        Rd = 287.05  #(J/Kg/K)
        # ratio of the two in air
        alpha = Rv/Rd
        # specific humidity calculation of partial pressure of water vapour
        # https://cran.r-project.org/web/packages/humidity/vignettes/humidity-measures.html equation 4
        # http://www.atmo.arizona.edu/students/courselinks/spring08/atmo336s1/courses/fall13/atmo551a/Site/ATMO_451a_551a_files/WaterVapor.pdf
        dataset['vpr'] = dataset['q'] * dataset['level'] * alpha/ (1+ (alpha - 1)* dataset['q'])

    elif 'r' in dataset.data_vars:
        h2o_sat_partial_p = cc_era(dataset['t'])
        dataset['vpr'] = h2o_sat_partial_p * dataset['r']/100
    
    else:
        raise ValueError("Must have specific or relative humidity as a data variable")

    dataset['vpr'].attrs['units'] = 'pascals'
    dataset['vpr'].attrs['long_name'] = 'Vapor Partial Pressure'

    return dataset

def geopotential_to_geopotential_heights(ds: xr.Dataset) -> xr.Dataset:
    """
    Convert height array from geopotential to geopotential heights

    Args:
    ds: xarray dataset with geopotentials

    Returns:
    ds: xarray dataarray with 'gph' added
    """
    # acceleration due to gravity for geopotential
    g = 9.81 # m s-2
    # convert from geopotential to geopotenial height
    ds['gph'] = ds['z'] / g

    ds['gph'] = ds['gph'].assign_attrs(units = 'meters', long_name = 'Geopotential Height')

    return ds

def interpolate_to_heights(dataset: xr.Dataset, min_alt = -200, n_heights = 300, max_alt: Union[None, float, xr.DataArray] = None) -> xr.Dataset:
    """
    Interpolate vapor pressure, air pressure, and temperatures from pressure level coordinates
    to geopotential height coordinates. Cubic spline is used for each lat and lon.

    Args:
    dataset: xarray dataset with vpr, t data variables, and with level as coordinate
    min_alt: [-200] minimum height in meters to interpolate form
    n_heights: [300] number of evenly space heights to use in interpolating

    Returns:
    interpolated_ds: xarray dataset with coordinate "heights" of n_heights and data variables
    "temperature", "air_pressure", and "vapor_pressure" interpolated to each geopotential height. 
    """

    assert 'vpr' in dataset.data_vars, "No vapor pressure data. Please calculate first."
    assert 'level' in dataset.coords, "Need pressure level as a coordinate."

    # calculate height range from -200 meters to highest potential height
    max_alt = dataset['gph'].max().round() if not max_alt else max_alt
    heights = np.linspace(min_alt, max_alt, n_heights)

    # create empty array to hold interpolated data but with n heights instead of 37 pressure levels
    new_size = (dataset['time'].size, dataset['longitude'].size, dataset['latitude'].size, n_heights)

    interpolated_ds = xr.Dataset(
            {
                'air_pressure': (['time','longitude', 'latitude','height'], np.zeros(new_size), dataset['level'].attrs),
                'temperature': (['time', 'longitude', 'latitude','height'], np.zeros(new_size), dataset['t'].attrs),
                'vapor_pressure': (['time', 'longitude', 'latitude','height'], np.zeros(new_size), dataset['vpr'].attrs),
            },  
            coords = {
                "longitude" : (["longitude"], dataset['longitude'].data, dataset['longitude'].attrs),
                "latitude" : (["latitude"], dataset['latitude'].data, dataset['latitude'].attrs),
                "height" : (["height"], heights, dataset['gph'].attrs),
                "time" : (["time"], dataset['time'].data, {'long_name' :'time'}),
            },
            attrs=dataset.attrs
        )

    # scipy 1d interpolate won't accept 2d array as y values so have to iterate
    for time in dataset.time:
        for lon in dataset.longitude:
            for lat in dataset.latitude:
                
                # get geopotential height for each pressure level at this lat and long
                hx = dataset['gph'].sel(time = time).sel(latitude = lat, longitude = lon)

                # convert those geopotential heights to ellipsoidal WGS84 heights
                # skipped to match pyAPS
                # hx = geopotential_heights_to_geoid(hx)

                # interpolate pressure levels to pressure at each height
                hy = dataset['level']
                # make cubic interpolator
                tck = interpolate.interp1d(hx, hy, axis = -1, kind = 'cubic', fill_value = "extrapolate")
                # interpolate what pressure would occur at each height
                interpolated_pressure = tck(heights)
                # add to interpolated dataset at correct lat and long
                interpolated_ds['air_pressure'].loc[{'time': time,  'latitude': lat, 'longitude': lon}] =  xr.DataArray(interpolated_pressure, coords = [heights], dims = 'height')

                # interpolate temperatures to temperature at each height level
                hy = dataset['t'].sel(time = time).sel(latitude = lat, longitude = lon).data
                tck = interpolate.interp1d(hx, hy, axis = -1, kind = 'cubic', fill_value = "extrapolate")
                interpolated_temperatures = tck(heights)
                interpolated_ds['temperature'].loc[{'time': time,  'latitude': lat, 'longitude': lon}] =  xr.DataArray(interpolated_temperatures, coords = [heights], dims = 'height')
                
                # interpolate vapor pressures to vapor pressures at each height level
                hy = dataset['vpr'].sel(time = time).sel(latitude = lat, longitude = lon).data
                tck = interpolate.interp1d(hx, hy, axis = -1, kind = 'cubic', fill_value = "extrapolate")
                interpolated_vapor= tck(heights)
                interpolated_ds['vapor_pressure'].loc[{'time': time,  'latitude': lat, 'longitude': lon}] =  xr.DataArray(interpolated_vapor, coords = [heights], dims = 'height')
    
    interpolated_ds['air_pressure']= interpolated_ds.air_pressure.assign_attrs(long_name = 'Air Pressure')

    return interpolated_ds