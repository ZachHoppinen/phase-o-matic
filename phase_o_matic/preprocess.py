import numpy as np
import xarray as xr
from scipy import interpolate
import pyproj

from typing import Union

def convert_pressure_to_pascals(dataset: xr.Dataset) -> xr.Dataset:
    """
    Converts pressure levels from millibars to pascals

    Args:
    dataset: xarray dataset with coordinate level in millibars

    Returns:
    dataset: xarray dataset with coordinate level in pascals
    """

    if dataset['level'].attrs['units'] != 'pascals':
        dataset['level'] = dataset['level'] * 100
        dataset['level'] = dataset['level'].assign_attrs(units = 'pascals', long_name = 'pressure_level')

    else:
        print("Pressure levels already converted to pascals")

    return dataset

def get_vapor_partial_pressure(dataset: xr.Dataset) -> xr.Dataset:
    """
    Get vapor partial pressure from specific humidity for each pressure level

    Args:
    dataset: dataset without vpr pressure as a variable

    Returns:
    dataset: dataset without "vpr" - vapor pressure as a variable
    """
    # molar gas constant of water vapor
    Rv = 461.495  #(J/Kg/K)
    # molar gas constant of dry air
    Rd = 287.05  #(J/Kg/K)
    # ratio of the two in air
    alpha = Rv/Rd

    assert 'q' in dataset.data_vars, "Must have specific humidity as a data variable"
    assert dataset['q'].attrs['standard_name'] == 'specific_humidity'

    # specific humidity calculation of partial pressure of water vapour
    dataset['vpr'] = dataset['q']* dataset['level']* alpha/ (1+ (alpha - 1)* dataset['q'])

    dataset['vpr'].attrs['units'] = 'pascals'

    return dataset

def geopotential_heights_to_geoid(heights: xr.DataArray) -> xr.DataArray:
    """
    Convert height array from geopotential to WGS84 Ellipsoidal heights

    Args:
    heights: xarray dataarray of geopotentials at lat and long

    Returns:
    heights: xarray dataarray of ellipsoidal heights
    """

    # acceleration due to gravity for geopotential
    g = 9.80665 # m s-2

    # convert from geopotential to geopotenial height
    heights = heights / g

    # convert to geometric height with earth radius
    # set earth radius from ERA5 documentation
    # https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation
    Er = 6371229.0 # meters 
    heights = heights * Er / (Er - heights) 

    # get geoid difference from at his location
    tform = pyproj.transformer.Transformer.from_crs(crs_from = 4979, crs_to = 5773)
    # get what the height of out data is different from the 0 m point on the geoid
    _, _, geoid_N = tform.transform(heights['latitude'], heights['longitude'], 0)
    # subtract geoid difference to get to ellipsoidal height
    elipsoidal_height = heights - geoid_N

    return elipsoidal_height

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

    # acceleration due to gravity for geopotential
    g = 9.80665 # m s-2

    # calculate height range from -200 meters to highest potential height
    max_alt = dataset['z'].max().round() / g if not max_alt else max_alt
    heights = np.linspace(min_alt, max_alt, n_heights)

    # create empty array to hold interpolated data but with n heights instead of 37 pressure levels
    new_size = (1, dataset['longitude'].size, dataset['latitude'].size, n_heights)

    interpolated_ds = xr.Dataset(
            {
                'air_pressure': (['time','longitude', 'latitude','height'], np.zeros(new_size), dataset['level'].attrs),
                'temperature': (['time', 'longitude', 'latitude','height'], np.zeros(new_size), dataset['t'].attrs),
                'vapor_pressure': (['time', 'longitude', 'latitude','height'], np.zeros(new_size), dataset['vpr'].attrs),
            },  
            coords = {
                "longitude" : (["longitude"], dataset['longitude'].data, dataset['longitude'].attrs),
                "latitude" : (["latitude"], dataset['latitude'].data, dataset['latitude'].attrs),
                "height" : (["height"], heights, {'units' :'meters', 'long_name' :'ellipsoidal_height'}),
                "time" : (["time"], dataset['time'].data, {'long_name' :'time'}),
            },
            attrs=dataset.attrs
        )

    # scipy 1d interpolate won't accept 2d array as y values so have to iterate
    for lon in dataset.longitude:
        for lat in dataset.latitude:

            # get geopotential for each pressure level at this lat and long
            hx = dataset['z'].isel(time = 0).sel(latitude = lat, longitude = lon)
            # convert those geopotentials to ellipsoidal WGS84 heights
            hx = geopotential_heights_to_geoid(hx)

            # interpolate pressure levels to pressure at each height
            hy = dataset['level']
            # make cubic interpolator
            tck = interpolate.interp1d(hx, hy, axis = -1, kind = 'cubic', fill_value = "extrapolate")
            # interpolate what pressure would occur at each height
            interpolated_pressure = tck(heights)
            # add to interpolated dataset at correct lat and long
            interpolated_ds['air_pressure'].loc[{'latitude': lat, 'longitude': lon}] =  xr.DataArray(interpolated_pressure, coords = [heights], dims = 'height')

            # interpolate temperatures to temperature at each height level
            hy = dataset['t'].isel(time = 0).sel(latitude = lat, longitude = lon).data
            tck = interpolate.interp1d(hx, hy, axis = -1, kind = 'cubic', fill_value = "extrapolate")
            interpolated_temperatures = tck(heights)
            interpolated_ds['temperature'].loc[{'latitude': lat, 'longitude': lon}] =  xr.DataArray(interpolated_temperatures, coords = [heights], dims = 'height')

            # interpolate vapor pressures to vapor pressures at each height level
            hy = dataset['vpr'].isel(time = 0).sel(latitude = lat, longitude = lon).data
            tck = interpolate.interp1d(hx, hy, axis = -1, kind = 'cubic', fill_value = "extrapolate")
            interpolated_vapor= tck(heights)
            interpolated_ds['vapor_pressure'].loc[{'latitude': lat, 'longitude': lon}] =  xr.DataArray(interpolated_vapor, coords = [heights], dims = 'height')
        
    return interpolated_ds