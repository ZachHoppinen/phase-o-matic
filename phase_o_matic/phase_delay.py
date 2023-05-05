import numpy as np
import xarray as xr
from scipy import integrate

def calculate_dry_refractivity(dataset: xr.Dataset) -> xr.Dataset:
    """
    Calculate dry refractivity from air pressure, vapor pressure, and temperature
    
    Args:
    dataset: xarray dataset with air_pressure, vapor_pressure, and temperature

    Returns:
    dataset: xarray dataset with dry refractivity added
    """

    assert dataset['air_pressure'].attrs['units'] == 'pascals'
    assert dataset['temperature'].attrs['units'] == 'K'
    
    # constant for dry refractivity
    k1 = 0.776 # K Pa^-1
    
    # calculate dry refractivity. Reference: https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.453-11-201507-S!!PDF-E.pdf
    dataset['N_dry'] = k1 * (dataset['air_pressure'] - dataset['vapor_pressure']) / dataset['temperature']

    return dataset

def calculate_wet_refractivity(dataset: xr.Dataset) -> xr.Dataset:
    """
    Calculate wet refractivity from vapor pressure, and temperature
    
    Args:
    dataset: xarray dataset with air_pressure, vapor_pressure, and temperature

    Returns:
    dataset: xarray dataset with wet refractivity added
    """

    assert dataset['vapor_pressure'].attrs['units'] == 'pascals'
    assert dataset['temperature'].attrs['units'] == 'K'

    # constants for wet refractivity
    k2 = 0.716 # K Pa^-1
    k3 = 3750 # K^2 Pa^-1

    # calculate wet refractivity. Reference: https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.453-11-201507-S!!PDF-E.pdf
    dataset['N_wet'] = k2 * dataset['vapor_pressure'] / dataset['temperature'] + k3 * dataset['vapor_pressure'] / dataset['temperature']**2

    return dataset

def integrate_refractivity(dataset: xr.Dataset) -> xr.Dataset:
    """
    Integrate combined refractivity through atmosphere from high to low.
    
    Args:
    dataset: xarray dataset with combined refractivity and heights (m)

    Returns:
    dataset: xarray dataset with cumulative integrated refractivity added
    """
    # check if heights are increasing
    if np.all(np.diff(dataset.height, 1) > 0):
        # reverse heights so we integrate from satellite to ground (high -> low)
        dataset = dataset.reindex(height=list(reversed(dataset.height)))
    # calculate integrated refractivity index from high to low at each lat, long
    integrated_N = integrate.cumtrapz(dataset['N'], x = dataset['height'])
    # get first value to pad integration from highest and next highest value
    start_val = 2*integrated_N[:, :, :, 0] - integrated_N[:, :, :, 1]
    # insert as new variable cumulative refractivity index
    dataset['cum_N'] = (['time', 'longitude', 'latitude', 'height'], np.insert(integrated_N, 0, start_val, -1))

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
    # calculate integrated refractivity index from high to low
    dataset = integrate_refractivity(dataset)
    # calculate integrated atmospheric phase delay through atmosphere
    dataset['atm_phase'] = -4 * np.pi / wavelength * 1.0e-6 *  dataset['cum_N']

    return dataset
