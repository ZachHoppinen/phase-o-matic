import numpy as np
import xarray as xr
import rioxarray as rxa
from scipy.interpolate import griddata

def regrid_twoD_coords(dA: xr.DataArray):
    """
    Regrid data array with 2d coordinates (latitude and longitude) to 1d coordinates.

    dataarray must have dims x and y and coordinates lat and lon both of which are 
    a function of the x and the y dimensions.

    This might be a better approach: https://gis.stackexchange.com/questions/455149/interpolate-irregularly-sampled-data-to-a-regular-grid

    Args:
    dA: xarray dataarray with 2d coordinates of "lat" and "lon" in wgs84
    and dims of "x" and "y".

    Returns:
    dA: xarray datarray with 1d coordinates "lon" and "lat" and dims "x" and "y".
    """
    dA = dA.rio.write_crs('EPSG:4326')
    xg, yg = np.meshgrid(np.linspace(dA.lon.min(), dA.lon.max(), dA.x.size), np.linspace(dA.lat.min(), dA.lat.max(), dA.y.size))
    points = (dA.lon.data.ravel(), dA.lat.data.ravel())
    data = griddata(points, dA.data.ravel(), (xg, yg), method = 'cubic', fill_value = np.nan)
    dA = xr.DataArray(data = data,
                            dims = ["y" ,"x"],
                            coords = {
                "lat": (["y"], np.linspace(dA.lat.min(), dA.lat.max(), dA.y.size)),
                "lon": (["x"] ,np.linspace(dA.lon.min(), dA.lon.max(), dA.x.size))})
    dA = dA.rio.write_crs('EPSG:4326')
    return dA