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

# Non-functional more xarray-onic method
# from scipy.interpolate import griddata
# import numpy as np
# import xarray as xr

# dem = pa.utils.read_data(os.path.join(data_dir, 'hgt.rdr'))
# dem = xr.DataArray(dem, 
#              coords = {
#     "lat": (("y", "x"), lat),
#     "lon": (("y", "x"), lon)
# },
# dims = ["y","x"])

# from scipy.interpolate import griddata
# import numpy as np
# import xarray as xr

# def interp_to_grid(u, xc, yc, new_lats, new_lons):
#     new_points = np.stack(np.meshgrid(new_lats, new_lons), axis = 2).reshape((new_lats.size * new_lons.size, 2))
#     z = griddata((xc, yc), u, (new_points[:,1], new_points[:,0]), method = 'nearest', fill_value = np.nan)
#     out = z.reshape((new_lats.size, new_lons.size), order = "F")
#     return out


# values = dem.data.ravel()
# lons = dem.lon.data.ravel()
# lats = dem.lat.data.ravel()
# _new_lats = np.linspace(dem.lat.min(), dem.lat.max(), dem.y.size)
# _new_lons = np.linspace(dem.lon.min(), dem.lon.max(), dem.x.size)
# new_lats = xr.DataArray(_new_lats, dims = "lat", coords = {"lat": _new_lats})
# new_lons = xr.DataArray(_new_lons, dims = "lon", coords = {"lon": _new_lons})

# gridded_ds = xr.apply_ufunc(interp_to_grid,
#                      values, lons, lats, new_lats, new_lons,
#                      input_core_dims = [[],[],[],["lat"],["lon"]],
#                      output_core_dims = [['y', 'x']],
#                      )
