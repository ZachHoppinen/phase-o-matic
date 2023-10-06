import numpy as np
import pandas as pd
import xarray as xr
import rioxarray as rxa
from shapely.geometry import box, Polygon
from pathlib import Path
from typing import Union

from phase_o_matic.download import download_era
from phase_o_matic.preprocess import get_vapor_partial_pressure, convert_pressure_to_pascals, interpolate_to_heights, geopotential_to_geopotential_heights
from phase_o_matic.phase_delay import calculate_refractive_indexes, get_delay

def presto_phase_delay(date: pd.Timestamp, 
                       dem: xr.DataArray, 
                       inc: xr.DataArray, 
                       work_dir = Path, 
                       subset: Union[None, Polygon] = None, 
                       wavelength : float = np.pi * 4, 
                       out_name = Union[str, None]) -> xr.Dataset:
    """
    Wrapping function to get dataset of phase delay for dem over a specific date.

    date: pd.Timestamp or strftime string of date to get phase delay on
    dem: xarray dataArray of elevations with coords for "latitude" and "longitude".
    inc: xarray dataArray of incidence angles with coords for "latitude" and "longitude"
        in radians or a single float to use over the image
    work_dir: pathlib Path or string of filepath to directory to era5 files
    subset: None - uses the boundary of dem to get phase delay or shapely Polygon to get subset of overall dem
    wavelength: either default 4 * pi (returns meters of delay) or radar wavelength in meters (returns radians)
    out_name: if None won't save. otherwise will save in work_dir as {out_name}.nc
    """

    # error checking on arguments
    if not isinstance(date, pd.Timestamp):
        try:
            date = pd.to_datetime(date)
        except:
            raise ValueError(f"Unable to parse date string {date} to pandas timestamp.")

    assert isinstance(dem, xr.DataArray), "Second positional argument dem must be a xr.DataArray"
    assert isinstance(inc, xr.DataArray), "third positional argument inc must be a xr.DataArray"

    for geom_ds, name in zip([dem, inc], ['dem', 'inc']):
        for old_name, coord in zip(['y', 'x'], ['latitude', 'longitude']):
            assert coord in geom_ds.coords, f"{name} xarray must use coordinate names latitude. Currently "\
                f"has coords {geom_ds.coords._names}. Consider using the xarray dataarray method:  "\
                f"{name}_ds.rename(dict(old_{old_name}_coord_name = '{coord}'))"
    
    if not isinstance(work_dir, Path):
        if isinstance(work_dir, str):
            work_dir = Path(work_dir)
        else:
            raise TypeError(f"Work dir: {work_dir} of type {type(work_dir)} is not \
                            a Pathlib path or string")

    work_dir = work_dir.expanduser().resolve()
    
    assert work_dir.exists(), f"Working directory {work_dir} does not exist."
    assert work_dir.is_dir(), f"Working directory {work_dir} is not a directory."

    assert isinstance(wavelength, float), "Wavelength must be a float. Provided: {wavelength}"

    # done error checking on arguments

    if not isinstance(subset, Polygon):
        subset = box(*dem.rio.bounds())

    # download ERA5 images to work directory subdirectory
    work_dir.joinpath('ERA5').mkdir(exist_ok = True)

    era_fp = download_era(date, out_dir = work_dir.joinpath('ERA5'), subset = subset, humid_param = 'specific_humidity')

    # open era5 
    era = xr.open_dataset(era_fp)

    # convert to pascals
    era = convert_pressure_to_pascals(era)

    # get vapor pressure data from humidity
    era = get_vapor_partial_pressure(era)

    # convert geopotential 'z' to geopotential heights 'gph'
    era = geopotential_to_geopotential_heights(era)

    # interpolate from pressure levels as coordinate of geopotential heights
    # to geopotential heights as coordinate of other variables
    ds = interpolate_to_heights(era, min_alt = dem.min(), max_alt = dem.max())

    # calculate refractive indexes (N, N_dry, N_wet)
    ds = calculate_refractive_indexes(ds)

    # get the delay in radians or meters
    ds = get_delay(ds, dem, inc, wavelength = wavelength)

    # save output if we have a provided out_name
    if isinstance(out_name, str):
        out_fp = work_dir.joinpath(out_name).with_suffix('.nc')
        
        # add in check/warning for existence and overwrite later
        if out_fp.exists():
            pass
        else:
            ds.to_netcdf(out_fp)

    return ds