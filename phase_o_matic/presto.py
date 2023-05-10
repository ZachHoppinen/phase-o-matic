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

def presto_phase_delay(date: pd.Timestamp, geometry: xr.DataArray, subset: Union[None, Polygon] = None, work_dir = Path, wavelength : float = np.pi * 4) -> xr.Dataset:

    if not isinstance(subset, Polygon):
            subset = box(*geometry.rio.bounds())

    out_fp = download_era(date, out_dir = work_dir, subset = subset, humid_param = 'specific_humidity')
    era = xr.open_dataset(out_fp)
    era = convert_pressure_to_pascals(era)
    era = get_vapor_partial_pressure(era)
    era = geopotential_to_geopotential_heights(era)
    ds = interpolate_to_heights(era)        
    ds = calculate_refractive_indexes(ds)
    ds = get_delay(ds, geometry, wavelength = 0.238403545)

    return ds