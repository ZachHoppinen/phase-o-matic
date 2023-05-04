import cdsapi
import pandas as pd
import xarray as xr
from shapely import geometry
from pathlib import Path
from typing import Union

def download_era(date: pd.Timestamp, out_dir: Union[str, Path], subset: Union[geometry.Polygon, None]) -> str:
    """
    Download era5 weather model for specific hourly timestep as netcdf. Can be subset to a specific geographic subset.

    Args:
    date: timestamp of the desired date and time. Will be rounded to nearest hour
    out_dir: directory to save file into
    subset: subset geometry to clip data to

    returns:
    out_fp: filepath of saved netcdf
    """
    # check to see if era5 token has been saved to .cdsapric file and create client for download
    assert Path('~/.cdsapirc').expanduser().exists(), "Must sign up for ERA5 account and save key to .cdsapric file"
    c = cdsapi.Client()

    # could get 'relative_humidity' but sticking with one.
    humidparam = 'specific_humidity'

    # pressure levels to download (this is all 37 possible)
    era_pressure_lvls = ['1','2','3','5','7','10','20','30','50', '70','100','125',\
    '150','175','200','225', '250','300','350','400','450','500','550','600','650',\
    '700','750','775','800','825', '850','875','900','925','950','975','1000']

    indict = {'product_type'   :'reanalysis',\
                'format'         :'netcdf',\
                'variable'       :['geopotential','temperature', humidparam],\
                'pressure_level' : era_pressure_lvls, \
                'date'           : date.strftime('%Y-%M-%d'),
                'time'           : date.strftime('%H:00')}

    out_fp = out_dir.joinpath(f"ERA5_{date.strftime('%Y-%M-%dT%H:00')}.nc")

    if subset:
        w, s, e, n = subset.bounds
        indict['area'] = f'/'.join([str(b) for b in [n, w, s, e]])
        out_fp = out_fp.with_stem(f"{out_fp.stem}_{'_'.join([str(b) for b in subset.bounds])}")

    c.retrieve('reanalysis-era5-pressure-levels', indict, target=out_fp)

    return out_fp