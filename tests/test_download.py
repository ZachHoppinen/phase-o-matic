import unittest
from shapely import geometry
import pandas as pd
import xarray as xr
from pathlib import Path

import sys
sys.path.append('.')
from phase_o_matic.download import download_era

class TestEraDownload(unittest.TestCase):
    """
    Test functionality of downloading ERA5 model data
    """

    def test_download(self):
        """
        Test downloading era5 data
        """
        area = geometry.box(-116, 45, -115, 46)
        ts = pd.to_datetime('2020-01-03T20:01:4600')
        out_dir = Path('~/scratch/era5').expanduser()

        out_fp = download_era(date = ts, out_dir = out_dir, subset = area)

        ds = xr.open_dataset(out_fp)

        self.assertEqual(ds.level.size, 37, "levels are not right size")
        self.assertEqual(ds.time[0], pd.to_datetime('2020-01-03T20:00:00.000000000'), "incorrect time data")
        self.assertEqual(ds['level'][10], 100, "spot checking pressure levels")
        self.assertEqual(ds.isel(level = 3)['q'].data.shape, (1, 5, 5), "checking data shape")

        