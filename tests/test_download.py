import unittest
from shapely import geometry
import pandas as pd
import xarray as xr
from pathlib import Path

import sys
sys.path.append('./')
sys.path.append('../')
from phase_o_matic.download import download_era

class TestEraDownload(unittest.TestCase):
    """
    Test functionality of downloading ERA5 model data
    """

    params = [geometry.box(-116, 45, -115, 46), pd.to_datetime('2020-01-03T20:01:0000'), Path('/tmp')]

    def test_download(self, params = params):
        """
        Test downloading era5 data. We only try once to avoid slowing things down.
        """
        area, ts, out_dir = params

        out_fp = download_era(date = ts, out_dir = out_dir, subset = area, humid_param ='specific_humidity')

        ds = xr.open_dataset(out_fp)

        self.assertEqual(ds.level.size, 37, "levels are not right size")
        self.assertEqual(ds.time[0], pd.to_datetime('2020-01-03T20:00:00.000000000'), "incorrect time data")
        self.assertEqual(ds['level'][10], 100, "spot checking pressure levels")
        self.assertTrue('z' in ds.data_vars, "geopotential heights present")
        self.assertTrue('t' in ds.data_vars, "Temperature present")
        self.assertTrue('q' in ds.data_vars, "Specific humidity present")
        self.assertEqual(ds['z'].data.shape, (1, 37, 5, 5), "checking data shape")

        try:
            out_fp = download_era(date = ts, out_dir = out_dir, subset = area, humid_param ='specific_humidity')
        except:
            self.fail('download failed with exisiting file.')
        
        out_fp = download_era(date = ts, out_dir = out_dir, subset = area, humid_param ='relative_humidity', LOS = True)
        ds = xr.open_dataset(out_fp)

        self.assertEqual(ds.level.size, 37, "levels are not right size")
        self.assertEqual(ds.time[0], pd.to_datetime('2020-01-03T20:00:00.000000000'), "incorrect time data")
        self.assertEqual(ds['level'][10], 100, "spot checking pressure levels")
        self.assertTrue('z' in ds.data_vars, "geopotential heights present")
        self.assertTrue('t' in ds.data_vars, "Temperature present")
        self.assertTrue('r' in ds.data_vars, "relative humidity present")
        self.assertEqual(ds['z'].data.shape, (1, 37, 17, 17), "checking data shape for LOS correction")
    
    def test_invalid_inputs(self, params = params):
        """
        Test for invalid user inputs
        """
        area, ts, out_dir = params

        # non existent directory
        self.assertRaises(AssertionError, download_era, ts, Path('/some/imginary/dir'), area)
        self.assertRaises(AssertionError, download_era, ts, '/some/imginary/dir', area)

        # date in the future
        self.assertRaises(AssertionError, download_era, pd.to_datetime('2024-01-01'), out_dir, area)
        self.assertRaises(AssertionError, download_era, '2024-01-01', out_dir, area)

        # subset outside globe
        self.assertRaises(AssertionError, download_era, ts, out_dir, geometry.box(-191, 45, -115, 46))
        self.assertRaises(AssertionError, download_era, ts, out_dir, geometry.box(-189, 45, -115, -93))
        self.assertRaises(AssertionError, download_era, ts, out_dir, geometry.box(-189, 45, -115, 96))


if __name__ == '__main__':
    unittest.main()