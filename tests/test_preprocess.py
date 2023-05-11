import unittest
from numpy.testing import assert_allclose

from shapely import geometry
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

import sys
sys.path.append('./')
sys.path.append('../')
from phase_o_matic.preprocess import convert_pressure_to_pascals, get_vapor_partial_pressure,\
    interpolate_to_heights, geopotential_to_geopotential_heights, check_start_end

class TestPreprocess(unittest.TestCase):
    """
    Test functionality of downloading ERA5 model data
    """

    levels = ['1','2','3','5','7','10','20','30','50', '70','100','125',\
        '150','175','200','225', '250','300','350','400','450','500','550','600','650',\
        '700','750','775','800','825', '850','875','900','925','950','975','1000']
    levels = np.array(levels).astype('int32')
    lats = np.linspace(43, 45, 10).astype('float32')
    lons = np.linspace(-116, -115, 10).astype('float32')
    test_ds = xr.Dataset(
        {
            'z': (['longitude', 'latitude','level','time'], 100*np.random.random((10, 10, 37, 1)).astype('float32'), {'units':'m**2 s**-2', 'long_name':'Geopotential', 'standard_name' :'geopotential'}),
            't': (['longitude', 'latitude','level','time'], np.random.random((10, 10, 37, 1)).astype('float32'), {'units' :'K', 'long_name' :'Temperature', 'standard_name' :'air_temperature'}),
            'q': (['longitude', 'latitude','level','time'], np.random.random((10, 10, 37, 1)).astype('float32'), {'units' :'kg kg**-1', 'long_name' :'Specific humidity', 'standard_name' :'specific_humidity'}),
        },  
        coords = {
            "longitude" : (["longitude"], lons, {'units' :'degrees_east', 'long_name' :'longitude'}),
            "latitude" : (["latitude"], lats, {'units' :'degrees_north', 'long_name' :'latitude'}),
            "level" : (["level"], levels, {'units' :'millibars', 'long_name' :'pressure_level'}),
            "time" : (["time"], [pd.to_datetime('2020-01-04T09:00')], {'long_name' :'time'}),
        },
        attrs={'Conventions' :'CF-1.6', 'history' :'2023-05-04 17:03:14 GMT by grib_'}
    )

    def test_pascal_convert(self, test_ds = test_ds):
        """
        Test downloading era5 data
        """
        original = test_ds.copy(deep = True)
        ds = convert_pressure_to_pascals(test_ds)

        self.assertEqual(ds['level'].units, 'pascals')

        assert_allclose(ds['level'], original['level']*100)

        # check to be sure it won't double convert
        ds = convert_pressure_to_pascals(ds)

        assert_allclose(ds['level'], original['level']*100)
        self.assertEqual(ds['level'].units, 'pascals')
    
    def test_vapor_pressure_calc(self, test_ds = test_ds):
        """
        Tests for succesful calculation of vapour partial pressure
        from specific humidity
        """
        original = test_ds.copy(deep = True)
        ds = get_vapor_partial_pressure(test_ds)

        self.assertTrue('vpr' in ds.data_vars)

        alpha = 461.495/287.05
        correct_vpr = original['q'] * original['level'] * alpha / (1 + (alpha - 1) * original['q'])
        assert_allclose(ds['vpr'], correct_vpr)

        ds['q'] = ds['q'].assign_attrs(standard_name = 'relative_humidity')
    
    def test_interpolate_to_heights(self, test_ds = test_ds):
        test_ds = convert_pressure_to_pascals(test_ds)
        test_ds = geopotential_to_geopotential_heights(test_ds)
        test_ds['vpr'] = xr.DataArray(np.random.random((10, 10, 37, 1)),\
                                    coords = [test_ds.longitude, test_ds.latitude, test_ds.level, test_ds.time],\
                                    dims = ['longitude', 'latitude', 'level', 'time'])

        original = test_ds.copy()
        ds = interpolate_to_heights(test_ds, n_heights = 300)

        self.assertTrue('height' in ds.coords)
        self.assertEqual(ds['height'].size, 300)

        ds = interpolate_to_heights(test_ds, n_heights = 200)

        self.assertTrue('height' in ds.coords)
        self.assertEqual(ds['height'].size, 200)

        for coords in ['latitude', 'longitude']:
            assert_allclose(ds[coords], original[coords], err_msg=f"Failed on coordinate compare: {coords}")
        
        self.assertTrue('air_pressure' in ds.data_vars)
        self.assertTrue('vapor_pressure' in ds.data_vars)
        self.assertTrue('temperature' in ds.data_vars)

        test_ds = original.isel(latitude= slice(0, 4), longitude = slice(4, 6))
        ds = interpolate_to_heights(test_ds)

        for coords in ['latitude', 'longitude']:
            assert_allclose(ds[coords], test_ds[coords], err_msg=f"Failed on coordinate compare: {coords}")
        
        ds = interpolate_to_heights(test_ds, min_alt=-20)
        self.assertEquals(ds.height.min(), -20)
        self.assertAlmostEquals(ds.height.max().data, test_ds['gph'].max().round().data , places = 1)

        ds = interpolate_to_heights(test_ds, min_alt=0.1)
        self.assertAlmostEquals(ds.height.min(), 0.1, places = 1)

        

if __name__ == '__main__':
    unittest.main()