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
    interpolate_to_heights, geopotential_to_geopotential_heights, cc_era

class TestPreprocess(unittest.TestCase):
    """
    Test functionality of preprocessing datasets
    """
    np.random.seed(1)
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
        Test converting to pascals from bars
        """
        original = test_ds.copy(deep = True)

        test_ds['level'] = test_ds['level'].assign_attrs(units = 'millibars')

        ds = convert_pressure_to_pascals(test_ds)

        self.assertEqual(ds['level'].units, 'pascals')

        assert_allclose(ds['level'], original['level']*100)

        # check to be sure it won't double convert
        ds = convert_pressure_to_pascals(ds)

        assert_allclose(ds['level'], original['level']*100, err_msg='Failed on double conversion')
        self.assertEqual(ds['level'].units, 'pascals')
        self.assertEqual(ds['level'].long_name, 'Pressure Level')

    def test_pascal_convert_errors(self, test_ds = test_ds):
        """
        Tests for error checking on pascal convert
        """
        original = test_ds.copy()

        test_ds.level.attrs = {}
        
        self.assertRaises(AssertionError, convert_pressure_to_pascals, test_ds)

        test_ds.level.attrs = {'units':'some other units'}
        self.assertRaises(ValueError, convert_pressure_to_pascals, test_ds)
    
    def test_cc_era(self):
        """
        Test calculation of expected vapor saturation pressure for air at different
        temperatures
        """
        expected = np.array([0.16272, 76.05, 195.87, 603.6, 3510, 10237])

        temps = xr.DataArray([200, 250, 260, 273, 300, 320])

        diff = np.abs(cc_era(temps).data.astype(int) - expected.astype(int))
        # large differences by k = 320 but the absolute value is small at 10kPa
        self.assertTrue(np.all(diff == [0,1,5,1,21,312]))

    def test_vapor_pressure_calc(self, test_ds = test_ds):
        """
        Test for succesful calculation of vapour partial pressure
        from specific humidity
        """
        original = test_ds.copy(deep = True)
        ds = get_vapor_partial_pressure(test_ds)

        self.assertTrue('vpr' in ds.data_vars)

        alpha = 461.495/287.05
        correct_vpr = original['q'] * original['level'] * alpha / (1 + (alpha - 1) * original['q'])
        assert_allclose(ds['vpr'], correct_vpr)

        self.assertEqual(ds['vpr'].units, 'pascals')
        self.assertEqual(ds['vpr'].long_name, 'Vapor Partial Pressure')

        test_ds = original.copy()

        test_ds['r'] = test_ds['q']
        test_ds = test_ds.drop_vars('q')

        ds = get_vapor_partial_pressure(test_ds)
        h20_sat = cc_era(original['t'])
        expected_vapor = h20_sat * original['q']/100

        assert_allclose(expected_vapor, ds['vpr'])

        self.assertEqual(ds['vpr'].units, 'pascals')
        self.assertEqual(ds['vpr'].long_name, 'Vapor Partial Pressure')

        def simplified_q_to_vpr(pt, q):
            return pt / (1 + 0.622/q)
        test = xr.DataArray(np.linspace(3.33583175e-06, 3.14360503e-03, 9), coords = [np.linspace(34, 820, 9)], dims = ['level']).to_dataset(name = 'q')
        assert_allclose(get_vapor_partial_pressure(test)['vpr'].data, simplified_q_to_vpr(test.level.data, test.q.data), atol = 0.1)

        test_ds = original.copy(deep = True)
        for level in test_ds.level:
            assert_allclose(simplified_q_to_vpr(level.data, test_ds.sel(level = level).q.data), get_vapor_partial_pressure(test_ds.sel(level = level))['vpr'].data, rtol = 0.4)
    
    def test_vapor_pressure_errors(self, test_ds = test_ds):
        """
        Tests for error checking on vapor pressure
        """

        original = test_ds.copy()
        test_ds = test_ds.drop_vars('q')
        self.assertRaises(ValueError, get_vapor_partial_pressure, test_ds)

    def test_geopotential_to_geopotential_heights(self, test_ds = test_ds):
        """
        Test for conversions between geopotential to geopotential heights
        """
        test_ds = convert_pressure_to_pascals(test_ds)
        original = test_ds['z'].copy()
        ds = geopotential_to_geopotential_heights(test_ds)

        self.assertTrue('gph' in test_ds.data_vars)

        assert_allclose(original/9.81, ds['gph'])

        self.assertEqual(ds['gph'].units, 'meters')
        self.assertEqual(ds['gph'].long_name, 'Geopotential Height')

    def test_interpolate_to_heights(self, test_ds = test_ds):
        """
        Test for interpolating from pressure levels as coordinate and geopotential 
        height as data variable to geopotential heights as indexing coordinate.
        """
        test_ds = convert_pressure_to_pascals(test_ds)
        test_ds = geopotential_to_geopotential_heights(test_ds)
        test_ds['vpr'] = xr.DataArray(np.random.random((10, 10, 37, 1)),\
                                    coords = [test_ds.longitude, test_ds.latitude, test_ds.level, test_ds.time],\
                                    dims = ['longitude', 'latitude', 'level', 'time'], attrs={'units':'pascals', 'long_name':'Vapor Partial Pressure'})

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
        self.assertEqual(ds.height.min(), -20)
        self.assertAlmostEqual(ds.height.max().data, test_ds['gph'].max().round().data , places = 1)

        ds = interpolate_to_heights(test_ds, min_alt=0.1)
        self.assertAlmostEqual(ds.height.min(), 0.1, places = 1)

        self.assertEqual(ds['height'].units, 'meters')
        self.assertEqual(ds['height'].long_name, 'Geopotential Height')

        self.assertEqual(ds['air_pressure'].units, 'pascals')
        self.assertEqual(ds['air_pressure'].long_name, 'Air Pressure')

        self.assertEqual(ds['vapor_pressure'].units, 'pascals')
        self.assertEqual(ds['vapor_pressure'].long_name, 'Vapor Partial Pressure')

        self.assertEqual(ds['temperature'].units, 'K')
        self.assertEqual(ds['temperature'].long_name, 'Temperature')

        self.assertEqual(ds['longitude'].units, 'degrees_east')
        self.assertEqual(ds['longitude'].long_name, 'longitude')

        self.assertEqual(ds['latitude'].units, 'degrees_north')
        self.assertEqual(ds['latitude'].long_name, 'latitude')

if __name__ == '__main__':
    unittest.main()