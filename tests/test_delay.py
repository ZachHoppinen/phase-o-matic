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
from phase_o_matic.phase_delay import calculate_dry_refractivity, calculate_wet_refractivity,\
    calculate_refractive_indexes, get_delay

class TestDelay(unittest.TestCase):
    """
    Test functionality of calculating refractive indexes and delay of radar signals
    """
    np.random.seed(1)

    heights = np.linspace(-200, 40000, 300)
    lats = np.linspace(43, 45, 10).astype('float32')
    lons = np.linspace(-116, -115, 13).astype('float32')
    test_ds = xr.Dataset(
        {
            'air_pressure': (['time','latitude', 'longitude','height'], 100*np.random.random((1, 10, 13, 300)).astype('float32'), {'units':'pascals', 'long_name':'pressure_level'}),
            'temperature': (['time','latitude', 'longitude','height'], np.random.random((1, 10, 13, 300)).astype('float32'), {'units' :'K', 'long_name' :'Temperature', 'standard_name' :'air_temperature'}),
            'vapor_pressure': (['time','latitude', 'longitude','height'], np.random.random((1, 10, 13, 300)).astype('float32'), {'units': 'pascals'}),
        },  
        coords = {
            "longitude" : (["longitude"], lons, {'units' :'degrees_east', 'long_name' :'longitude'}),
            "latitude" : (["latitude"], lats, {'units' :'degrees_north', 'long_name' :'latitude'}),
            "height" : (["height"], heights, {'units' : 'meters', 'long_name' :'geopotential_heights'}),
            "time" : (["time"], [pd.to_datetime('2020-01-04T09:00')], {'long_name' :'time'}),
        },
        attrs={'Conventions' :'CF-1.6', 'history' :'2023-05-04 17:03:14 GMT by grib_'}
    )

    def test_dry_refractivity(self, test_ds = test_ds):
        """
        Testing for dry refractivity
        """

        original = test_ds.copy()

        test_ds = calculate_dry_refractivity(test_ds)

        self.assertTrue('N_dry' in test_ds.data_vars)

        assert_allclose(test_ds['N_dry'], 1.0e-6 * 0.776 * 287.05 / 9.81 * (original['air_pressure']  - original['air_pressure'].isel(height = -1)))

        test_ds = original.copy()
        test_ds = test_ds.interp(height = np.linspace(-100, 20000, 200))
        original = test_ds.copy()
        test_ds = calculate_dry_refractivity(test_ds)

        assert_allclose(test_ds['N_dry'], 1.0e-6 * 0.776 * 287.05 / 9.81 * (original['air_pressure']  - original['air_pressure'].isel(height = -1)))

        test_ds = original.copy()
        test_ds = test_ds.interp(height = np.linspace(20000, -100, 200))
        original = test_ds.copy()
        test_ds = calculate_dry_refractivity(test_ds)

        original = original.reindex(height=list(reversed(original.height)))
        assert_allclose(test_ds['N_dry'], 1.0e-6 * 0.776 * 287.05 / 9.81 * (original['air_pressure']  - original['air_pressure'].isel(height = -1)))

        self.assertEqual(test_ds['N_dry'].units, '')
        self.assertEqual(test_ds['N_dry'].long_name, 'Hydrostatic Refractivity')

    def test_dry_refractivity_errors(self, test_ds = test_ds):
        """
        Test for correct errors in dry refractivity
        """

        test_ds['air_pressure'] = test_ds['air_pressure'].assign_attrs({'units':'bars'})

        self.assertRaises(AssertionError, calculate_dry_refractivity, test_ds)

        test_ds['air_pressure'] = test_ds['air_pressure'].assign_attrs({'units':'pascals'})
        test_ds['temperature'] = test_ds['temperature'].assign_attrs({'units':'C'})

        self.assertRaises(AssertionError, calculate_dry_refractivity, test_ds)

        test_ds['air_pressure'].attrs = {}
        test_ds['temperature'].attrs = {}
        try:
            calculate_dry_refractivity(test_ds)
        except AssertionError:
            self.fail(f"Calculate wet refractivity raised exception with empty attrs.")
    
    def test_wet_refractivity(self, test_ds = test_ds):
        """
        Testing for wet refractivity calculations
        """

        original = test_ds.copy()

        test_ds = calculate_wet_refractivity(test_ds)

        self.assertTrue('N_wet' in test_ds.data_vars)

        if Path('./tests/test_data/wet_N_seed_1.npy').exists():
            N_wet_correct = np.load('./tests/test_data/wet_N_seed_11.npy')
            assert_allclose(test_ds['N_wet'].data, N_wet_correct)
        elif Path('./test_data/wet_N_seed_1.npy').exists():
            N_wet_correct = np.load('./test_data/wet_N_seed_11.npy')
            assert_allclose(test_ds['N_wet'].data, N_wet_correct)

        self.assertEqual(test_ds['N_wet'].units, '')
        self.assertEqual(test_ds['N_wet'].long_name, 'Water Vapor Refractivity')
    
    def test_wet_refractivity_errors(self, test_ds = test_ds):
        """
        Test for correct errors in wet refractivity
        """

        test_ds['vapor_pressure'] = test_ds['vapor_pressure'].assign_attrs({'units':'bars'})

        self.assertRaises(AssertionError, calculate_wet_refractivity, test_ds)

        test_ds['vapor_pressure'] = test_ds['vapor_pressure'].assign_attrs({'units':'pascals'})
        test_ds['temperature'] = test_ds['temperature'].assign_attrs({'units':'C'})

        self.assertRaises(AssertionError, calculate_wet_refractivity, test_ds)

        test_ds['vapor_pressure'].attrs = {}
        test_ds['temperature'].attrs = {}
        try:
            calculate_wet_refractivity(test_ds)
        except AssertionError:
            self.fail(f"Calculate wet refractivity raised exception with empty attrs.")
    
    def test_combo_refractivity(self, test_ds = test_ds):
        """
        Test for calculating dry and wet refractivities together
        """

        original = test_ds.copy()

        test_ds = calculate_refractive_indexes(test_ds)

        self.assertTrue('N_dry' in test_ds.data_vars)
        self.assertTrue('N_wet' in test_ds.data_vars)
        self.assertTrue('N' in test_ds.data_vars)

        N_dry_correct = 1.0e-6 * 0.776 * 287.05 / 9.81 * (original['air_pressure']  - original['air_pressure'].isel(height = -1))
        assert_allclose(test_ds['N_dry'], N_dry_correct)

        directory = Path(__file__).parent.resolve()

        if directory.joinpath('test_data/wet_N_seed_1.npy').exists():
            N_wet_correct = np.load('./tests/test_data/wet_N_seed_11.npy')
            assert_allclose(test_ds['N_wet'].data, N_wet_correct)
        
        if "N_wet_correct" in locals():
            assert_allclose(test_ds['N'].data, N_wet_correct + N_dry_correct)
        
        self.assertEqual(test_ds['N'].units, '')
        self.assertEqual(test_ds['N'].long_name, 'Total Refractivity')
    
    dem = xr.DataArray(np.linspace(1000, 3000, 200*300).reshape(200, 300),
                    coords = [np.linspace(43, 45, 200), np.linspace(-116, -115, 300)],
                    dims = ['latitude','longitude']
    )

    inc = xr.DataArray(np.linspace(0, np.deg2rad(90), 200*300).reshape(200, 300),
                    coords = [np.linspace(43, 45, 200), np.linspace(-116, -115, 300)],
                    dims = ['latitude','longitude']
    )

    def test_get_delay(self, test_ds = test_ds, dem = dem, inc = inc):
        """
        Test for getting radar delay across DEM from atmospheric refractive index
        """
        test_ds = calculate_refractive_indexes(test_ds)
        original = test_ds.copy()

        test_ds = get_delay(test_ds, dem, inc)

        self.assertTrue('delay' in test_ds.data_vars)

        self.assertEqual(test_ds['delay'].attrs['units'], 'meters')

        directory = Path(__file__).parent.resolve()
        if directory.joinpath('test_data/delay_seed_1.npy').exists():
            delay_correct = np.load('./tests/test_data/delay_seed_1.npy')
            assert_allclose(test_ds['delay'].data, delay_correct)
        
        # failure here suggests we didn't align to the DEM correctly
        try:
            xr.align(dem, test_ds, join='exact')
        except:
            self.fail('DEM and test_ds not aligned after getting delay')
        
        test_ds = original.copy()
        try:
            test_ds = get_delay(test_ds, dem, inc = np.pi)
        except:
            self.fail('Failed using float as incidence angle')
        assert_allclose(test_ds['delay'].data, np.cos(np.expand_dims(inc.data, 0)) * delay_correct / np.cos(np.pi))
        

        test_ds = original.copy()
        try:
            test_ds = get_delay(test_ds, dem, inc = inc, wavelength = 10)
        except:
            self.fail('Failed using non default wavelength')
        
        self.assertEqual(test_ds['delay'].attrs['units'], 'radians')

        assert_allclose(test_ds['delay'].data, 4 * np.pi * delay_correct / 10)

        self.assertEqual(test_ds['delay'].long_name, 'Atmospheric Delay')
        
        
    def test_get_delay_errors(self, test_ds = test_ds, dem = dem, inc = inc):
        """
        Test for corret errors through from delay function
        """

        test_ds = test_ds.drop_vars('N')
        self.assertRaises(AssertionError, get_delay, test_ds, dem, inc)
        
        test_ds = calculate_refractive_indexes(test_ds)

        inc_to_high = inc * 100
        
        self.assertRaises(AssertionError, get_delay, test_ds, dem, inc_to_high)
        self.assertRaises(AssertionError, get_delay, test_ds, dem, 9)

        self.assertRaises(AssertionError, get_delay, test_ds, dem, inc, -1)

        inc_original = inc.copy()
        inc['latitude'] = inc['latitude'] + 100
        self.assertRaises(AssertionError, get_delay, test_ds, dem, inc)

        inc = inc_original.copy()
        inc['longitude'] = inc['longitude'] - 100
        self.assertRaises(AssertionError, get_delay, test_ds, dem, inc)

        inc = inc_original.copy()
        dem_high = dem*10000
        self.assertRaises(AssertionError, get_delay, test_ds, dem_high, inc)

        dem_low = dem-10000
        self.assertRaises(AssertionError, get_delay, test_ds, dem_low, inc)