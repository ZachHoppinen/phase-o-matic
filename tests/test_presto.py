import unittest

from shapely import geometry
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

import sys
sys.path.append('./')
sys.path.append('../')
from phase_o_matic import presto_phase_delay

class TestPresto(unittest.TestCase):
    """
    Test functionality of wrapping functions
    """
    
    dem = xr.DataArray(np.random.random(100).reshape(10,10),
                 coords = [np.arange(10), np.arange(10) + 10],
                 dims = ['latitude','longitude'])

    inc = xr.DataArray(np.random.random(100).reshape(10,10),
                 coords = [np.arange(10), np.arange(10) + 10],
                 dims = ['latitude','longitude'])

    def test_presto_errors(self, inc = inc, dem = dem):
        
        self.assertRaises(AssertionError, presto_phase_delay, '2020-01-30', dem, "", '/tmp')
        self.assertRaises(AssertionError, presto_phase_delay, '2020-01-30', 1, inc, '/tmp')
        self.assertRaises(AssertionError, presto_phase_delay, '2020-01-30', None, inc, '/tmp')
        self.assertRaises(AssertionError, presto_phase_delay, '2020-01-30', np.random.random(10), inc, '/tmp')

        self.assertRaises(AssertionError, presto_phase_delay, '2020-01-30', dem.rename(dict(latitude = 'y')), inc, '/tmp')

        self.assertRaises(AssertionError, presto_phase_delay, '2020-01-30', dem.rename(dict(longitude = 'x')), inc, '/tmp')

        self.assertRaises(AssertionError, presto_phase_delay, '2020-01-30', dem, inc.rename(dict(latitude = 'y')), '/tmp')

        self.assertRaises(AssertionError, presto_phase_delay, '2020-01-30', dem, inc.rename(dict(longitude = 'x')), '/tmp')

        self.assertRaises(ValueError, presto_phase_delay, 'abc', dem, inc, '/tmp')

        self.assertRaises(ValueError, presto_phase_delay, '2020-02-31', dem, inc, '/tmp')

        self.assertRaises(TypeError, presto_phase_delay, '2020-01-30', dem, inc, 10)

        self.assertRaises(AssertionError, presto_phase_delay, '2020-01-30', dem, inc, '/some/imaginary/dir')

        self.assertRaises(AssertionError, presto_phase_delay, '2020-01-30', dem, inc, './test_data/wet_N.npy')

        self.assertRaises(AssertionError, presto_phase_delay, '2020-01-30', dem, inc, '.tests/test_data/wet_N.npy')


