from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray as rxa
import matplotlib.pyplot as plt

def convert_pressure_to_pascals(dataset: xr.Dataset) -> xr.Dataset:

    if dataset['level'].attrs['units'] != 'pascals':
        dataset['level'] = dataset['level'] * 100
        dataset['level'].attrs['units'] = 'pascals'

    else:
        print("Pressure levels already converted to pascals")
    return dataset

def get_vapour_partial_pressure(dataset: xr.Dataset):
    Rv = 461.495
    Rd = 287.05
    alpha = Rv/Rd
    if dataset['q'].attrs['standard_name'] == 'specific_humidity':
        # specific humidity calculation of partial pressure of water vapour
        dataset['vpr'] = dataset['q']*dataset['level']*alpha/(1+(alpha - 1)*dataset['q'])

    return dataset