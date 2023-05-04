def initconst():
    '''Initialization of various constants needed for computing delay.
    from: https://github.com/insarlab/PyAPS/blob/244552cdfcf4e1a55de5f1439be4f08eb45872ec/src/pyaps3/processor.py#L16

    Args:
        * None

    Returns:
        * constdict (dict): Dictionary of constants
    '''

    constdict = {}
    constdict['k1'] = 0.776   #(K/Pa)
    constdict['k2'] = 0.716   #(K/Pa)
    constdict['k3'] = 3750    #(K^2.Pa)
    constdict['g'] = 9.81     #(m/s^2)
    constdict['Rd'] = 287.05  #(J/Kg/K)
    constdict['Rv'] = 461.495 #(J/Kg/K)
    constdict['mma'] = 29.97  #(g/mol)
    constdict['mmH'] = 2.0158 #(g/mol)
    constdict['mmO'] = 16.0   #(g/mol)
    constdict['Rho'] = 1000.0 #(kg/m^3)

    constdict['a1w'] = 611.21 # hPa
    constdict['a3w'] = 17.502 #
    constdict['a4w'] = 32.19  # K
    constdict['a1i'] = 611.21 # hPa
    constdict['a3i'] = 22.587 #
    constdict['a4i'] = -0.7   # K
    constdict['T3'] = 273.16  # K
    constdict['Ti'] = 250.16  # K
    constdict['nhgt'] = 300   # Number of levels for interpolation (OLD 151)
    constdict['minAlt'] = -200.0
    constdict['maxAlt'] = 50000.0
    constdict['minAltP'] = -200.0
    return constdict

#############Clausis-Clayperon for ECMWF###########################
def cc_era(tmp,cdic):
    '''Clausius Clayperon law used by ERA Interim.
    https://github.com/insarlab/PyAPS/blob/244552cdfcf4e1a55de5f1439be4f08eb45872ec/src/pyaps3/era.py#LL13C1-L46C16

    Args:
        * tmp  (np.ndarray) : Temperature.
        * cdic (dict)       : Dictionary of constants

    Returns:
        * esat (np.ndarray) : Water vapor saturation partial pressure.'''


    a1w = cdic['a1w']
    a3w = cdic['a3w']
    a4w = cdic['a4w']
    a1i = cdic['a1i']
    a3i = cdic['a3i']
    a4i = cdic['a4i']
    T3  = cdic['T3']
    Ti  = cdic['Ti'] 

    esatw = a1w*np.exp(a3w*(tmp-T3)/(tmp-a4w))
    esati = a1i*np.exp(a3i*(tmp-T3)/(tmp-a4i))
    esat = esati.copy()
    for k in range(len(tmp)):
        if (tmp[k] >= T3):
            esat[k] = esatw[k]
        elif (tmp[k] <= Ti):
            esat[k] = esati[k]
        else:
            wgt = (tmp[k]-Ti)/(T3-Ti)
            esat[k] = esati[k] + (esatw[k]-esati[k])*wgt*wgt

    return esat

#############Clausius-Clapeyron for ECMWF as used in Jolivet et al 2011#
def cc_eraorig(tmp,cdic):
    '''This routine takes temperature profiles and returns Saturation water vapor
    partial pressure using the Clausius-Clapeyron law applied in Jolivet et al. 2011,
    GRL, doi:10.1029/2011GL048757. It can be used in case you are using Relative.
    Humidity from ECMWF models.

    from: https://github.com/insarlab/PyAPS/blob/244552cdfcf4e1a55de5f1439be4f08eb45872ec/src/pyaps3/delay.py#LL11C1-L32C16

    Args:
        * tmp  (np.ndarray) : Temperature
        * cdic (dict)       : Dictionnary of constants

    Returns:
        * esat (np.ndarray) : Saturation water vapor partial pressure.
    '''

    a1w=cdic['a1w']
    T3=cdic['T3']
    Rv=cdic['Rv']

    esat=a1w*np.exp( (2.5e6/Rv) * ( (1/T3) - (1/tmp) ) )

    return esat
