# -*- coding: utf-8 -*-
"""
General meteorological utility functions
"""

import numpy as np
import sys

def rrmixv1(p,T,humv,kindhum):

    """
    computes mixing ratio in (g/g)

    Input:      rrmixv1(p,T,humv,kindhum)
                p     ... pressure in [hPa]
                T     ... temperature [K]
                humv  ... humidity variable; one of: dew point [K]
                                                     relative humidity [0-1]
                kindhum . kind of humidity variable: 1 = dew point [K]
                                                     2 = relative humidity [0-1]
    Output:     mixv1
    """

    #Define local constant
    eps= 0.62198

    if kindhum == 1:      # dew point is the input
        esat = eswat1(humv)
        mixv1 = eps*esat/(p - esat)
    elif kindhum == 2:     # relative humidity is the input
        esat  = eswat1(T)

    if kindhum == 1 or kindhum == 2:
        mixv1=np.where(esat >= 0.616*p,0.,eps*humv*esat/(p - humv*esat))
    else:
        sys.exit("kindhum needs to be 1 or 2")

    return mixv1

def eswat1(T):
    """
    Saturation vapor pressure over water, Goff-Gratch formulation
    (based on exact integration of Clausius-Clapeyron equation)

    Input:      T     ... temperature [K]
    Output:     mixv1
    """

    #Define local constants
    C1=7.90298
    C2=5.02808
    C3=1.3816e-7
    C4=11.344
    C5=8.1328e-3
    C6=3.49149

    RMIXV  = 373.16 / T

    ES     = - C1*( RMIXV - 1) + C2*np.log10( RMIXV ) - \
               C3*( 10**( C4*(1 - 1/RMIXV)) - 1 ) + \
               C5*( 10**( - C6*(RMIXV - 1)) - 1 )

    eswat  = 1013.246 * 10**ES
    return eswat

# END OF METEO_UTILITIES.PY