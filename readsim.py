#!/usr/bin/env python
# -*- coding: utf-8 -*-

#Author: David Leutwyler, Marina Dütsch, Mathias Hauser
#Date: March 2015

import numpy as np
import sys
from namelist import mtn_topo, h_ratio, w_ratio, leeHill_rel


class dotdict(dict):

    """dot.notation access to dictionary attributes"""

    def __getattr__(self, attr):
        return self.get(attr)
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

# -----------------------------------------------------------------------------


def readsim(filename, varnames):

    if isinstance(varnames, str):
        varnames = [varnames]

    # variables to load
    variables = ('u00', 'th00', 'thl', 'dx', 'nz', 'nx', 'time', 'topomx',
                 'topowd', 'height', 'x')

    # Display accumulated precipitation as well
    if 'specific_rain_water_content' in varnames:
        varnames.append('accumulated_precipitation')

    # allows to access var.xp
    var = dotdict()
    # with automatically closes the file
    with np.load(filename) as data:
        for variable in variables:
            var[variable] = data[variable]

        for varname in varnames:
            try:
                var[varname] = data[varname]
            except:
                sys.exit("Variable not in NetCDF File or wrong timestep passed")

    if 'specific_rain_water_content' in varnames:
        varnames.remove('accumulated_precipitation')

    # convert
    var.dx = var.dx / 1000.
    var.topomx = var.topomx / 1000.
    var.topowd = var.topowd / 1000.
    var.zp = var.height / 1000.

    var.xp = np.zeros(shape=var.zp.shape[-2:])
    var.xp[:, :] = var.x[np.newaxis, :]                           # Add an Axis

    # Create Topography
    var.topo = np.zeros(var.nx)
    x = np.arange(var.nx, dtype='float32')
    
    if mtn_topo==1: # lee mtn
        x0 = (var.nx - 1)/2. + 1
        x1 = (var.nx - 1)*leeHill_rel + 1
        x_lee = (x+1 - x1)*var.dx
        x = (x+1 - x0)*var.dx
        
        toponf_main = var.topomx*np.exp(-(x/float(var.topowd))**2)
        toponf_lee = var.topomx*h_ratio*np.exp(-(x_lee/float(var.topowd*w_ratio))**2)
        toponf = np.where(toponf_main>toponf_lee,toponf_main,toponf_lee)

    elif mtn_topo==2: # downstream mtn
        x0 = (var.nx - 1)/2. + 1
        x1 = (var.nx - 1)*(leeHill_rel+5/32) + 1
        x_lee = (x+1 - x1)*var.dx
        x = (x+1 - x0)*var.dx
        
        toponf_main = var.topomx*np.exp(-(x/float(var.topowd))**2)
        toponf_lee = var.topomx*h_ratio*np.exp(-(x_lee/float(var.topowd*w_ratio))**2)
        toponf = np.where(toponf_main>toponf_lee,toponf_main,toponf_lee)
    
    elif mtn_topo==3: # Witch of Agnesi mtn
        x0 = (var.nx - 1)/2. + 1
        x = (x+1 - x0)*var.dx
        
        toponf = var.topomx * var.topowd**2 / (x**2 + float(var.topowd)**2)
        
    elif mtn_topo==4: # Witch of Agnesi mtn with lee hill
        x0 = (var.nx - 1)/2. + 1
        x1 = (var.nx - 1)*leeHill_rel + 1
        x_lee = (x+1 - x1)*var.dx
        x = (x+1 - x0)*var.dx
        
        toponf_main = var.topomx * var.topowd**2 / (x**2 + float(var.topowd)**2)
        toponf_lee = var.topomx*h_ratio * float(var.topowd*w_ratio)**2 / (x_lee**2 + float(var.topowd*w_ratio)**2)
        toponf = np.where(toponf_main>toponf_lee,toponf_main,toponf_lee)
        
    else:
        x0 = (var.nx - 1) / 2. + 1
        x = (x + 1 - x0) * var.dx
        toponf = var.topomx * np.exp(-(x / float(var.topowd)) ** 2)
        
    var.topo[1:-1] = toponf[1:-1] + 0.25 * (toponf[0:-2] - 2. * toponf[1:-1] +
                                            toponf[2:])

    # calculate theta levels
    var.dth = var.thl / var.nz
    theta1d = np.arange(var.nz) * var.dth + var.th00 + var.dth / 2.
    var.theta = np.zeros(shape=var.zp.shape[-2:])
    var.theta[:, :] = theta1d[:, np.newaxis]

    return var
