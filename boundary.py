# -*- coding: utf-8 -*-
import numpy as np


def periodic(phi,nx,nb):
    """
    This subroutine makes the array phi(n1,n2) periodic. At the left
    and right border the number of 'nb' points is overwritten. The
    periodicity of this operation is 'nx'.
    Based on periodic.m from the full isentropic model in MATLAB, 2014.

    Input:  periodic(phi,nx,nb)
    Output: phi
    """
    phi[0:nb,:] = phi[nx:(nb+nx),:]
    phi[(nb+nx):(nx+2*nb),:] = phi[nb:(2*nb),:]

    return phi


def relax(phi,nx,nb,phi1,phi2):
    """
    Relaxation of boundary conditions.
    Based on relax.m from the full isentropic model in MATLAB, 2014.

    Input:  relax(phi,nx,nb,phi1,phi2)
    Output: phi
    """

    # Relaxation is done over nr grid points
    nr = 8
    n = 2*nb + nx

    # initialize relaxation array
    rel = np.array([1, 0.99, 0.95, 0.8, 0.5, 0.2, 0.05, 0.01])

    # relaxation boundary conditions
    if len(list(phi.shape)) == 2: #Marina: phi.ndim == 2
        for i in range(0,nr):
            phi[i,:] = phi1*rel[i] + phi[i,:]*(1 - rel[i])
            phi[n-1-i,:] = phi2*rel[i] + phi[n-1-i,:]*(1 - rel[i])
    else:
        for i in range(0,nr):
            phi[i] = phi1*rel[i] + phi[i]*(1 - rel[i])
            phi[n-1-i] = phi2*rel[i] + phi[n-1-i]*(1 - rel[i])

    return phi

# END OF BOUNDARY.PY
