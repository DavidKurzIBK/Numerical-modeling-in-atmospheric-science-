# -*- coding: utf-8 -*-
import numpy as np
from namelist import idbg, idthdt, nx, nxb, nb, nz, dth, dt # global variables

k = np.arange(1,nz-1)

def prog_isendens(sold,snow,unow,dtdx,dthetadt=None):
    """
    Prognostic step for isentropic mass density

    Input:      prog_isendens(sold,snow,unow,dtdx,dthetadt)
    Output:     snew
    """
    if idbg == 1:
        print('Prognostic step: Isentropic mass density ...\n')

    # Declare
    snew = np.zeros((nxb,nz))

    # *** Exercise 2.1/5.2 isentropic mass density ***
    # *** time step for isentropic mass density ***
    # *** edit here ***
    i = nb+np.arange(0,nx)
    snew[i,:] = sold[i,:] - dtdx/2 * ((unow[i+1,:]+unow[i+2,:]) * snow[i+1,:] - (unow[i-1,:]+unow[i,:]) * snow[i-1,:])
    
    if idthdt:
        ii,kk = np.ix_(i,k)
        snew[ii,kk] = snew[ii,kk]- dt/dth * (snow[ii,kk+1]-snow[ii,kk-1]) * (dthetadt[ii,kk]+dthetadt[ii,kk+1]) / 2
    # *** Exercise 2.1/5.2 isentropic mass density ***

    return snew

def prog_velocity(uold,unow,mtg,dtdx,dthetadt=None):
    """
    Prognostic step for momentum

    Input:      prog_velocity(uold,unow,mtg,dtdx,dthetadt)
    Output:     unew
    """
    if idbg == 1:
        print('Prognostic step: Velocity ...\n')

    # Declare
    unew = np.zeros((nx+1+2*nb,nz))
    
    # *** Exercise 2.1/5.2 velocity ***
    # *** time step for momentum ***
    # *** edit here ***
    i = nb+np.arange(0,nx+1)
    unew[i,:] = uold[i,:] - unow[i,:]* dtdx * (unow[i+1,:]-unow[i-1,:]) - 2*dtdx*(mtg[i,:]-mtg[i-1,:])
    
    if idthdt:
        ii,kk = np.ix_(i,k)
        unew[ii,kk] = unew[ii,kk]- dt/dth * (unow[ii,kk+1]-unow[ii,kk-1]) * (dthetadt[ii,kk+1]+dthetadt[ii-1,kk+1] + dthetadt[ii,kk]+dthetadt[ii-1,kk]) / 4 
    
    # *** Exercise 2.1/5.2 velocity ***
    return unew

def prog_moisture(unow,qvold,qcold,qrold,
                  qvnow,qcnow,qrnow,qvnew,qcnew,qrnew,dtdx,dthetadt=None):
    """
    Prognostic step for hydrometeors

    Input:      prog_moisture(unow,qvold,qcold,qrold, \
                              qvnow,qcnow,qrnow,qvnew,qcnew,qrnew,dtdx, \
                              dthetadt)
    Output:     qvnew,qcnew,qrnew
    """

    if idbg == 1:
        print('Prognostic step: Moisture scalars ...\n')

    # Declare
    qvnew = np.zeros((nx+2*nb,nz))
    qcnew = np.zeros((nx+2*nb,nz))
    qrnew = np.zeros((nx+2*nb,nz))

    # *** Exercise 4.1/5.2 moisture advection ***
    
    i = nb+np.arange(0,nx)
    
    # Advection
    qvnew[i,:] = qvold[i,:] - dtdx/2 * ((qvnow[i+1,:] - qvnow[i-1,:]) * (unow[i,:] + unow[i+1,:]))
    qcnew[i,:] = qcold[i,:] - dtdx/2 * ((qcnow[i+1,:] - qcnow[i-1,:]) * (unow[i,:] + unow[i+1,:]))
    qrnew[i,:] = qrold[i,:] - dtdx/2 * ((qrnow[i+1,:] - qrnow[i-1,:]) * (unow[i,:] + unow[i+1,:]))
    
    # Conservation form
    # qvnew[i,:] = qvold[i,:] - dtdx/2 * ((unow[i+1,:]+unow[i+2,:]) * qvnow[i+1,:] - (unow[i-1,:]+unow[i,:]) * qvnow[i-1,:])
    # qcnew[i,:] = qcold[i,:] - dtdx/2 * ((unow[i+1,:]+unow[i+2,:]) * qcnow[i+1,:] - (unow[i-1,:]+unow[i,:]) * qcnow[i-1,:])
    # qrnew[i,:] = qrold[i,:] - dtdx/2 * ((unow[i+1,:]+unow[i+2,:]) * qrnow[i+1,:] - (unow[i-1,:]+unow[i,:]) * qrnow[i-1,:])
    
    if idthdt:
        ii,kk = np.ix_(i,k)
        qvnew[ii,kk] = qvnew[ii,kk]- dt/dth * (qvnow[ii,kk+1]-qvnow[ii,kk-1]) * (dthetadt[ii,kk]+dthetadt[ii,kk+1]) / 2
        qcnew[ii,kk] = qcnew[ii,kk]- dt/dth * (qcnow[ii,kk+1]-qcnow[ii,kk-1]) * (dthetadt[ii,kk]+dthetadt[ii,kk+1]) / 2
        qrnew[ii,kk] = qrnew[ii,kk]- dt/dth * (qrnow[ii,kk+1]-qrnow[ii,kk-1]) * (dthetadt[ii,kk]+dthetadt[ii,kk+1]) / 2
    # *** Exercise 4.1/5.2  ***
    
    return qvnew,qcnew,qrnew


def prog_numdens(unow,ncold,nrold,ncnow,nrnow,ncnew,nrnew,dtdx,dthetadt=None):
    """
    Prognostic step for number densities

    Input:      prog_numdens(unow,ncold,nrold,ncnow,nrnow,ncnew,nrnew,
                             dthetadt=0)
    Output:     ncnew,nrnew
    """

    if idbg == 1:
       print('Prognostic step: Number densities ...')

    # Declare
    ncnew = np.zeros((nx+2*nb,nz))
    nrnew = np.zeros((nx+2*nb,nz))

    # *** Exercise 5.1/5.2 number densities ***
    i = nb+np.arange(0,nx)
    
    ncnew[i,:] = ncold[i,:] - dtdx/2 * ((ncnow[i+1,:] - ncnow[i-1,:]) * (unow[i,:] + unow[i+1,:]))
    nrnew[i,:] = nrold[i,:] - dtdx/2 * ((nrnow[i+1,:] - nrnow[i-1,:]) * (unow[i,:] + unow[i+1,:]))
    
    if idthdt:
        ii,kk = np.ix_(i,k)
        ncnew[ii,kk] = ncnew[ii,kk]- dt/dth * (ncnow[ii,kk+1]-ncnow[ii,kk-1]) * (dthetadt[ii,kk]+dthetadt[ii,kk+1]) / 2
        nrnew[ii,kk] = nrnew[ii,kk]- dt/dth * (nrnow[ii,kk+1]-nrnow[ii,kk-1]) * (dthetadt[ii,kk]+dthetadt[ii,kk+1]) / 2
        
    # *** Exercise 5.1/5.2  *

    return ncnew,nrnew
