import numpy as np

from namelist import idbg,nb,nx,nx1,imoist_diff,imoist,imicrophys,irelax    # import global variables
from boundary import periodic


def horizontal_diffusion(tau,unew,snew,
    qvnew=None,qcnew=None,qrnew=None,ncnew=None,nrnew=None):
    """
    Horizontal diffusion for dry model.

    Input:  horizontal_diffusion(tau,unew,unew,snew,snew)
    Output: unew,snew
    """
    ind = tau>0

    if idbg == 1 and np.size(ind) > 0:
        print('Apply diffusion and gravity wave absorber ...\n')

    taumat = np.ones((unew.shape))*tau

    if np.all(tau <= 0):
        return
    else:
        sel = taumat > 0

        i = np.arange(nb,nx1+nb)
        unew[i,:] = (unew[i,:] + taumat[i,:]*(
            unew[i-1,:] - 2.*unew[i,:] +
            unew[i+1,:])/4.)*sel[i,:] + \
            unew[i,:]*~sel[i,:]

        i = np.arange(nb,nx+nb)
        snew[i,:] = (snew[i,:] + taumat[i,:]*(
            snew[i-1,:] - 2.*snew[i,:] +
            snew[i+1,:])/4.)*sel[i,:] + \
            snew[i,:]*~sel[i,:]
 
        if imoist==1 and imoist_diff ==1:
            qvnew[i,:] = (qvnew[i,:] + taumat[i,:]*(
                qvnew[i-1,:] - 2.*qvnew[i,:] +
                qvnew[i+1,:])/4.)*sel[i,:] + \
                qvnew[i,:]*~sel[i,:]

            qcnew[i,:] = (qcnew[i,:] + taumat[i,:]*(
                qcnew[i-1,:] - 2.*qcnew[i,:] +
                qcnew[i+1,:])/4.)*sel[i,:] + \
                qcnew[i,:]*~sel[i,:]

            qrnew[i,:] = (qrnew[i,:] + taumat[i,:]*(
                qrnew[i-1,:] - 2.*qrnew[i,:] +
                qrnew[i+1,:])/4.)*sel[i,:] + \
                qrnew[i,:]*~sel[i,:]

            if imicrophys==2:
                nrnew[i,:] = (nrnew[i,:] + taumat[i,:]*(
                    nrnew[i-1,:] - 2.*nrnew[i,:] +
                    nrnew[i+1,:])/4.)*sel[i,:] + \
                    nrnew[i,:]*~sel[i,:]

                ncnew[i,:] = (ncnew[i,:] + taumat[i,:]*(
                    ncnew[i-1,:] - 2.*ncnew[i,:] +
                    ncnew[i+1,:])/4.)*sel[i,:] + \
                    ncnew[i,:]*~sel[i,:]

    # exchange periodic boundaries
    if irelax == 0:
        unew = periodic(unew,nx1,nb)
        snew = periodic(snew,nx,nb)

        if imoist==1 and imoist_diff ==1:
            qvnew = periodic(qvnew,nx,nb)
            qcnew = periodic(qcnew,nx,nb)
            qrnew = periodic(qrnew,nx,nb)

            if imicrophys==2:
                ncnew = periodic(ncnew,nx,nb)
                nrnew = periodic(nrnew,nx,nb)

    if imoist==0:
        return unew, snew
    elif imicrophys==0 or imicrophys==1:
        return unew, snew, qvnew, qcnew, qrnew
    elif imicrophys==2:
        return unew, snew, qvnew, qcnew, qrnew, ncnew, nrnew

# END OF DIFFUSION.PY
