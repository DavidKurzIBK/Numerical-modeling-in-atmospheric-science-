# -*- coding: utf-8 -*-
import numpy as np
# from netCDF4 import Dataset #python-netcdf4 is not supportet by some
# linux distors
from namelist import idbg, nb, nx, nz, dt, dx, out_fname,    \
    u00, bv00, th00, pref, xl, thl, dth,  \
    topomx, topowd, imoist, imicrophys,\
    idthdt


def makeoutput(unow, snow, zht, its_out, its, Z, U, S, T, qvnow=None,
               qcnow=None, qrnow=None, tot_prec=None, prec=None,
               nrnow=None, ncnow=None, QV=None, QC=None, QR=None,
               TOT_PREC=None, PREC=None, NR=None, NC=None,
               dthetadt=None, DTHETADT=None):
    """
    Write current fields into output fields. Boundaries not written to disk.

    Input:  makeoutput(Unow, unow, snow, zht, its_out, its, Z, U, S, T, UU)
    Output: its_out, Z, U, S, T
    """

    if idbg == 1:
        print('Prepare output...\n')

    i = nb + np.arange(nx)
    k = np.arange(nz)
    its_out = its_out + 1

    # Horizontal destagger
    uout = 0.5 * (unow[i, :] + unow[i + 1, :])
    # Vertical destagger
    if idthdt == 1:
        dthetadtout = 0.5 * (dthetadt[:, k] + dthetadt[:, k + 1])

    print('Writing output...\n')

    T[its_out] = its * dt
    Z[its_out, :, :] = np.swapaxes(zht[i, :], 0, 1)
    U[its_out, :, :] = np.swapaxes(uout, 0, 1)
    S[its_out, :, :] = np.swapaxes(snow[i, :], 0, 1)

    #if imoist == 1 and imicrophys != 0:
    if imoist == 1:
        QV[its_out, :, :] = np.swapaxes(qvnow[i, :], 0, 1)
        QC[its_out, :, :] = np.swapaxes(qcnow[i, :], 0, 1)
        QR[its_out, :, :] = np.swapaxes(qrnow[i, :], 0, 1)
        if imicrophys != 0:
            TOT_PREC[its_out, :] = tot_prec[i]
            PREC[its_out, :] = prec[i]
            if idthdt == 1:
                DTHETADT[its_out, :, :] = np.swapaxes(dthetadtout[i, :], 0, 1)
        if imicrophys == 2:
            NR[its_out, :, :] = np.swapaxes(nrnow[i, :], 0, 1)
            NC[its_out, :, :] = np.swapaxes(ncnow[i, :], 0, 1)

    if imoist == 0:
        return its_out, Z, U, S, T
    elif imoist == 1:
        if imicrophys == 0 or imicrophys == 1:
            if idthdt == 1:
                return its_out, Z, U, S, T, QC, QV, QR, TOT_PREC, PREC, DTHETADT
            else:
                return its_out, Z, U, S, T, QC, QV, QR, TOT_PREC, PREC

        if imicrophys == 2:
            if idthdt == 1:
                return its_out, Z, U, S, T, QC, QV, QR, TOT_PREC, PREC, NR, NC, DTHETADT
            else:
                return its_out, Z, U, S, T, QC, QV, QR, TOT_PREC, PREC, NR, NC

# -----------------------------------------------------------------------------


def write_output(nout, Z, U, S, T, QV=None, QC=None, QR=None, PREC=None,
                 TOT_PREC=None, NR=None, NC=None, DTHETADT=None):
    """
    Record output in .npz file.

    Input:  write_output(nout)
    Output: none
    """
    if idbg == 1 or idbg == 0:
        print('Writing to file %s.npz \n' % out_fname)
        print('Output contains %u output steps\n' % nout)

    # destagger height
    Z = 0.5 * (Z[:, 0:nz, :] + Z[:, 1:nz + 1, :])
    x_out = dx * np.arange(0, nx) / 1000.
    z_out = Z[0, :, 0] / 1000.

    u_out = u00
    bv_out = bv00
    th_out = th00
    p_ref = pref

    np.savez(out_fname, u00=u00, thl=thl, th00=th00, topomx=topomx,
             topowd=topowd, nx=nx, nz=nz, dx=dx, time=T, x=x_out, z=z_out)

    # write data to binary
    if imoist == 0:
        np.savez(out_fname, u00=u00, thl=thl, th00=th00, topomx=topomx,
                 topowd=topowd, nx=nx, nz=nz, dx=dx, time=T, x=x_out,
                 z=z_out, height=Z, horizontal_velocity=U,
                 isentropic_density=S)
    else:
        if imoist == 1:
            np.savez(out_fname, u00=u00, thl=thl, th00=th00,
                     topomx=topomx, topowd=topowd, nx=nx, nz=nz, dx=dx,
                     time=T, x=x_out, z=z_out, height=Z, horizontal_velocity=U,
                     isentropic_density=S, specific_humidity=QV,
                     specific_cloud_liquid_water_content=QC,
                     specific_rain_water_content=QR,
                     accumulated_precipitation=TOT_PREC,
                     precipitation_rate=PREC)
            if idthdt == 1:
                np.savez(out_fname, u00=u00, thl=thl, th00=th00, topomx=topomx,
                         topowd=topowd, nx=nx, nz=nz, dx=dx, time=T, x=x_out,
                         z=z_out, height=Z, horizontal_velocity=U,
                         isentropic_density=S, specific_humidity=QV,
                         specific_cloud_liquid_water_content=QC,
                         specific_rain_water_content=QR,
                         accumulated_precipitation=TOT_PREC,
                         precipitation_rate=PREC,
                         latent_heat_tendency=DTHETADT)
            if imicrophys == 2:
                np.savez(out_fname, u00=u00, thl=thl, th00=th00, topomx=topomx,
                         topowd=topowd, nx=nx, nz=nz, dx=dx, time=T, x=x_out,
                         z=z_out, height=Z, horizontal_velocity=U,
                         isentropic_density=S, specific_humidity=QV,
                         specific_cloud_liquid_water_content=QC,
                         specific_rain_water_content=QR,
                         accumulated_precipitation=TOT_PREC,
                         precipitation_rate=PREC,
                         rain_number_density=NR,
                         cloud_number_density=NC)
                if idthdt == 1:
                    np.savez(out_fname, u00=u00, thl=thl, th00=th00,
                             topomx=topomx, topowd=topowd, nx=nx, nz=nz, dx=dx,
                             time=T, x=x_out, z=z_out, height=Z,
                             horizontal_velocity=U, isentropic_density=S,
                             specific_humidity=QV,
                             specific_cloud_liquid_water_content=QC,
                             specific_rain_water_content=QR,
                             accumulated_precipitation=TOT_PREC,
                             precipitation_rate=PREC,
                             rain_number_density=NR, cloud_number_density=NC,
                             latent_heat_tendency=DTHETADT)

# -----------------------------------------------------------------------------


    # Write netCDF file
    #filename = out_fname + '.nc'
    #ncfile = Dataset(filename,mode='w')

    # create the dimensions: nx,nz,iters
    # ncfile.createDimension('x',nx)
    # ncfile.createDimension('z',nz)
    # ncfile.createDimension('time',nout)

    # define coordinate variables
    #xx      = ncfile.createVariable('x',np.dtype('float32').char,('x',))
    #zz      = ncfile.createVariable('z',np.dtype('float32').char,('z',))
    #t       = ncfile.createVariable('time',np.dtype('float32').char,('time',))

    # assign units attributes to coordinate variables data.
    #xx.units    = 'km'
    #zz.units    = 'km'
    #t.units     = 's'

    # write data to coordinate variables
    #xx[:]   = x_out
    #zz[:]   = z_out
    #t[:]    = T[:]

    # define parameter variables
    #ui      = ncfile.createVariable('u00',np.dtype('float32').char)
    #bvi     = ncfile.createVariable('bv00',np.dtype('float32').char)
    #thi     = ncfile.createVariable('th00',np.dtype('float32').char)
    #pr      = ncfile.createVariable('pref',np.dtype('float32').char)
    #domx    = ncfile.createVariable('xl',np.dtype('float32').char)
    #domz    = ncfile.createVariable('thl',np.dtype('float32').char)
    #resx    = ncfile.createVariable('nx',np.dtype('float32').char)
    #resz    = ncfile.createVariable('nz',np.dtype('float32').char)
    #ntout   = ncfile.createVariable('nout',np.dtype('float32').char)
    #dxx     = ncfile.createVariable('dx',np.dtype('float32').char)
    #dthh    = ncfile.createVariable('dth',np.dtype('float32').char)
    #dtt     = ncfile.createVariable('dt',np.dtype('float32').char)
    #mth     = ncfile.createVariable('topomx',np.dtype('float32').char)
    #mtwd    = ncfile.createVariable('topowd',np.dtype('float32').char)

    # write data to parameter variables
    #ui[0]      = u_out
    #bvi[0]     = bv_out
    #thi[0]     = th_out
    #pr[0]      = p_ref
    #domx[0]    = xl
    #domz[0]    = thl
    #resx[0]    = nx
    #resz[0]    = nz
    #ntout[0]   = nout
    #dxx[0]     = dx
    #dthh[0]    = dth
    #dtt[0]     = dt
    #mth[0]     = topomx
    #mtwd[0]    = topowd

    # create Z, U, S variables
    # heights     = ncfile.createVariable('height',np.dtype('float32').char,('time',
    #'z','x'))
    # velocity    = ncfile.createVariable('horizontal_velocity',
    # np.dtype('float32').char,('time','z',
    #'x'))
    # isendens    = ncfile.createVariable('isentropic_density',
    # np.dtype('float32').char,('time','z',
    #'x'))
    # if imoist == 1:
    # hum          = ncfile.createVariable('specific_humidity',np.dtype('float32').char,('time',
    #'z','x'))
    # cloud_water  = ncfile.createVariable('specific_cloud_liquid_water_content',np.dtype('float32').char,('time',
    #'z','x'))
    # rain_water  = ncfile.createVariable('specific_rain_water_content',np.dtype('float32').char,('time',
    #'z','x'))
    #tot_precip   = ncfile.createVariable('accumulated_precipitation',np.dtype('float32').char,('time','x'))
    #precip       = ncfile.createVariable('precipitation_rate',np.dtype('float32').char,('time','x'))
    # if imicrophys == 2:
    # num_rain          = ncfile.createVariable('rain_number_density',np.dtype('float32').char,('time',
    #'z','x'))
    # num_cloud  = ncfile.createVariable('cloud_number_density',np.dtype('float32').char,('time',
    #'z','x'))
    # if idthdt == 1:
    # dtheta  = ncfile.createVariable('latent_heat_tendency',np.dtype('float32').char,('time',
    #'z','x'))
    # set units attribute
    #heights.units   = 'm'
    #velocity.units  = 'm/s'
    #isendens.units  = ''

    # if imoist == 1:
    #hum.units         = 'kg/kg'
    #cloud_water.units = 'kg/kg'
    #rain_water.units  = 'kg/kg'
    #tot_precip.units  = 'mm'
    #precip.units      = 'mm h'
    # if imicrophys == 2:
    #num_rain.units = '1/kg'
    #num_cloud.units= '1/kg'
    # if idthdt == 1:
    #dtheta.units = 'K/h'

    # write data to variables
    #heights[:,:,:]  = Z[:,:,:]
    #velocity[:,:,:] = U[:,:,:]
    #isendens[:,:,:] = S[:,:,:]
    # if imoist == 1:
    #hum[:,:,:]         = QV[:,:,:]
    #cloud_water[:,:,:] = QC[:,:,:]
    #rain_water[:,:,:]  = QR[:,:,:]
    #tot_precip[:,:]    = TOT_PREC[:,:]
    #precip[:,:]        = PREC[:,:]
    # if imicrophys == 2:
    #num_rain[:,:,:]  = NR[:,:,:]
    #num_cloud[:,:,:] = NC[:,:,:]
    # if idthdt == 1:
    # dtheta[:,:,:]=DTHETADT[:,:,:]
    # ncfile.close()

# END OF OUTPUT.PY
