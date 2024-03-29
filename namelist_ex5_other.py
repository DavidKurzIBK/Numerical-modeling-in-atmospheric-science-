# -*- coding: utf-8 -*-
"""
File defining the global variables used in the main program
and all subfunctions.
"""

# --------------------------------------------------------
# --------------------- USER NAMELIST --------------------
# --------------------------------------------------------

# Output control
#-------------------------------------------------
out_fname   = 'output'          # file name of output
iout        = 360               # write every iout-th time-step into the output file
iiniout     = 1                 # write initial field (0 = no, 1 = yes)

# Domain size
#-------------------------------------------------
xl      = 500000.               # domain size  [m]
nx      = 100                   # number of grid points in horizontal direction
dx      = xl/nx                 # horizontal resolution [m]
thl     = 60.                   # domain depth  [K]
nz      = 60                    # vertical resolution
dt      = 10                    # time step [s] 10
diff    = 0.08                  # (horizontal) diffusion coefficient
time    = 24*60*60              # integration time [s]

# Topography
#-------------------------------------------------
topomx  = 1500                   # mountain height [m] 500
topowd  = 50000                 # mountain half width [m] 50000
topotim = 1800                  # mountain growth time [s]

# Initial atmosphere
#-------------------------------------------------
u00     = 0.                    # initial velocity [m/s]
bv00    = 0.01                  # Brunt-Vaisalla frequency N [1/s]
th00    = 280.                  # potential temperature at surface --> 300

ishear  = 1                     # wind shear simulation (0 = no shear, 1 = shear)
k_shl   = 29                    # bottom level of wind shear layer (ishear = 1)
                                # bottom level of wind layer is 0 (index)
k_sht   = 36                    # top level of wind shear layer (ishear = 1)
                                # top level of wind layer is nz-1 (index)
u00_sh  = 10.                   # initial velocity below shear layer [m/s] (ishear = 1)
                                # u00 is speed above shear layer [m/s]   #orig 0.

# Boundaries
#-------------------------------------------------
nab     = 30                    # number of grid points in absorber
diffabs = 1.                    # maximum value of absorber
irelax  = 1                     # lateral boundaries (0 = periodic, 1 = relax)
nb      = 2                     # number of boundary points on each side

# Print options
#-------------------------------------------------
idbg    = 0                     # print debugging text (0 = not print, 1 = print)
iprtcfl = 1                     # print Courant number (0 = not print, 1 = print)
itime   = 1                     # print computation time (0 = not print, 1 = print)

# Physics: Moisture
#-------------------------------------------------
imoist          = 1             # include moisture (0 = dry, 1 = moist)
imoist_diff     = 0             # apply diffusion to qv, qc, qr (0 = off, 1 = on)
imicrophys      = 2             # include microphysics (0 = off, 1 = kessler, 2 = two moment)
idthdt          = 1             # couple physics to dynamics (0 = off, 1 = on)
iern            = 0             # evaporation of rain droplets (0 = off, 1 = on)
imoist_pert     = 0             # initial moisture perturbation

# Options for Kessler scheme
#-------------------------------------------------
vt_mult         = 2            # multiplication factor for termianl fall velocity
autoconv_th     = 0.0001        # critical cloud water mixing ratio for the onset
                                # of autoconversion [kg/kg]
autoconv_mult   = 2.            # multiplication factor for autoconversion
sediment_on     = 1             # switch to turn on / off sedimentation

#----------------------------------------------------------------------
#----------------------------------------------------------------------
#----------------------------------------------------------------------

# Physical constants
#--------------------------
g       = 9.81                  # gravity
cp      = 1004.                 # specific heat of air at constant pressure
r       = 287.                  # gas constant of air [J/kgK]
r_v     = 461.                  # gas constant of vapor [J/kgK]
rdcp    = r/cp                  # short cut for R/Cp
cpdr    = cp/r                  # short cut for Cp/R
pref    = 100*1000.             # reference pressure in SI units (Pa, not hPa!)
z00     = 0.                    # surface height
prs00   = pref                  # upstream surface pressure (= ref. pressure)
exn00   = cp*(prs00/pref)**rdcp #

# compute input parameters
#--------------------------
dth     = thl/nz                # spacing between vertical layers [K]
nts     = round(time/dt,0)      # number of iterations
nout    = int(nts/iout)         # number of output steps

nx1     = nx + 1                # number of staggered gridpoints in x
nz1     = nz + 1                # number of staggered gridpoints in z
nxb     = nx + 2*nb             # x range of unstaggered variable
nxb1    = nx1 + 2*nb            # x range of staggered variable

# END OF NAMELIST.PY
