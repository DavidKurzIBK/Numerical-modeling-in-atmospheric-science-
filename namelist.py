# -*- coding: utf-8 -*-
"""
File defining the global variables used in the main program
and all subfunctions.
"""

# --------------------------------------------------------
# --------------------- USER NAMELIST --------------------
# --------------------------------------------------------

#
# New VARS
#-------------------------------------------------
mtn_topo = 0 #  1: lee hill; 2: downstream hill
leeHill_rel = 5/8 # relative position of lee hill
w_ratio = 5/16 # width ratio of lee hill
h_ratio = 3/4 # height ratio of lee hill

surf_friction = 0 # part of ishear; linear surface wind profil

moist_setup = 3 # part of imoist, 1:lower moist layer,  2: include second moist layer, 3: low high RH layer
h_moistLayer_up = 12
h_moistLayer_low = 6

#-------------------------------------------------

# Output control
#-------------------------------------------------
out_fname   = 'outputs/output_ws'      # file name of output
# iout        = 360               # write every iout-th time-step into the output file
iiniout     = 1                 # write initial field (0 = no, 1 = yes)

# Domain size
#-------------------------------------------------
xl      = 150000.               # domain size  [m]
nx      = 150                   # number of grid points in horizontal direction
dx      = xl/nx                 # horizontal resolution [m]
thl     = 60.                   # domain depth  [K]
nz      = 60                    # vertical resolution
dt      = 2                     # time step [s]
iout    = 3600/dt               # write every iout-th time-step into the output file
diff    = 0.3                   # (horizontal) diffusion coefficient
time    = 12*60*60              # integration time [s]


# Topography
#-------------------------------------------------
topomx  = 1000                  # mountain height [m]
topowd  = 20000                 # mountain half width [m]
topotim = 7200                  # mountain growth time [s]

# Initial atmosphere
#-------------------------------------------------
u00     = 0.                   # initial velocity [m/s] (above shear layer if present)
bv00    = 0.01                  # Brunt-Vaisalla frequency [1/s]
th00    = 280.                  # potential temperature at surface

ishear  = 1                     # wind shear simulation (0 = no shear, 1 = shear)
k_shl   = 6                     # bottom level of wind shear layer (ishear = 1)
                                # bottom level of wind layer is 0 (index)
k_sht   = 12                     # top level of wind shear layer (ishear = 1)
                                # top level of wind layer is nz-1 (index)
u00_sh  = 10.                   # initial velocity below shear layer [m/s] (ishear = 1)
                                # u00 is speed above shear layer [m/s]   #orig 0.

# Boundaries
#-------------------------------------------------
nab     = 30                     # number of grid points in absorber
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
imoist          = 1            # include moisture (0 = dry, 1 = moist)
imoist_diff     = 1             # apply diffusion to qv, qc, qr (0 = off, 1 = on)
imicrophys      = 1             # include microphysics (0 = off, 1 = kessler, 2 = two moment)
idthdt          = 1             # couple physics to dynamics (0 = off, 1 = on)
iern            = 0             # evaporation of rain droplets (0 = off, 1 = on)

# Options for Kessler scheme
#-------------------------------------------------
vt_mult         = 1.            # multiplication factor for termianl fall velocity
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
