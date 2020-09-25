import os, sys
import numpy as np
sys.path.append('../')
from cube_parser import cube_parser

<<<<<<< HEAD
# Specify template parameters
startv = -9.699         # starting velocity (LSRK), wrt Vsys, in [km/s]
ch_spacing = 0.0159     # velocity channel spacing in [km/s]
nchan = None            # (optional): number of channels; if None, will make 
                        # the cube symmetric around Vsys
restfreq = 230.538e9    # rest frequency of line in [GHz]
Vsys = 0.               # systemic velocity in [km/s]
RA = 240.               # phase center RA in [degrees]
DEC = -40.              # phase center DEC in [degrees]
config = '5'            # ALMA configuration
total_time = '15min'    # total on-source time for simulation
integ = '30s'           # integration time interval for simulation



# Set the channel velocities in [m/s]
if nchan is None:
    nchan = np.int(2 * np.abs(startv) / ch_spacing + 1)
vel = (startv + ch_spacing * np.arange(nchan)) * 1000


# Set a filename that describes the template
outfile = 'template_ch' + str(np.int(1000*ch_spacing)) + \
          'ms_V0' + str(int(1000*startv)) + \
          'ms_nchan' + str(nchan) + '_config' + config + '_dt' + \
          total_time + '_tinteg' + integ

# compute a dummy model cube in a FITS file
os.system('rm -rf template_cubes/'+outfile+'.fits')
foo = cube_parser(dist=150., r_max=300., r_l = 300., FOV=8.0, Npix=256, 
                  restfreq=restfreq, RA=RA, DEC=DEC, Vsys=Vsys,
                  vel=vel, outfile='template_cubes/'+outfile+'.fits')

# notification
print('Wrote a cube: \n    template_cubes/'+outfile+'.fits')



# Mock observations

# output a parameter file (ASCII) for use inside CASA
f = open('template.pars.txt', 'w')
f.write(outfile+'\n')
f.write(config+'\n')
f.write(total_time+'\n')
f.write(integ+'\n')
f.close()

# run simulation script in CASA
=======
"""
This code creates a template cube and its corresponding FT (in both MS and 
UVFITS formats) for a generic model, using the CASA/simobserve capabilities.  
It is meant to be used as a blank slate for imposing different models upon at 
a later step (using 'mk_data.py').

The code imposes the observational parameters of the model, with the exception 
of noise (which can/should be dealt with in 'mk_data.py' and its dependent 
codebase.  This includes a set of *LSRK* channels, the target coordinates, the 
observing configuration, date and hour angle range, integration time, and the 
duration of the execution block (EB).  These parameters are saved in an ASCII
file for future reference (and subsequent use).

"""

### Specify template parameters

# bookkeeping
tname = 'std_medr_highv'    # base filename: append str(spec_oversample)+'x' if
			    # spec_oversample > 1

# spectral settings
ch_spacing = 61.           # frequency channel spacing in [kHz]
restfreq = 230.538e9       # rest frequency of line in [Hz]
Vsys = 0.0                 # systemic velocity (LSRK) in [km/s]
Vspan = 15.                # velocity half-span (Vsys +/- ~Vspan) in [km/s]
spec_oversample = 10       # how many channels per ch_spacing desired?

# target coordinates
RA = '16:00:00.00'         # phase center RA 
DEC = '-40:00:00.00'       # phase center DEC
HA = '0.0h'                # hour angle at start of observations
date = '2021/05/21'        # date string

# observing parameters
config = '5'               # ALMA configuration
total_time = '15min'       # total on-source time for simulation
integ = '30s'              # integration time interval for simulation


#=============================================================================#

### Constants
c_ = 2.9979e5              # speed of light in [km/s]


### Set the channels in frequency and LSRK velocity domains
nu_span = restfreq * (Vsys - Vspan) / c_
nu_sys = restfreq * (1. - Vsys / c_)
nchan = np.int(2 * np.abs(nu_span) / (ch_spacing*1e3 / spec_oversample) + 1)
freq = nu_sys - nu_span - (ch_spacing*1e3 / spec_oversample) * np.arange(nchan)
vel = c_ * (1. - freq / restfreq)


### Parse the target coordinates into degrees
RA_pieces = [np.float(RA.split(':')[i]) for i in np.arange(3)]
RAdeg = 15 * np.sum(np.array(RA_pieces) / [1., 60., 3600.])
DEC_pieces = [np.float(DEC.split(':')[i]) for i in np.arange(3)]
DECdeg = np.sum(np.array(DEC_pieces) / [1., 60., 3600.])


### Make a dummy template FITS cube for CASA simulations
if spec_oversample > 1:
    outfile = tname+str(spec_oversample)+'x'
else: outfile = tname
cube_parser(dist=150., r_max=300., r_l = 300., FOV=8.0, Npix=256, 
            restfreq=restfreq, RA=RAdeg, DEC=DECdeg, Vsys=Vsys, vel=vel*1e3, 
            outfile='template_cubes/'+outfile+'.fits')


### Record-keeping of template simulation parameters
f = open('template_params/'+outfile+'.params.txt', 'w')
f.write(outfile+'\n' + str(ch_spacing)+'\n' + str(restfreq)+'\n')
f.write(str(Vsys)+'\n' + str(Vspan)+'\n' + str(spec_oversample)+'\n')
f.write(RA+'\n' + DEC+'\n' + date+'\n' + HA+'\n')
f.write(config+'\n' + total_time+'\n' + integ)
f.close()


### Run simulation script in CASA
f = open('run_template.txt', 'w')
f.write(outfile)
f.close()
>>>>>>> d4a9c5ff4c73602ea621599da853d2b0f119f646
os.system('casa --nologger --nologfile -c mock_obs_alma.py')
