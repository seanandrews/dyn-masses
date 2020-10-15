import os, sys
import numpy as np
sys.path.append('../')
from cube_parser import cube_parser

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
Vsys = 4.0                 # systemic velocity (LSRK) in [km/s]
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
c_ = 2.99792e5              # speed of light in [km/s]


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
os.system('casa --nologger --nologfile -c CASA_scripts/mock_obs_alma.py')
