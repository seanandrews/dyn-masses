import os, sys
import numpy as np
sys.path.append('../')
from cube_parser import cube_parser

# Specify template parameters
startv = -10.           # starting velocity (LSRK), wrt Vsys, in [km/s]
ch_spacing = 0.1        # velocity channel spacing in [km/s]
nchan = None            # (optional): number of channels; if None, will make 
                        # the cube symmetric around Vsys
restfreq = 230.538e9    # rest frequency of line in [GHz]
Vsys = 0.               # systemic velocity in [km/s]
RA = 240.               # phase center RA in [degrees]
DEC = -40.              # phase center DEC in [degrees]
config = '5'            # ALMA configuration
total_time = '15min'    # total on-source time for simulation
integ = '6s'            # integration time interval for simulation



# Set the channel velocities in [m/s]
if nchan is None:
    nchan = np.int(2 * np.abs(startv) / ch_spacing + 1)
vel = (startv + ch_spacing * np.arange(nchan)) * 1000

# Set a filename that describes the template
outfile = 'template_ch' + str(np.int(1000*ch_spacing)) + \
          'ms_V0' + str(int(startv)) + \
          'kms_nchan' + str(nchan) + '_config' + config + '_dt' + \
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
os.system('casa --nologger --nologfile -c mock_obs_alma.py')
