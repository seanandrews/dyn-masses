import os, sys
import numpy as np
sys.path.append('../')
from cube_parser import cube_parser

def mk_template_cube(vel_span, ch_width, restfreq=230.538e9, vsys=0.0,
                     RA=240., DEC=-40.):

    # set velocities based on inputs
    vel = vsys - vel_span + ch_width * np.arange(2 * vel_span / ch_width + 1)
    vel *= 1000

    # compute a dummy model cube in a FITS file
    outfile = 'template_cube_vs' + str(vel_span) + '_ch' + str(ch_width)
    os.system('rm -rf template_cubes/'+outfile+'.fits')
    foo = cube_parser(inc=30., PA=40., mstar=1.0, FOV=8.0, dist=150., Npix=256, 
                      Tb0=80, Tbq=-0.5, r_max=300, vsys=0.0, Tbmax=500, vel=vel,
                      restfreq=restfreq, RA=RA, DEC=DEC,
                      outfile='template_cubes/'+outfile+'.fits')

    # notification
    print('Wrote a template cube to template_cubes/'+outfile+'.fits')

    return 0
