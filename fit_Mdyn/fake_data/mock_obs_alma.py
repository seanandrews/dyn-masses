import os
import numpy as np

# parse inputs
io = np.loadtxt('template.pars.txt', dtype=str)
cube, conf_str, dt_str, int_str = io[0], io[1], io[2], io[3]

# configuration file
cfg_dir = '/pool/asha0/casa-pipeline-release-5.6.1-8.el6/data/alma/simmos/'
cfg_str = cfg_dir + 'alma.cycle7.' + conf_str + '.cfg'

# generate (u,v) tracks
os.chdir('template_sims/')
default('simobserve')
simobserve(project=cube+'.sim', skymodel='../template_cubes/'+cube+'.fits', 
           antennalist=cfg_str, totaltime=dt_str, integration=int_str, 
           thermalnoise='', refdate='2021/05/01', mapsize='10arcsec')

# make a template UVFITS file
infile = cube+'.sim/'+cube+'.sim.alma.cycle7.'+conf_str+'.ms'
exportuvfits(vis=infile, fitsfile='../template_uvfits/'+cube+'.uvfits',
             datacolumn='data', overwrite=True)
os.chdir('../')

# notification
print('Wrote a UVFITS file to \n    template_uvfits/'+cube+'.uvfits')
