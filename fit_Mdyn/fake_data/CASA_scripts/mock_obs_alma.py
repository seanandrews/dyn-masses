import os
import numpy as np

# read in which template to run
which_temp = str(np.loadtxt('run_template.txt', dtype=str))

# parse inputs
io = np.loadtxt('template_params/'+which_temp+'.params.txt', dtype=str)
cube, ra_str, dec_str, date_str, ha_str = io[0], io[6], io[7], io[8], io[9]
conf_str, dt_str, int_str = io[10], io[11], io[12]

# configuration file
cfg_dir = '/pool/asha0/casa-pipeline-release-5.6.1-8.el6/data/alma/simmos/'
cfg_str = cfg_dir + 'alma.cycle7.' + conf_str + '.cfg'

# generate (u,v) tracks
os.chdir('template_sims/')
default('simobserve')
simobserve(project=cube+'.sim', skymodel='../template_cubes/'+cube+'.fits', 
           antennalist=cfg_str, totaltime=dt_str, integration=int_str, 
           thermalnoise='', refdate=date_str, hourangle=ha_str, 
           mapsize='10arcsec')

# make a template UVFITS file
infile = cube+'.sim/'+cube+'.sim.alma.cycle7.'+conf_str+'.ms'
exportuvfits(vis=infile, fitsfile='../template_uvfits/'+cube+'.uvfits',
             datacolumn='data', overwrite=True)
os.chdir('../')

os.system('rm -rf run_template.txt')
