import os, sys
import numpy as np
execfile('/home/sandrews/mypy/keplerian_mask/keplerian_mask.py')

# simulation
fname = 'simp3_std_medr_medv_shortint_noiseless.tavg'

# frequencies
chanstart, chanwidth, nchan = '-5.6km/s', '0.16km/s', 121

# imaging
robust = 2.0
thresh = '7.4mJy'
restfreq = 230.538e9
imsize = 512
cell = '0.02arcsec'

# mask parameters
inc, PA, mstar, dist, zr, Vsys = 40., 310., 0.7, 150., 0.23, 4.
rmax, nbeams = 1.35, 1.3

#########


# - fixed imaging parameters
cleanscales = [0, 5, 10, 15]
extens = ['.image', '.mask', '.model', '.pb', '.psf', '.residual', '.sumwt']

# import the FITS into MS format
os.system('rm -rf sim_MS/'+fname+'.ms')
importuvfits(fitsfile='sim_uvfits/'+fname+'.uvfits', vis='sim_MS/'+fname+'.ms')

# make a dirty image to guide a Keplerian mask
for ext in extens:
    os.system('rm -rf sim_images/im_'+fname+'_dirty'+ext)
tclean(vis='sim_MS/'+fname+'.ms', 
       imagename='sim_images/im_'+fname+'_dirty', specmode='cube', 
       start=chanstart, width=chanwidth, nchan=nchan, outframe='LSRK', 
       restfreq=str(restfreq / 1e9)+'GHz', imsize=imsize, cell=cell, 
       deconvolver='multiscale', scales=cleanscales, niter=0, 
       weighting='briggs', robust=robust, interactive=False, nterms=1, 
       restoringbeam='common')

# - Make a keplerian mask from the (noise-free) dirty image
os.system('rm -rf sim_images/im_'+fname+'_dirty.mask.image')
make_mask('sim_images/im_'+fname+'_dirty.image', inc=inc, PA=PA, zr=zr,
          mstar=mstar, dist=dist, vlsr=Vsys, r_max=rmax, nbeams=nbeams)

# - Make a CLEAN image 
for ext in extens:
    os.system('rm -rf sim_images/im_'+fname+ext)
tclean(vis='sim_MS/'+fname+'.ms', 
       imagename='sim_images/im_'+fname, specmode='cube', 
       start=chanstart, width=chanwidth, nchan=nchan, outframe='LSRK', 
       restfreq=str(restfreq / 1e9)+'GHz', imsize=imsize, cell=cell, 
       deconvolver='multiscale', scales=cleanscales, niter=10000000, 
       threshold=thresh, weighting='briggs', robust=robust, 
       mask='sim_images/im_'+fname+'_dirty.mask.image', interactive=False, 
       nterms=1, restoringbeam='common')

# export to a FITS cube
exportfits('sim_images/im_'+fname+'.image', 
           'sim_images/im_'+fname+'.image.fits', overwrite=True)

# cleanup
os.system('rm -rf *.last')
os.system('rm -rf *.log')
