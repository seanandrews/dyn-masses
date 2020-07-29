import os
import numpy as np
execfile('/home/sandrews/mypy/keplerian_mask/keplerian_mask.py')


# simulation
fname = 'simp3_std_highv_medr_noiseless'

# channelization
chanstart, chanwidth, nchan = '-3.5km/s', '0.2km/s', 35

# window / rebin / shift the MS to these velocities?
do_tavg = False
do_rebin = True
do_Hann = False
do_shift = False

# mask parameters
mstar, inc, PA, zr, rmax = 0.7, 40., 310., 0.23, 1.35

# imaging parameters
rms = '6.6mJy'
restfreq = 230.538e9

#########

# - fixed imaging parameters
cleanscales = [0, 5, 10, 15]
extens = ['.image', '.mask', '.model', '.pb', '.psf', '.residual', '.sumwt']

# import the FITS into MS format
os.system('rm -rf sim_MS/'+fname+'.ms')
importuvfits(fitsfile='sim_uvfits/'+fname+'.uvfits', vis='sim_MS/'+fname+'.ms')


# convolve with Hanning kernel?
if do_Hann:
    # do the convolution
    fname_new = fname+'.hann'
    os.system('rm -rf sim_MS/'+fname_new+'.ms')
    hanningsmooth(vis='sim_MS/'+fname+'.ms', 
                  outputvis='sim_MS/'+fname_new+'.ms', datacolumn='data')
    
    # export the convolved file
    fname = fname_new
    exportuvfits(vis='sim_MS/'+fname+'.ms',
                 fitsfile='sim_uvfits/'+fname+'.uvfits', datacolumn='data',
                 overwrite=True)

# time average?
if do_tavg:
    # do the time-averaging
    fname_new = fname+'.tavg'
    os.system('rm -rf sim_MS/'+fname_new+'.ms')
    split(vis='sim_MS/'+fname+'.ms', outputvis='sim_MS/'+fname_new+'.ms',
          datacolumn='data', timebin='30s')

    # export the time-averaged file
    fname = fname_new
    exportuvfits(vis='sim_MS/'+fname+'.ms',
                 fitsfile='sim_uvfits/'+fname+'.uvfits', datacolumn='data',
                 overwrite=True)
    

# rebin velocity grid?
if do_rebin:
    # do the regridding
    fname_new = fname+'.mstv2'
    os.system('rm -rf sim_MS/'+fname_new+'.ms')
    mstransform(vis='sim_MS/'+fname+'.ms', outputvis='sim_MS/'+fname_new+'.ms',
                datacolumn='data', regridms=True, mode='velocity',
                start=chanstart, width=chanwidth, nchan=nchan, outframe='LSRK',
                veltype='radio', restfreq=str(restfreq / 1e9)+'GHz')

    # export the regridded file
    #fname = fname_new
    exportuvfits(vis='sim_MS/'+fname_new+'.ms', 
                 fitsfile='sim_uvfits/'+fname_new+'.uvfits', datacolumn='data',
                 overwrite=True)

    fname_new = fname+'.cvel2'
    os.system('rm -rf sim_MS/'+fname_new+'.ms')
    cvel(vis='sim_MS/'+fname+'.ms', outputvis='sim_MS/'+fname_new+'.ms',
         mode='velocity', start=chanstart, width=chanwidth,
         nchan=nchan, outframe='LSRK', veltype='radio', 
         restfreq=str(restfreq / 1e9)+'GHz')

    # export the regridded file
    exportuvfits(vis='sim_MS/'+fname_new+'.ms',
                 fitsfile='sim_uvfits/'+fname_new+'.uvfits', datacolumn='data',
                 overwrite=True)


if do_shift:
    # do the interpolation
    fname_new = fname+'.shift'
    os.system('rm -rf sim_MS/'+fname_new+'.ms')
    # with mstransform
    mstransform(vis='sim_MS/'+fname+'.ms', outputvis='sim_MS/'+fname_new+'.ms',
                datacolumn='data', regridms=True, mode='velocity',
                start=chanstart, width=chanwidth, nchan=nchan, outframe='LSRK',
                veltype='radio', restfreq=str(restfreq / 1e9)+'GHz')

    # export the regridded file
    fname = fname_new
    exportuvfits(vis='sim_MS/'+fname+'.ms',
                 fitsfile='sim_uvfits/'+fname+'.uvfits', datacolumn='data',
                 overwrite=True)


sys.exit()

    

# - Make a dirty image to guide a clean mask
for ext in extens:
    os.system('rm -rf sim_images/im_'+fname+'_dirty'+ext)
tclean(vis='sim_MS/'+fname+'.ms', 
       imagename='sim_images/im_'+fname+'_dirty', specmode='cube', 
       start=chanstart, width=chanwidth, nchan=nchan, outframe='LSRK', 
       restfreq='230.538GHz', imsize=512, cell='0.02arcsec', 
       deconvolver='multiscale', scales=cleanscales, niter=0, 
       weighting='natural', interactive=False, nterms=1, 
       restoringbeam='common')

# - Make a keplerian mask from the (noise-free) dirty image
os.system('rm -rf sim_images/im_'+fname+'_dirty.mask.image')
make_mask('sim_images/im_'+fname+'_dirty.image', inc=inc, PA=PA, mstar=mstar, 
          dist=150., vlsr=0.0, r_max=rmax, nbeams=1.3)

# - Make a CLEAN image 
for ext in extens:
    os.system('rm -rf sim_images/im_'+fname+ext)
tclean(vis='sim_MS/'+fname+'.ms', 
       imagename='sim_images/im_'+fname, specmode='cube', 
       start=chanstart, width=chanwidth, nchan=nchan, outframe='LSRK', 
       restfreq='230.538GHz', imsize=512, cell='0.02arcsec', 
       deconvolver='multiscale', scales=cleanscales, niter=10000000, 
       threshold=rms, weighting='natural', 
       mask='sim_images/im_'+fname+'_dirty.mask.image', interactive=False, 
       nterms=1, restoringbeam='common')
