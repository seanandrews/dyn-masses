import os
import numpy as np
execfile('/home/sandrews/mypy/keplerian_mask/keplerian_mask.py')


# simulation
fname = 'simp3_std_highv_medr_noiseless'

# postprocessing steps and parameters
do_tavg = False
do_regrid = False
do_shift = False
do_image = False

# time-averaging
out_tint = '30s'

# regridding, shifting, imaging
chanstart, chanwidth, nchan = '-3.5km/s', '0.2km/s', 35

# imaging
robust = 2.0
thresh = '6.6mJy'
restfreq = 230.538e9
imsize = 512
cell = '0.02arcsec'

# mask parameters
inc, PA, mstar, dist, zr, vlsr = 40., 310., 0.7, 150., 0.23, 0.
rmax, nbeams = 1.35, 1.3


#########


# - fixed imaging parameters
cleanscales = [0, 5, 10, 15]
extens = ['.image', '.mask', '.model', '.pb', '.psf', '.residual', '.sumwt']

# import the FITS into MS format
os.system('rm -rf sim_MS/'+fname+'.ms')
importuvfits(fitsfile='sim_uvfits/'+fname+'.uvfits', vis='sim_MS/'+fname+'.ms')


# time average?
if do_tavg:
    # do the time-averaging
    fname_new = fname+'.tavg'
    os.system('rm -rf sim_MS/'+fname_new+'.ms')
    split(vis='sim_MS/'+fname+'.ms', outputvis='sim_MS/'+fname_new+'.ms',
          datacolumn='data', timebin=out_tint)

    # export the time-averaged file
    fname = fname_new
    exportuvfits(vis='sim_MS/'+fname+'.ms',
                 fitsfile='sim_uvfits/'+fname+'.uvfits', datacolumn='data',
                 overwrite=True)
    

# regrid channels?
if do_regrid:
    # do the regridding
    fname_new = fname+'.regrid'
    os.system('rm -rf sim_MS/'+fname_new+'.ms')
    mstransform(vis='sim_MS/'+fname+'.ms', outputvis='sim_MS/'+fname_new+'.ms',
                datacolumn='data', regridms=True, mode='velocity',
                start=chanstart, width=chanwidth, nchan=nchan, outframe='LSRK',
                veltype='radio', restfreq=str(restfreq / 1e9)+'GHz')

    # export the regridded file
    fname = fname_new
    exportuvfits(vis='sim_MS/'+fname_new+'.ms', 
                 fitsfile='sim_uvfits/'+fname_new+'.uvfits', datacolumn='data',
                 overwrite=True)


# interpolation?
if do_shift:
    # do the interpolation
    fname_new = fname+'.shift'
    os.system('rm -rf sim_MS/'+fname_new+'.ms')
    mstransform(vis='sim_MS/'+fname+'.ms', outputvis='sim_MS/'+fname_new+'.ms',
                datacolumn='data', regridms=True, mode='velocity',
                start=chanstart, width=chanwidth, nchan=nchan, outframe='LSRK',
                veltype='radio', restfreq=str(restfreq / 1e9)+'GHz')

    # export the interpolated file
    fname = fname_new
    exportuvfits(vis='sim_MS/'+fname+'.ms',
                 fitsfile='sim_uvfits/'+fname+'.uvfits', datacolumn='data',
                 overwrite=True)


# imaging? 
if do_image:
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
    exportfits('sim_images/im_'+fname, 'sim_images/im_'+fname+'.fits', 
               overwrite=True)