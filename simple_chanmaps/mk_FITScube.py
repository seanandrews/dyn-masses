import numpy as np
import time
import sys
from astropy.io import fits
from vis_sample.classes import *
from simple_disk import simple_disk

def mk_FITScube(inc=45., PA=90., mstar=1.0, FOV=5., dist=150., Npix=256,
                Tb0=150., Tbq=-1.0, r_max=250., vsys=0., Tbmax=300., vel=None,
                datafile=None, restfreq=230.538e9, RA=65., DEC=25., 
                outfile=None):


    # constants
    CC = 2.9979245800000e10
    KK = 1.3807e-16


    # generate an emission model
    disk = simple_disk(inc=inc, PA=PA, mstar=mstar, FOV=FOV, dist=dist, 
                       Npix=Npix, Tb0=Tb0, Tbq=Tbq, r_max=r_max, Tbmax=Tbmax)


    # decide on velocities
    if datafile is not None:
        # load datafile header
        dat = fits.open(datafile)
        hdr = dat[0].header

        # frequencies
        freq0 = hdr['CRVAL4']
        indx0 = hdr['CRPIX4']
        nchan = hdr['NAXIS4']
        dfreq = hdr['CDELT4']
        freqs = freq0 + (np.arange(nchan) - indx0 + 1) * dfreq

        # velocities
        vel = CC * (1. - freqs / restfreq) / 100.
    else:
        freqs = restfreq * (1. - vel / (CC / 100.))     


    # adjust for systemic velocity
    vlsr = vel - (vsys * 1000.)


    # generate channel maps
    cube = disk.get_cube(vlsr)


    # convert from brightness temperatures to Jy / pixel
    pixel_area = (disk.cell_sky * np.pi / (180. * 3600.))**2
    for i in range(len(freqs)):
        cube[i,:,:] *= 1e23 * pixel_area * 2 * freqs[i]**2 * KK / CC**2


    # if an 'outfile' specified, pack the cube into a FITS file 
    if outfile is not None:
        hdu = fits.PrimaryHDU(cube[:,::-1,:])
        header = hdu.header
    
        # basic header inputs
        header['EPOCH'] = 2000.
        header['EQUINOX'] = 2000.
        header['LATPOLE'] = -1.436915713634E+01
        header['LONPOLE'] = 180.

        # spatial coordinates
        header['CTYPE1'] = 'RA---SIN'
        header['CUNIT1'] = 'DEG'
        header['CDELT1'] = -disk.cell_sky / 3600.
        header['CRPIX1'] = 0.5 * disk.Npix + 0.5
        header['CRVAL1'] = RA
        header['CTYPE2'] = 'DEC--SIN'
        header['CUNIT2'] = 'DEG'
        header['CDELT2'] = disk.cell_sky / 3600.
        header['CRPIX2'] = 0.5 * disk.Npix + 0.5
        header['CRVAL2'] = DEC

        # frequency coordinates
        header['CTYPE3'] = 'FREQ'
        header['CUNIT3'] = 'Hz'
        header['CRPIX3'] = 1.
        header['CDELT3'] = freqs[1]-freqs[0]
        header['CRVAL3'] = freqs[0]
        header['SPECSYS'] = 'LSRK'
        header['VELREF'] = 257

        # intensity units
        header['BSCALE'] = 1.
        header['BZERO'] = 0.
        header['BUNIT'] = 'JY/PIXEL'
        header['BTYPE'] = 'Intensity'

        # output FITS
        hdu.writeto(outfile, overwrite=True)

        return cube[:,::-1,:]

    # otherwise, return a vis_sample SkyObject
    else:
        # adjust cube formatting
        mod_data = np.rollaxis(cube[:,::-1,:], 0, 3)

        # spatial coordinates
        npix_ra = disk.Npix
        mid_pix_ra = 0.5 * disk.Npix + 0.5
        delt_ra = -disk.cell_sky / 3600.
        if (delt_ra < 0):
            mod_data = np.fliplr(mod_data)
        mod_ra = (np.arange(npix_ra) - (mid_pix_ra-0.5))*np.abs(delt_ra)*3600.
        
        npix_dec = disk.Npix
        mid_pix_dec = 0.5 * disk.Npix + 0.5
        delt_dec = disk.cell_sky / 3600.
        if (delt_dec < 0):
            mod_data = np.flipud(mod_data)
        mod_dec = (np.arange(npix_dec)-(mid_pix_dec-0.5))*np.abs(delt_dec)*3600.

        # spectral coordinates
        try:
            nchan_freq = len(freqs)
            mid_chan_freq = freqs[0]
            mid_chan = 1
            delt_freq = freqs[1] - freqs[0]
            mod_freqs = (np.arange(nchan_freq)-(mid_chan-1))*delt_freq + \
                        mid_chan_freq
        except:
            mod_freqs = [0]

        # return a vis_sample SkyImage object
        return SkyImage(mod_data, mod_ra, mod_dec, mod_freqs, None)
