import os, sys
import numpy as np
from astropy.io import fits
from vis_sample import vis_sample
from vis_sample.file_handling import import_data_uvfits
sys.path.append('../')
from mk_FITScube import mk_FITScube

# template file
temp_uv  = 'cfg5_15min_dv0.1kms_v0-7.52kms_nch153'
template = 'template_uvfits/template_'+temp_uv+'.uvfits'

# output files
fout = 'simp3_std_highv_medr'



# RMS noise per naturally weighted beam per channel in output
RMS = 9.4	# in mJy

# inc, PA, mstar, Tb0, Tbq, r_l, z0, vsys, dx, dy]
theta = [40., 130., 0.7, 65., -0.5, 200., 23., 0.0, 0.0, 0.0]
FOV = 8.0
dist = 150.
Npix = 256
Tbmax = 500.
restfreq = 230.538e9
fixed = FOV, dist, Npix, Tbmax, restfreq


### - extract the velocities from the template file
dat = fits.open(template)
hdr = dat[0].header
freq0 = hdr['CRVAL4']
idx0  = hdr['CRPIX4']
nchan = hdr['NAXIS4']
dfreq = hdr['CDELT4']
freqs = freq0 + (np.arange(nchan) - idx0 + 1) * dfreq
vel = 2.99792e10 * (1. - freqs / restfreq) / 100.


### - compute a model cube (SkyImage object)
foo = mk_FITScube(inc=theta[0], PA=theta[1], mstar=theta[2], FOV=FOV, 
                  dist=dist, Npix=Npix, Tb0=theta[3], Tbq=theta[4], 
                  r_l=theta[5], z0=theta[6], vsys=theta[7], Tbmax=Tbmax, 
                  r_max=theta[5], restfreq=restfreq, vel=vel)


### - sample it on the template (u,v) spacings: NOISE FREE
os.system('rm -rf sim_uvfits/'+fout+'_noiseless.uvfits')
vis_sample(imagefile=foo, uvfile=template, mu_RA=theta[7], mu_DEC=theta[8], 
           mod_interp=False, outfile='sim_uvfits/'+fout+'_noiseless.uvfits')


### - clone and corrupt the datafile according to the desired noise
vis = import_data_uvfits('sim_uvfits/'+fout+'_noiseless.uvfits')
clone = fits.open('sim_uvfits/'+fout+'_noiseless.uvfits')
clone_data = clone[0].data
nvis, nchan = vis.VV.shape[1], vis.VV.shape[0]
clone_vis = clone_data['data']
sig_noise = 1e-3 * RMS * np.sqrt(nvis)
npol = clone[0].header['NAXIS3']
if (npol == 2):
    darr = np.zeros([vis.VV.shape[1], vis.VV.shape[0], 2, 3])
    noise1 = np.random.normal(0, sig_noise, (nvis, nchan)) + \
             np.random.normal(0, sig_noise, (nvis, nchan))*1.j
    noise2 = np.random.normal(0, sig_noise, (nvis, nchan)) + \
             np.random.normal(0, sig_noise, (nvis, nchan))*1.j
    darr[:,:,0,0] = np.real(vis.VV.T) + noise1.real * np.sqrt(2.)
    darr[:,:,1,0] = np.real(vis.VV.T) + noise2.real * np.sqrt(2.)
    darr[:,:,0,1] = np.imag(vis.VV.T) + noise1.imag * np.sqrt(2.)
    darr[:,:,1,1] = np.imag(vis.VV.T) + noise2.imag * np.sqrt(2.)
    darr[:,:,0,2] = 0.5 * np.ones_like(vis.wgts) / sig_noise**2
    darr[:,:,1,2] = 0.5 * np.ones_like(vis.wgts) / sig_noise**2
else:
    darr = np.zeros([vis.VV.shape[1], vis.VV.shape[0], 2, 3])
    noise = np.random.normal(0, sig_noise, (nvis, nchan)) + \
            np.random.normal(0, sig_noise, (nvis, nchan))*1.j
    darr[:,:,0,0] = np.real(vis.VV.T) + noise.real
    darr[:,:,0,1] = np.imag(vis.VV.T) + noise.imag
    darr[:,:,0,2] = np.ones_like(vis.wgts) / sig_noise**2

# output the NOISY clone into a UVFITS file
clone_data['data'] = np.expand_dims(np.expand_dims(np.expand_dims(darr,1),1),1)
clone.writeto('sim_uvfits/'+fout+'_noisy.uvfits', overwrite=True)


# notification
print('Wrote sim_uvfits/'+fout+'_noisy.uvfits')
