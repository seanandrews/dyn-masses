import os, sys
import numpy as np
from astropy.io import fits
from vis_sample import vis_sample
from vis_sample.file_handling import import_data_uvfits
sys.path.append('../')
from cube_parser import cube_parser

# template file
temp_uv  = 'template_ch15ms_V0-9699ms_nchan1221_config5_dt15min_tinteg30s'
template = 'template_uvfits/'+temp_uv+'.uvfits'

# output files
fout = 'simp3_std_medv_medr_10xHIGHV'



# RMS noise per naturally weighted beam per channel in output
RMS = 7.4	# in mJy

# inc, PA, mstar, r_l, z0, zpsi, Tb0, Tbq, Tbmax_b, xi_nt, vsys, dx, dy]
theta = [40., 130., 0.7, 200., 2.3, 1.0, 205., 0.5, 20., 0.05, 0., 0., 0.]
FOV = 8.0
dist = 150.
Npix = 256
Tbmax = 500.
r0 = 10.
mu_l = 28.
restfreq = 230.538e9

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
foo = cube_parser(inc=theta[0], PA=theta[1], dist=dist, mstar=theta[2], r0=r0, 
                  r_l=theta[3], z0=theta[4], zpsi=theta[5], 
                  Tb0=theta[6], Tbq=theta[7], Tbmax=Tbmax, Tbmax_b=theta[8],
                  xi_nt=theta[9], FOV=FOV, Npix=Npix, mu_l=mu_l, 
                  Vsys=theta[10], restfreq=restfreq, vel=vel)


### - sample it on the template (u,v) spacings: NOISE FREE
os.system('rm -rf sim_uvfits/'+fout+'_noiseless.uvfits')
vis_sample(imagefile=foo, uvfile=template, mu_RA=theta[11], mu_DEC=theta[12], 
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
