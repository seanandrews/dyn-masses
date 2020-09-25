import os, sys
import numpy as np
from astropy.io import fits
from scipy.interpolate import interp1d
from scipy.ndimage import convolve1d
from vis_sample import vis_sample
from vis_sample.file_handling import import_data_uvfits
import matplotlib.pyplot as plt
sys.path.append('../')
from cube_parser import cube_parser

"""
Decription of what this code does.

"""

# desired output channels
#chanstart_out = -9.6	# km/s
#chanwidth_out = 0.08	# km/s
#nchan_out = 241


# bookkeeping
template_file = 'std_medr_medv10x'
outdata_file  = 'simp3_std_medr_medv_test'

fetch_freqs = True



# RMS noise per naturally weighted beam per channel in output
RMS = 7.4	# in mJy
#RMS = 10.5	# in mJy


# Constants
c_ = 2.99792e5


# Extract TOPO / LSRK frame frequencies from template
if fetch_freqs:
    print('Computing LSRK frequencies for fixed TOPO channels...')
    f = open('template_freqs.txt', 'w')
    f.write(template_file)
    f.close()
    os.system('casa --nologger --nologfile -c fetch_freqs.py')
    print('...Finished frequency calculations.')


### Specify simulation parameters
# free parameters
#       inc, PA, mstar, r_l, z0, zpsi, Tb0, Tbq, Tbmax_b, xi_nt, vsys, dx, dy
theta = [40, 130, 0.7, 200, 2.3, 1, 205, 0.5, 20, 0.05, 0, 0, 0]
# fixed parameters
FOV, dist, Npix, Tbmax, r0, mu_l = 8.0, 150., 256, 500., 10., 28.


### Fetch some information about the template
io = np.loadtxt('template_params/'+template_file+'.params.txt', dtype=str)
restfreq, t_integ = np.float(io[2]), np.float(io[12][:-1])
ch_spacing, spec_oversampling = np.float(io[1]), np.int(io[5])


### Fetch the frequencies; convert LSRK to velocity
datf = np.load('template_params/'+template_file+'.freq_conversions.npz')
freq_TOPO = datf['freq_TOPO']
freq_LSRK = datf['freq_LSRK']
v_LSRK = c_ * (1. - freq_LSRK / restfreq)

# midpt velocities
freq_LSRK_t = datf['freq_LSRK'][:,::10].copy()
v_LSRK_t = c_ * (1. - freq_LSRK_t / restfreq)
midstamp = np.int(v_LSRK_t.shape[0] / 2)
freq_LSRK_mid, v_LSRK_mid = freq_LSRK_t[midstamp,:], v_LSRK_t[midstamp,:]

chanstart_out = v_LSRK_mid[33]
chanwidth_out = np.mean(np.diff(v_LSRK_mid))
nchan_out = 123


### Compute mock visibilities in TOPO frame
# clone the template
clone = fits.open('template_uvfits/'+template_file+'.uvfits')

# configure clone inputs
tvis = import_data_uvfits('template_uvfits/'+template_file+'.uvfits')
tvis.rfreq = np.mean(freq_TOPO)
nvis, nchan = tvis.VV.shape[1], tvis.VV.shape[0]
darr = np.zeros([nvis, nchan, 2, 3])

# number of visibilities per time stamp
nperstamp = np.int(nvis / t_integ)

# cycle through each timestamp; calculate visibilities for that set of LSRK
# frequencies; populate the appropriate part of the mock dataset
print('Computing visibilities in TOPO frame...')
for i in range(v_LSRK.shape[0]):
    # tracker
    print('timestamp '+str(i)+' / '+str(v_LSRK.shape[0]))

    # compute a model cube (SkyImage object)
    foo = cube_parser(inc=theta[0], PA=theta[1], dist=dist, mstar=theta[2],
                      r0=r0, r_l=theta[3], z0=theta[4], zpsi=theta[5],
                      Tb0=theta[6], Tbq=theta[7], Tbmax=Tbmax, Tbmax_b=theta[8],
                      xi_nt=theta[9], FOV=FOV, Npix=Npix, mu_l=mu_l,
                      Vsys=theta[10], restfreq=restfreq, vel=1e3*v_LSRK[i,:])

    # sample its Fourier Transform on the template (u,v) spacings
    mvis = vis_sample(imagefile=foo, uu=tvis.uu, vv=tvis.vv, mu_RA=theta[11],
                      mu_DEC=theta[12], mod_interp=False)

    # populate the appropriate parts of cloned array with these visibilities
    ix_lo, ix_hi = i * nperstamp, (i + 1) * nperstamp
    darr[ix_lo:ix_hi,:,0,0] = mvis.real[ix_lo:ix_hi,:]
    darr[ix_lo:ix_hi,:,1,0] = mvis.real[ix_lo:ix_hi,:]
    darr[ix_lo:ix_hi,:,0,1] = mvis.imag[ix_lo:ix_hi,:]
    darr[ix_lo:ix_hi,:,1,1] = mvis.imag[ix_lo:ix_hi,:]

# copy the weights from the original file (noiseless still)
darr[:,:,0,2], darr[:,:,1,2] = tvis.wgts, tvis.wgts

# output the clone into a UVFITS file (pre-SRF, pre-noise injection), which
# contains the appropriate LSRK data packed into the TOPO frame
cldata = np.expand_dims(np.expand_dims(np.expand_dims(darr,1),1),1)
clone[0].data['data'] = cldata
clone.writeto('sim_uvfits/'+outdata_file+'.TOPO.noiseless.pre-SRF.uvfits',
              overwrite=True)
clone.close()
print('...finished calculation of TOPO frame visibilities.')


### Spectral Response Function (SRF) convolution (in TOPO frame channels)
# Load the visibilities
dat = fits.open('sim_uvfits/'+outdata_file+'.TOPO.noiseless.pre-SRF.uvfits')
hdr = dat[0].header
vis = np.squeeze(dat[0].data['data'])
dat.close()

# Assign channel indices
chan = np.arange(len(freq_TOPO)) / spec_oversampling

# Create the SRF kernel
xmu = chan - np.mean(chan)
SRF = 0.5 * np.sinc(xmu) + 0.25 * np.sinc(xmu - 1) + 0.25 * np.sinc(xmu + 1)

# Convolution
print('Convolution with SRF kernel...')
vis_SRF = convolve1d(vis, SRF / np.sum(SRF), axis=1, mode='nearest')
print('...convolution completed')

# Decimate by over-sampling factor
vis_out = vis_SRF[:,::spec_oversampling,:,:].copy()
freqout_TOPO = freq_TOPO[::spec_oversampling].copy()
freqout_LSRK = freq_LSRK[:,::spec_oversampling].copy()


### Create a dummy/shell UVFITS file to store the outputs
# Pass variables into CASA
os.system('rm -rf dummy.txt')
f = open('dummy.txt', 'w')
f.write('sim_uvfits/'+outdata_file+'.TOPO.noiseless.pre-SRF.uvfits\n')
f.write(str(chanstart_out)+'\n'+str(chanwidth_out)+'\n')
f.write(str(nchan_out)+'\n'+str(restfreq/1e9))
f.close()
os.system('casa --nologger --nologfile -c make_dummy.py')

# Extract the output frequencies and velocities
dat = fits.open('dummy.uvfits')
dhdr = dat[0].header
nu0, ix0, dnu = dhdr['CRVAL4'], dhdr['CRPIX4'], dhdr['CDELT4'],
freq_out = nu0 + (np.arange(nchan_out) - ix0 + 1) * dnu
v_out = c_ * (1 - freq_out / restfreq)
dat.close()


### Interpolate into desired output frequency grid (in LSRK channels)
# Populate an LSRK frequency grid for easier interpolation
freqgrid_LSRK = np.zeros((vis_out.shape[0], vis_out.shape[1]))
nperstamp = np.int(vis_out.shape[0] / t_integ)
for i in range(freqout_LSRK.shape[0]):
    ix_lo, ix_hi = i * nperstamp, (i + 1) * nperstamp
    freqgrid_LSRK[ix_lo:ix_hi, :] = freqout_LSRK[i,:]

# Interpolate to this output frequency grid
# (this is what MSTRANSFORM would do)
vis_interp = np.zeros((vis_out.shape[0], len(freq_out)), dtype=complex)
for i in range(vis_out.shape[0]):
    re_int = interp1d(freqgrid_LSRK[i,:], vis_out[i,:,0,0], 
                      fill_value='extrapolate')
    im_int = interp1d(freqgrid_LSRK[i,:], vis_out[i,:,0,1],
                      fill_value='extrapolate')
    vis_interp[i,:] = re_int(freq_out) + 1j*im_int(freq_out)


### Repackage mock observations, with and without noise
# Create output arrays
outf = np.zeros((vis_interp.shape[0], vis_interp.shape[1], 2, 3))
outn = outf.copy()

# Noiseless output visibilities
outf[:,:,0,0] = vis_interp.real
outf[:,:,1,0] = vis_interp.real
outf[:,:,0,1] = vis_interp.imag
outf[:,:,1,1] = vis_interp.imag
outf[:,:,0,2] = 0.5 * np.ones_like(vis_interp.real)
outf[:,:,1,2] = 0.5 * np.ones_like(vis_interp.real)

# Noisy output visibilities
sig_noise = 1e-3 * RMS * np.sqrt(vis_out.shape[0])
noiseXX = np.random.normal(0, sig_noise, (vis_out.shape[0], nchan_out)) + \
          np.random.normal(0, sig_noise, (vis_out.shape[0], nchan_out))*1j
noiseYY = np.random.normal(0, sig_noise, (vis_out.shape[0], nchan_out)) + \
          np.random.normal(0, sig_noise, (vis_out.shape[0], nchan_out))*1j
outn[:,:,0,0] = vis_interp.real + noiseXX.real * np.sqrt(2.)
outn[:,:,1,0] = vis_interp.real + noiseYY.real * np.sqrt(2.)
outn[:,:,0,1] = vis_interp.imag + noiseXX.imag * np.sqrt(2.)
outn[:,:,1,1] = vis_interp.imag + noiseYY.imag * np.sqrt(2.)
outn[:,:,0,2] = 0.5 * np.ones_like(vis_interp.real) / sig_noise**2
outn[:,:,1,2] = 0.5 * np.ones_like(vis_interp.real) / sig_noise**2


### Output
clonef = fits.open('dummy.uvfits')
cfdata = np.expand_dims(np.expand_dims(np.expand_dims(outf, 1), 1), 1)
clonef[0].data['data'] = cfdata
clonef.writeto('sim_uvfits/'+outdata_file+'_noiseless.uvfits', overwrite=True)
clonef.close()

clonen = fits.open('dummy.uvfits')
cndata = np.expand_dims(np.expand_dims(np.expand_dims(outn, 1), 1), 1)
clonen[0].data['data'] = cndata
clonen.writeto('sim_uvfits/'+outdata_file+'_noisy.uvfits', overwrite=True)
clonen.close()

### Cleanup
os.system('rm -rf dummy.uvfits')
