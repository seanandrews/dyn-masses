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
chanstart_out = -5.6	# km/s
chanwidth_out = 0.08	# km/s
nchan_out = 241


# bookkeeping
template_file = 'std_medr_highv10x'
outdata_file = 'simp3_std_medr_highv_1024pix'
fetch_freqs = True



# RMS noise per naturally weighted beam per channel in output
#RMS = 7.44	# in mJy (appropriate for medr_medv)
RMS = 10.52     # in mJy (appropriate for medr_highv)

### Specify simulation parameters
# free parameters
#       inc, PA, mstar, r_l, z0, zpsi, Tb0, Tbq, Tbmax_b, dV0, dVq, vsys, dx, dy
theta = [40, 130, 0.7, 200, 2.3, 1, 205, 0.5, 20, 347.6, 0.25, 4.0, 0, 0]
# fixed parameters
FOV, dist, Npix, Tbmax, r0 = 8.0, 150., 1024, 500., 10.



###########

# Constants
c_ = 2.99792e5


### Extract TOPO / LSRK frame frequencies from template
if fetch_freqs:
    print('Computing LSRK frequencies for fixed TOPO channels...')
    f = open('template_freqs.txt', 'w')
    f.write(template_file)
    f.close()
    os.system('casa --nologger --nologfile -c CASA_scripts/fetch_freqs.py')
    print('...Finished frequency calculations.')

io = np.loadtxt('template_params/'+template_file+'.params.txt', dtype=str)
restfreq, t_integ = np.float(io[2]), np.float(io[12][:-1])
ch_spacing, spec_oversampling = np.float(io[1]), np.int(io[5])

datf = np.load('template_params/'+template_file+'.freq_conversions.npz')
freq_TOPO = datf['freq_TOPO']
freq_LSRK = datf['freq_LSRK']
v_LSRK = c_ * (1. - freq_LSRK / restfreq)


### Compute mock visibilities on the fixed TOPO-frame channels (w/ + w/o noise)
# configure output arrays, based on template formatting
tvis = import_data_uvfits('template_uvfits/'+template_file+'.uvfits')
tvis.rfreq = np.mean(freq_LSRK)    # mean frequency for FT calculation
nvis, nchan = tvis.VV.shape[1], tvis.VV.shape[0]   
clean_arr = np.zeros([nvis, nchan, 2, 3])
noisy_arr = np.zeros([nvis, nchan, 2, 3])

# number of visibilities per time stamp
nperstamp = np.int(nvis / t_integ)

# configure noise                                 
sigma_out = 1e-3 * RMS * np.sqrt(2 * nvis)
sigma_noise = sigma_out * np.sqrt(np.pi * spec_oversampling) 
noiseXX = np.random.normal(0, sigma_noise, (nvis, nchan)) + \
          np.random.normal(0, sigma_noise, (nvis, nchan))*1j
noiseYY = np.random.normal(0, sigma_noise, (nvis, nchan)) + \
          np.random.normal(0, sigma_noise, (nvis, nchan))*1j

# cycle through each timestamp; calculate visibilities for that set of LSRK
# frequencies; populate the appropriate part of the mock dataset
print('Computing visibilities in TOPO frame...')
for i in range(v_LSRK.shape[0]):
    print('timestamp '+str(i)+' / '+str(v_LSRK.shape[0]))

    # compute a model cube (SkyImage object)
    foo = cube_parser(inc=theta[0], PA=theta[1], dist=dist, mstar=theta[2],
                      r0=r0, r_l=theta[3], z0=theta[4], zpsi=theta[5],
                      Tb0=theta[6], Tbq=theta[7], Tbmax=Tbmax, Tbmax_b=theta[8],
                      dV0=theta[9], dVq=theta[10], FOV=FOV, Npix=Npix, 
                      Vsys=theta[11], restfreq=restfreq, vel=1e3*v_LSRK[i,:])

    # sample its Fourier Transform on the template (u,v) spacings
    mvis = vis_sample(imagefile=foo, uu=tvis.uu, vv=tvis.vv, mu_RA=theta[12],
                      mu_DEC=theta[13], mod_interp=False)

    # populate the appropriate parts of cloned array with these visibilities
    ix_lo, ix_hi = i * nperstamp, (i + 1) * nperstamp
    clean_arr[ix_lo:ix_hi,:,0,0] = mvis.real[ix_lo:ix_hi,:]
    clean_arr[ix_lo:ix_hi,:,1,0] = mvis.real[ix_lo:ix_hi,:]
    clean_arr[ix_lo:ix_hi,:,0,1] = mvis.imag[ix_lo:ix_hi,:]
    clean_arr[ix_lo:ix_hi,:,1,1] = mvis.imag[ix_lo:ix_hi,:]

    # the same, but with noise added
    noisy_arr[ix_lo:ix_hi,:,0,0] = mvis.real[ix_lo:ix_hi,:] + \
                                   noiseXX.real[ix_lo:ix_hi,:]
    noisy_arr[ix_lo:ix_hi,:,1,0] = mvis.real[ix_lo:ix_hi,:] + \
                                   noiseYY.real[ix_lo:ix_hi,:]
    noisy_arr[ix_lo:ix_hi,:,0,1] = mvis.imag[ix_lo:ix_hi,:] + \
                                   noiseXX.imag[ix_lo:ix_hi,:]
    noisy_arr[ix_lo:ix_hi,:,1,1] = mvis.imag[ix_lo:ix_hi,:] + \
                                   noiseYY.imag[ix_lo:ix_hi,:]

print('...finished calculation of TOPO frame visibilities.')


### Spectral Response Function (SRF) convolution
# Assign channel indices
chan = np.arange(len(freq_TOPO)) / spec_oversampling

# Create the SRF kernel
xmu = chan - np.mean(chan)
SRF = 0.5 * np.sinc(xmu) + 0.25 * np.sinc(xmu - 1) + 0.25 * np.sinc(xmu + 1)

# Convolution
print('Convolution with SRF kernel...')
fvis_SRF = convolve1d(clean_arr, SRF / np.sum(SRF), axis=1, mode='nearest')
nvis_SRF = convolve1d(noisy_arr, SRF / np.sum(SRF), axis=1, mode='nearest')
print('...convolution completed')


### Decimate by over-sampling factor
fvis_out = fvis_SRF[:,::spec_oversampling,:,:].copy()
nvis_out = nvis_SRF[:,::spec_oversampling,:,:].copy()
freqout_TOPO = freq_TOPO[::spec_oversampling].copy()
freqout_LSRK = freq_LSRK[:,::spec_oversampling].copy()


### Create a dummy/shell UVFITS file on desired output LSRK channels
# Pass variables into CASA
os.system('rm -rf dummy.txt')
f = open('dummy.txt', 'w')
f.write('template_uvfits/'+template_file+'.uvfits\n')
f.write(str(chanstart_out)+'\n'+str(chanwidth_out)+'\n')
f.write(str(nchan_out)+'\n'+str(restfreq/1e9))
f.close()
os.system('casa --nologger --nologfile -c CASA_scripts/make_dummy.py')

# Extract the output frequencies
dum = fits.open('dummy.uvfits')
dhdr = dum[0].header
nu0, ix0, dnu = dhdr['CRVAL4'], dhdr['CRPIX4'], dhdr['CDELT4']
freq_out = nu0 + (np.arange(nchan_out) - ix0 + 1) * dnu
dum.close()


### Interpolate into desired output LSRK frequency channels 
# Populate an LSRK frequency grid for easier interpolation
freqgrid_LSRK = np.zeros((fvis_out.shape[0], fvis_out.shape[1]))
nperstamp = np.int(fvis_out.shape[0] / t_integ)
for i in range(freqout_LSRK.shape[0]):
    ix_lo, ix_hi = i * nperstamp, (i + 1) * nperstamp
    freqgrid_LSRK[ix_lo:ix_hi,:] = freqout_LSRK[i,:]

# Interpolate to the output frequency grid
# (this is what MSTRANSFORM would do)
clean_out = np.zeros((nvis, nchan_out, 2, 3))
noisy_out = clean_out.copy()
for i in range(nvis):
    fvis_interp = interp1d(freqgrid_LSRK[i,:], fvis_out[i,:,:,:], axis=0,
                           fill_value='extrapolate')
    clean_out[i,:,:,:] = fvis_interp(freq_out)

    nvis_interp = interp1d(freqgrid_LSRK[i,:], nvis_out[i,:,:,:], axis=0,
                           fill_value='extrapolate')
    noisy_out[i,:,:,:] = nvis_interp(freq_out)

# Set the visibility weights to proper values
wgts = np.ones((nvis, nchan_out)) / sigma_out**2
clean_out[:,:,0,2] = wgts
clean_out[:,:,1,2] = wgts
noisy_out[:,:,0,2] = wgts
noisy_out[:,:,1,2] = wgts


### Output
clonef = fits.open('dummy.uvfits')
cfdata = np.expand_dims(np.expand_dims(np.expand_dims(clean_out, 1), 1), 1)
clonef[0].data['data'] = cfdata
clonef.writeto('sim_uvfits/'+outdata_file+'_noiseless.uvfits', overwrite=True)
clonef.close()

clonen = fits.open('dummy.uvfits')
cndata = np.expand_dims(np.expand_dims(np.expand_dims(noisy_out, 1), 1), 1)
clonen[0].data['data'] = cndata
clonen.writeto('sim_uvfits/'+outdata_file+'_noisy.uvfits', overwrite=True)
clonen.close()


### Cleanup
os.system('rm -rf dummy.uvfits')
