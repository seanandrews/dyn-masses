import os, sys, time
import numpy as np
from astropy.io import fits
from vis_sample import vis_sample
from vis_sample.file_handling import import_data_uvfits
import matplotlib.pyplot as plt

# parse and package the UNWINDOWED DATA
data_set = 'simp3_std_medv_medr_STARTHIV_noiseless'
data_file = 'fake_data/sim_uvfits/'+data_set+'.uvfits'
raw_vis = import_data_uvfits(data_file)

# extract the proper velocities from the data file
dat = fits.open(data_file)
hdr = dat[0].header
freq0 = hdr['CRVAL4']
indx0 = hdr['CRPIX4']
nchan = hdr['NAXIS4']
dfreq = hdr['CDELT4']
freqs = freq0 + (np.arange(nchan) - indx0 + 1) * dfreq
raw_vis.freqs = freqs
print(raw_vis.VV.shape, raw_vis.freqs.shape)


# parse and package the CASA-WINDOWED DATA
data_set = 'simp3_std_medv_medr_STARTHIV_noiseless.hann'
data_file = 'fake_data/sim_uvfits/'+data_set+'.uvfits'
casa_vis = import_data_uvfits(data_file)

# extract the proper velocities from the data file
dat = fits.open(data_file)
hdr = dat[0].header
freq0 = hdr['CRVAL4']
indx0 = hdr['CRPIX4']
nchan = hdr['NAXIS4']
dfreq = hdr['CDELT4']
freqs = freq0 + (np.arange(nchan) - indx0 + 1) * dfreq
casa_vis.freqs = freqs
print(casa_vis.VV.shape, casa_vis.freqs.shape)


# plot the unwindowed versus the casa-windowed data
idx = 1500
plt.plot(raw_vis.freqs / 1e9, raw_vis.VV.real[:,idx], 'oC0')
plt.plot(casa_vis.freqs / 1e9, casa_vis.VV.real[:,idx], 'oC1', alpha=0.8, 
         markersize=2.5)

#for i in range(len(casa_vis.freqs)):
#    print(casa_vis.VV[i,idx], raw_vis.VV[i,idx])


from scipy.signal import convolve

window = np.array([0.0, 0.25, 0.5, 0.25, 0.0])
myhann = convolve(raw_vis.VV[:,idx], window, mode='same')

plt.plot(raw_vis.freqs / 1e9, myhann.real, 'oC2', alpha=0.6, markersize=2.3)

plt.show()
plt.clf()


diff_hanns = (myhann.real - casa_vis.VV.real[:,idx]) / myhann.real
diff_raw = (myhann.real - raw_vis.VV.real[:,idx]) / myhann.real

plt.plot(raw_vis.freqs[10:-10] / 1e9, diff_hanns[10:-10], 'oC0')
plt.show()



# how to do this quickly on all visibilities?
from scipy.ndimage import convolve1d

print('here i am')
t0 = time.time()
testhann_real = convolve1d(raw_vis.VV.real, window, axis=0, mode='nearest')
testhann_imag = convolve1d(raw_vis.VV.imag, window, axis=0, mode='nearest')
print(time.time()-t0)


diff_21 = (casa_vis.VV.real[:,idx] - testhann_real[:,idx]) / casa_vis.VV.real[:,idx]
plt.plot(raw_vis.freqs[10:-10] / 1e9, diff_21[10:-10], 'oC0')
plt.show()

diff_21 = (casa_vis.VV.imag[:,idx] - testhann_imag[:,idx]) / casa_vis.VV.imag[:,idx]
plt.plot(raw_vis.freqs[10:-10] / 1e9, diff_21[10:-10], 'oC0')
plt.show()

