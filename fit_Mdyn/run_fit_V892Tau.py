import os, sys, time
import numpy as np
import copy as copy
import scipy.constants as sc
from astropy.io import fits
from cube_parser import cube_parser
from vis_sample import vis_sample
from vis_sample.file_handling import import_data_uvfits
from scipy.ndimage import convolve1d
from scipy.interpolate import interp1d
import emcee
from multiprocessing import Pool
os.environ["OMP_NUM_THREADS"] = "1"


### FILE LOADING FROM SERVER
### -------------------------

# working_dir = '/Users/justinvega/Documents/GitHub/dyn-masses/fit_Mdyn/'
# filename_fits = 'V892Tau_data.uvfits'
# filename_npz = 'V892Tau.freq_conversions.npz'
# wwwfits = 'https://www.cfa.harvard.edu/~sandrews/data/'
#
# import urllib.request
# if not os.path.isdir(working_dir+'fake_data/data_uvfits/'):
#     os.mkdir(working_dir+'fake_data/data_uvfits/')
# if not os.path.exists(working_dir+'fake_data/data_uvfits/'+filename_fits):
#     print('Downloading UVFits...19 MB')
#     urllib.request.urlretrieve(wwwfits+filename_fits, working_dir+'fake_data/data_uvfits/'+filename_fits)
# if not os.path.exists(working_dir+'fake_data/data_uvfits/'+filename_npz):
#     print('Downloading Freq Conversions...32 K')
#     urllib.request.urlretrieve(wwwfits+filename_npz, 'fake_data/data_uvfits/'+filename_npz)



### ASSIGN DATA TO FIT
### ------------------
# locate data
datadir  = 'fake_data/data_uvfits/'
datafile = 'V892Tau_data'
suffix = '_posinc'

# velocity range to fit
vlo, vhi = -7., 23.	# low and high LSRK velocities to fit [km/s]
vclo, vchi = 6.32, 8.24
# --> censored ranges should go here too

# spectral line information
nu_l = 230.538e9	# rest frequency of line [Hz]

# spectral signal processing
chbin = 2               # number of channels for binned averaging
chpad = 3               # number of channels to pad for SRF convolution



############


### CONSTANTS
### ---------
c_ = 2.99792e8          # speed of light [m/s]


### PROCESS DATA
### ------------
# load data visibilities with native channel spacings (LSRK)
data = import_data_uvfits(datadir+datafile+'.uvfits')

# extract the native channel frequencies, convert to LSRK velocities [m/s]
hdr = fits.open(datadir+datafile+'.uvfits')[0].header
freq0, idx0, nchan = hdr['CRVAL4'], hdr['CRPIX4'], hdr['NAXIS4']
data.freqs = freq0 + (np.arange(nchan) - idx0 + 1) * hdr['CDELT4']
vlsrk_native = c_ * (1. - data.freqs / nu_l)

# identify the subset of channel indices in the desired velocity range
vhi_idx = np.min(np.where(vlsrk_native < vlo * 1e3))
vlo_idx = np.max(np.where(vlsrk_native > vhi * 1e3)) + 1
Nch = np.abs(vhi_idx - vlo_idx)

# extract the subset of native channels of interest, padded for windowing
data.VV = data.VV[vlo_idx-chpad:vhi_idx+chpad,:]
data.wgts = data.wgts[:,vlo_idx-chpad:vhi_idx+chpad].T
data.freqs = data.freqs[vlo_idx-chpad:vhi_idx+chpad]
vlsrk_native = c_ * (1. - data.freqs / nu_l)
data.rfreq = np.mean(data.freqs)


# find the LSRK velocities that correspond to the midpoint of the execution
# block (*HARD-CODED: still need to figure this out for real data*)
#
freq_npz = 'V892Tau.freq_conversions'
df = np.load('fake_data/data_uvfits/' + freq_npz + '.npz')
freq_LSRK_t = df['freq_LSRK'].copy()
v_LSRK_t = c_ * (1. - freq_LSRK_t / nu_l)
midstamp = np.int(v_LSRK_t.shape[0] / 2)
freq_LSRK_mid, v_LSRK_mid = freq_LSRK_t[midstamp,:], v_LSRK_t[midstamp,:]

# grab only the subset of channels that span our desired outputs
vhi_idx = np.min(np.where(v_LSRK_mid < np.min(vlsrk_native))) #- 1
vlo_idx = np.max(np.where(v_LSRK_mid > np.max(vlsrk_native))) + 1
v_LSRK_mid = v_LSRK_mid[vlo_idx:vhi_idx]
freq_LSRK_mid = freq_LSRK_mid[vlo_idx:vhi_idx]


# make a copy of the input (native) data to bin
data_bin = copy.deepcopy(data)

# clip the unpadded data, so divisible by factor chbin
data_bin.VV = data_bin.VV[chpad:chpad+Nch-(Nch % chbin),:]
data_bin.wgts = data_bin.wgts[chpad:chpad+Nch-(Nch % chbin),:]
data_bin.freqs = data_bin.freqs[chpad:chpad+Nch-(Nch % chbin)]

# identify which binned channels are censored (FALSE)
data_bin_vlsrk = c_ * (1 - data_bin.freqs / nu_l)
cens_chans = np.ones_like(data_bin_vlsrk, dtype='bool')
badc = (data_bin_vlsrk >= 1e3 * vclo) & (data_bin_vlsrk <= 1e3 * vchi)
cens_chans[badc] = 0
donotuse = np.any(cens_chans.reshape((-1, chbin)), axis=1)

# binning (weighted, decimated average)
avg_wts = data_bin.wgts.reshape((-1, chbin, data_bin.wgts.shape[1]))
data_bin.VV = np.average(data_bin.VV.reshape((-1, chbin, data_bin.VV.shape[1])),
                         weights=avg_wts, axis=1)
data_bin.wgts = np.sum(avg_wts, axis=1)
data_bin.freqs = np.average(data_bin.freqs.reshape(-1, chbin), axis=1)
data_bin.rfreq = np.mean(data_bin.freqs)
Nch_bin = len(data_bin.freqs)


### PRECALCULATED QUANTITIES
### ------------------------
# covariance matrix and its inverse
Mbin = (5./16.)*np.eye(Nch_bin) + \
       (3./32.)*(np.eye(Nch_bin, k=-1) + np.eye(Nch_bin, k=1))
Mbin_inv = np.linalg.inv(Mbin)

# log-likelihood normalization constant
dterm = np.empty(data_bin.VV.shape[1])
for i in range(len(dterm)):
    sgn, lndet = np.linalg.slogdet(Mbin / data_bin.wgts[:,i])
    dterm[i] = sgn * lndet
L0 = -0.5 * (np.prod(data_bin.VV.shape) * np.log(2 * np.pi) + np.sum(dterm))


# now censor the appropriate binned channels (set weights = 0)
data_bin.wgts[donotuse == False,:] = 0




### INITIALIZE FOR POSTERIOR SAMPLING
### ---------------------------------
# fixed model parameters
FOV, dist, Npix, Tbmax, r0 = 6.0, 134.5, 256, 1500., 10. # previously dist = 130, Tbmax = 700

# initialize walkers
p_lo = np.array([ 45, 40, 4.5, 150, 0.1,  0.5,  150,
                 0.2,   5, 7.5, -0.18,  0.10])
p_hi = np.array([ 65, 60, 6.5, 350,   5,  1.5,  350,
                 0.8,  30, 9.0, -0.08,  0.20])
ndim, nwalk = len(p_lo), 5 * len(p_lo)
p0 = [np.random.uniform(p_lo, p_hi, ndim) for i in range(nwalk)]

# 1 model to set up GCF, corr caches
theta = p0[0]

# sound speed
mu_l, mH = 28, sc.m_p + sc.m_e
csound = np.sqrt(2 * sc.k * theta[6] / (mu_l * mH))

foo = cube_parser(inc=theta[0], PA=theta[1], dist=dist, mstar=theta[2], r0=r0,
                  r_l=theta[3], z0=theta[4], zpsi=theta[5],
                  Tb0=theta[6], Tbq=theta[7], Tbmax=Tbmax, Tbmax_b=theta[8],
                  dV0=csound, dVq=0.5*theta[7], FOV=FOV, Npix=Npix,
                  Vsys=theta[9], restfreq=nu_l, vel=v_LSRK_mid)

tvis, gcf, corr = vis_sample(imagefile=foo, uu=data.uu, vv=data.vv,
                             return_gcf=True, return_corr_cache=True,
                             mod_interp=False)


### PRIOR FUNCTIONAL FORMS
### ----------------------
# uniform
def lnpU(theta, lo, hi):
    if ((theta >= lo) and (theta <= hi)):
        return 0
    else:
        return -np.inf

# normal
def lnpN(theta, mu, sig):
    return -0.5 * np.exp(-0.5 * (theta - mu)**2 / sig**2)

# normal + uniform
def lnpNU(theta, mu, sig, lo, hi):
    if ((theta < lo) or (theta > hi)):
        return -np.inf
    else:
        return -0.5 * np.exp(-0.5 * (theta - mu)**2 / sig**2)

### LOG(POSTERIOR)
### --------------
def lnprob(theta):

    # calculate prior
    ptheta = np.empty_like(theta)
    ptheta[0] = lnpNU(theta[0], 54.5, 2., 0., 90.) #i (for negative inclination: lnpNU(theta[0], -54.5, 2., -90., 0.))
    ptheta[1] = lnpNU(theta[1], 52.1, 2., 0., 360.) #PA
    ptheta[2] = lnpU(theta[2], 0., 10.) #m
    ptheta[3] = lnpNU(theta[3], 230., 30., r0, 0.5*(dist * FOV)) #r_l
    ptheta[4] = lnpNU(theta[4], 1.0, 0.2, 0.1, 10.) #z0
    ptheta[5] = lnpNU(theta[5], 1.0, 0.2, 0., 1.5) #zpsi
    ptheta[6] = lnpU(theta[6], 5., Tbmax) #Tb0
    ptheta[7] = lnpU(theta[7], 0., 2.) #Tbq
    ptheta[8] = lnpNU(theta[8], 20., 2., 5., 50.) #Tback
    ptheta[9] = lnpNU(theta[9], 8.2, 0.1, 7.5, 9.0) #v_sys
    ptheta[10] = lnpNU(theta[10], -0.13, 0.02, -0.25, 0.0) #dx
    ptheta[11] = lnpNU(theta[11],  0.15, 0.02, 0.0, 0.25) #dy
    lnprior = np.sum(ptheta)
    if (lnprior == -np.inf):
        return -np.inf, -np.inf

    # calculate sound speed
    csound = np.sqrt(2 * sc.k * theta[6] / (mu_l * mH))

    # generate a model cube
    mcube = cube_parser(inc=theta[0], PA=theta[1], dist=dist, r0=r0,
                        mstar=theta[2], r_l=theta[3], z0=theta[4],
                        zpsi=theta[5], Tb0=theta[6], Tbq=theta[7],
                        Tbmax=Tbmax, Tbmax_b=theta[8], dV0=csound,
                        dVq=0.5*theta[7], FOV=FOV, Npix=Npix,
                        Vsys=theta[9], restfreq=nu_l, vel=v_LSRK_mid)

    # sample the FT of the cube onto the observed (u,v) points
    mvis = vis_sample(imagefile=mcube, mu_RA=theta[10], mu_DEC=theta[11],
                      gcf_holder=gcf, corr_cache=corr, mod_interp=False)

    # window the visibilities
    SRF_kernel = np.array([0.0, 0.25, 0.5, 0.25, 0.0])
    mvis_re = convolve1d(mvis.real, SRF_kernel, axis=1, mode='nearest')
    mvis_im = convolve1d(mvis.imag, SRF_kernel, axis=1, mode='nearest')
    mvis = mvis_re + 1.0j*mvis_im

    # interpolation
    fint = interp1d(freq_LSRK_mid, mvis, axis=1, fill_value='extrapolate')
    mvis = fint(data.freqs)

    # excise the padded boundary channels to avoid edge effects
    mvis = mvis[:,chpad:-chpad].T
    mwgt = data.wgts[chpad:-chpad,:]

    # clip for binning
    mvis = mvis[:mvis.shape[0]-(mvis.shape[0] % chbin),:]
    mwgt = mwgt[:mvis.shape[0]-(mvis.shape[0] % chbin),:]

    # bin (weighted, decimated average)
    mvis_bin = np.average(mvis.reshape((-1, chbin, mvis.shape[1])),
                          weights=mwgt.reshape((-1, chbin, mwgt.shape[1])),
                          axis=1)

    # compute the log-likelihood
    resid = np.absolute(data_bin.VV - mvis_bin)
    lnL = -0.5 * np.tensordot(resid, np.dot(Mbin_inv, data_bin.wgts * resid))

    # return the posterior
    return lnL + L0 + lnprior, lnprior


### CONFIGURE EMCEE BACKEND
### -----------------------
filename = 'posteriors/'+datafile+suffix+'.h5'
#os.system('rm -rf '+filename)
backend = emcee.backends.HDFBackend(filename)
#backend.reset(nwalk, ndim)

# run the sampler
max_steps = 7000
with Pool() as pool:
    sampler = emcee.EnsembleSampler(nwalk, ndim, lnprob, pool=pool,
                                    backend=backend)
    t0 = time.time()
    sampler.run_mcmc(p0, max_steps, progress=True)
t1 = time.time()

print(' ')
print(' ')
print(' ')
print('This run took %.2f hours' % ((t1 - t0) / 3600))
