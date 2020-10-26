import os, sys, time
import numpy as np
import copy as copy
from astropy.io import fits
from cube_parser import cube_parser
from vis_sample import vis_sample
from vis_sample.file_handling import import_data_uvfits
from scipy.ndimage import convolve1d
from scipy.interpolate import interp1d
import dynesty
import pickle
from multiprocessing import Pool

### FILE LOADING FROM SERVER
### -------------------------
# locate working directory for pickling output
wdir = '/Users/justinvega/Documents/GitHub/dyn-masses/fit_Mdyn/pickles/'

# working_dir = '/Users/justinvega/Documents/GitHub/dyn-masses/fit_Mdyn/'
# filename_fits = 'simp3_std_medr_medv_noiseless.uvfits'
# filename_npz = 'std_medr_medv10x.freq_conversions.npz'
# wwwfits = 'https://www.cfa.harvard.edu/~sandrews/data/'
# wwwnpz = 'https://www.cfa.harvard.edu/~sandrews/data/'

# import urllib.request
# if not os.path.isdir(working_dir+'fake_data/sim_uvfits/'):
#     os.mkdir(working_dir+'fake_data/sim_uvfits/')
# if not os.path.exists(working_dir+'fake_data/sim_uvfits/'+filename_fits):
#     print('Downloading UVFits...76 MB')
#     urllib.request.urlretrieve(wwwfits+filename_fits, working_dir+'fake_data/sim_uvfits/'+filename_fits)
# if not os.path.exists(working_dir+'fake_data/template_params/'+filename_npz):
#     print('Downloading NPZ... 336KB')
#     urllib.request.urlretrieve(wwwnpz+filename_npz, working_dir+'fake_data/template_params/'+filename_npz)


### ASSIGN DATA TO FIT
### ------------------
# locate data
datadir  = 'fake_data/sim_uvfits/'
datafile = 'simp3_std_medr_medv_noiseless'

# this is the "truth"!
theta_true = [40, 130, 0.7, 200, 2.3, 1, 205, 0.5, 20, 347.6, 4.0, 0, 0]


# velocity range to fit
vlo, vhi = 3, 5.	# low and high LSRK velocities to fit [km/s] (default -1, 9)
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
vlo_idx = np.max(np.where(vlsrk_native < vlo * 1e3))
vhi_idx = np.min(np.where(vlsrk_native > vhi * 1e3)) + 1
Nch = vhi_idx - vlo_idx

# extract the subset of native channels of interest, padded for windowing
data.VV = data.VV[vlo_idx-chpad:vhi_idx+chpad,:]
data.wgts = data.wgts[:,vlo_idx-chpad:vhi_idx+chpad].T
data.freqs = data.freqs[vlo_idx-chpad:vhi_idx+chpad]
vlsrk_native = c_ * (1. - data.freqs / nu_l)
data.rfreq = np.mean(data.freqs)

# find the LSRK velocities that correspond to the midpoint of the execution
# block (*HARD-CODED: still need to figure this out for real data*)
#
#template_name = '_'.join(datafile.split('_')[1:-1])+'10x'
template_name = 'std_medr_medv10x'
df = np.load('fake_data/template_params/'+template_name+'.freq_conversions.npz')
freq_LSRK_t = df['freq_LSRK'][:,::10].copy()
v_LSRK_t = c_ * (1. - freq_LSRK_t / nu_l)
midstamp = np.int(v_LSRK_t.shape[0] / 2)
freq_LSRK_mid, v_LSRK_mid = freq_LSRK_t[midstamp,:], v_LSRK_t[midstamp,:]

# grab only the subset of channels that span our desired outputs
vlo_idx = np.max(np.where(v_LSRK_mid < np.min(vlsrk_native))) - 1
vhi_idx = np.min(np.where(v_LSRK_mid > np.max(vlsrk_native))) + 1
v_LSRK_mid = v_LSRK_mid[vlo_idx:vhi_idx]
freq_LSRK_mid = freq_LSRK_mid[vlo_idx:vhi_idx]

# make a copy of the input (native) data to bin
data_bin = copy.deepcopy(data)

# clip the unpadded data, so divisible by factor chbin
data_bin.VV = data_bin.VV[chpad:chpad+Nch-(Nch % chbin),:]
data_bin.wgts = data_bin.wgts[chpad:chpad+Nch-(Nch % chbin),:]
data_bin.freqs = data_bin.freqs[chpad:chpad+Nch-(Nch % chbin)]

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



### INITIALIZE FOR POSTERIOR SAMPLING
### ---------------------------------
# fixed model parameters
#FOV, dist, Npix, Tbmax, r0 = 8.0, 150., 256, 500., 10.
FOV, dist, Npix, Tbmax, r0 = 8.0, 150., 256, 500., 10.

# initialize walkers
p_lo = np.array([ 30, 120, 0.5, 100,   0,  0.5,  155,
                 0.2,   5, 300, 3.5, -0.1, -0.1])
p_hi = np.array([ 50, 140, 0.9, 300,   5,  1.5,  255,
                 0.8,  30, 400, 4.5,  0.1,  0.1])
ndim, nwalk = len(p_lo), 5 * len(p_lo)
p0 = [np.random.uniform(p_lo, p_hi, ndim) for i in range(nwalk)]

# compute 1 model to set up GCF, corr caches
theta = p0[0]
foo = cube_parser(inc=theta[0], PA=theta[1], dist=dist, mstar=theta[2], r0=r0,
                  r_l=theta[3], z0=theta[4], zpsi=theta[5],
                  Tb0=theta[6], Tbq=theta[7], Tbmax=Tbmax, Tbmax_b=theta[8],
                  dV0=theta[9], dVq=0.5*theta[7], FOV=FOV, Npix=Npix,
                  Vsys=theta[10], restfreq=nu_l, vel=v_LSRK_mid)

tvis, gcf, corr = vis_sample(imagefile=foo, uu=data.uu, vv=data.vv,
                             return_gcf=True, return_corr_cache=True,
                             mod_interp=False)


### PRIOR FUNCTIONAL FORMS
### ----------------------
# uniform
def pUform(u):
    """Transforms the uniform random variable `u ~ Unif[0., 1.)`
    to the parameter of interest `x ~ Unif[lo, hi)`."""

    x = np.array(u)

    x[0] *= 90                    # i, scale to [0, 90.)
    x[1] *= 360                   # PA, scale to [0, 360.)
    x[2] *= 5                     # M*, scale to [0, 5.)
    x[3] = (0.5*(dist * FOV)-r0) * u[6] + r0 # r_l, scale to [r0, 0.5*(dist * FOV))
    x[4] = 3 * u[4] + 1           # z_0, scale to [1, 4) (originally [0, 10.) in emcee)
    x[5] *= 1.5                   # z_psi, scale to [0, 1.5.)
    x[6] = (200) * u[6] + 100     # Tb0, scale to [100, 300) (originally [5, Tbmax) in emcee)
    x[7] *= 2                     # Tbq, scale to [0, 2.)
    x[8] = 45 * u[8] + 5          # Tback, scale to [5, 50.)
    x[9] = (180) * u[9] + 240     # dV0, scale to [240, 420) (now follows doppler linewidth np.sqrt(2kTb0/mu*m_H, originally [0, 1000.) in emcee)
    x[10] += 3.5                  # vsys, scale to [3.5, 4.5)
    x[11] = 0.4 * u[11] - 0.2     # dx, scale to [-0.2, 0.2)
    x[12] = 0.4 * u[12] - 0.2     # dy, scale to [-0.2, 0.2)

    return x


### LOG(POSTERIOR)
### --------------
def lnprob(theta):

    # generate a model cube
    mcube = cube_parser(inc=theta[0], PA=theta[1], dist=dist, r0=r0,
                        mstar=theta[2], r_l=theta[3], z0=theta[4],
                        zpsi=theta[5], Tb0=theta[6], Tbq=theta[7],
                        Tbmax=Tbmax, Tbmax_b=theta[8], dV0=theta[9],
                        dVq=0.5*theta[7], FOV=FOV, Npix=Npix,
                        Vsys=theta[10], restfreq=nu_l, vel=v_LSRK_mid)

    # sample the FT of the cube onto the observed (u,v) points
    mvis = vis_sample(imagefile=mcube, mu_RA=theta[11], mu_DEC=theta[12],
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
    return lnL

# run dynesty sampler (currently set to 100000 likelihood calculations)
maxcall = 100000
with Pool() as pool:
    dsampler = dynesty.DynamicNestedSampler(lnprob, pUform, ndim=13, bound='multi', sample='rwalk', nlive=500, pool=pool, queue_size=8)
    dsampler.run_nested(maxcall=maxcall) #print_progress=False if you don't want to see every iteration in terminal
    res = dsampler.results
    with open(wdir + 'dynesty_results_%s_logL.pickle' % (maxcall), 'wb') as f:
        print('\n Storing Pickle file...')
        pickle.dump(res, f)
