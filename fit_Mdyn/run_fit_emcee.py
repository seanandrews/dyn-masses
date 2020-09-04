import os, sys, time
import numpy as np
import copy as copy
from astropy.io import fits
from cube_parser import cube_parser
from vis_sample import vis_sample
from vis_sample.file_handling import import_data_uvfits
from scipy.ndimage import convolve1d
import emcee
from multiprocessing import Pool
os.environ["OMP_NUM_THREADS"] = "1"


# USER INTERFACE
# --------------
# locate data
datadir  = 'fake_data/sim_uvfits/'
datafile = 'simp3_std_medr_medv_noiseless'

# spectral signal processing
chbin = 2               # number of channels to average over
chpad = 3               # number of channels to pad for SRF convolution
vlo, vhi = -5., 5.    # low and high LSRK velocities to process, in [km/s]



# CONSTANTS
# ---------
c_ = 2.99792e8          # speed of light in [m/s]
restfreq = 230.538e9    # 12CO J=2-1 in [Hz]



# PROCESS DATA
# ------------
# load data visibilities with native channel spacings (LSRK)
dvis_native = import_data_uvfits(datadir+datafile+'.uvfits')

# extract the native channel frequencies, convert to LSRK velocities in [m/s]
hdr = fits.open(datadir+datafile+'.uvfits')[0].header
freq0, idx0, nchan = hdr['CRVAL4'], hdr['CRPIX4'], hdr['NAXIS4']
dvis_native.freqs = freq0 + (np.arange(nchan) - idx0 + 1) * hdr['CDELT4']
vlsrk_native = c_ * (1. - dvis_native.freqs / restfreq)

# identify the subset of channel indices of interest
vlo_idx = np.max(np.where(vlsrk_native < vlo*1e3))
vhi_idx = np.min(np.where(vlsrk_native > vhi*1e3)) + 1
nnative = vhi_idx - vlo_idx

# extract the subset of native channels of interest, padded for windowing
dvis_native.VV = dvis_native.VV[vlo_idx-chpad:vhi_idx+chpad,:]
dvis_native.wgts = dvis_native.wgts[:,vlo_idx-chpad:vhi_idx+chpad].T
dvis_native.freqs = dvis_native.freqs[vlo_idx-chpad:vhi_idx+chpad]
vlsrk_native = c_ * (1. - dvis_native.freqs / restfreq)
dvis_native.rfreq = np.mean(dvis_native.freqs)

# make a copy of the input (native) data to bin
dvis = copy.deepcopy(dvis_native)

# clip the unpadded data, so divisible by factor chbin
dvis.VV = dvis.VV[chpad:chpad+nnative-(nnative % chbin),:]
dvis.wgts = dvis.wgts[chpad:chpad+nnative-(nnative % chbin),:]
dvis.freqs = dvis.freqs[chpad:chpad+nnative-(nnative % chbin)]

# weighted decimated average
dvis.VV = np.average(dvis.VV.reshape((-1, chbin, dvis.VV.shape[1])),
                     weights=dvis.wgts.reshape((-1, chbin, dvis.wgts.shape[1])),
                     axis=1)
dvis.wgts = np.sum(dvis.wgts.reshape((-1, chbin, dvis.wgts.shape[1])), axis=1)
dvis.freqs = np.average(dvis.freqs.reshape(-1, chbin), axis=1)
dvis.rfreq = np.mean(dvis.freqs)
vlsrk = c_ * (1. - dvis.freqs / restfreq)
nch = len(vlsrk)


# PRECALCULATE PARTIAL COVARIANCE MATRIX
# --------------------------------------
# for case of ~similar weights and 2x binning
Mx2 = np.eye(nch) + 0.3 * np.eye(nch, k=-1) + 0.3 * np.eye(nch, k=1)
Mx2_inv = np.linalg.inv(Mx2)


# INITIALIZE
# ----------
# fixed model parameters
FOV, dist, Npix, Tbmax, r0, mu_l = 8.0, 150., 256, 500., 10., 28.

# initialize walkers
p_lo = np.array([30, 120, 0.5, 100, 0, 0.5, 155, 0.2,  5, 0, -0.1, -0.1, -0.1])
p_hi = np.array([50, 140, 0.9, 300, 5, 1.5, 255, 0.8, 30, 0.1, 0.1, 0.1, 0.1])
ndim, nwalk = len(p_lo), 5 * len(p_lo)
p0 = [np.random.uniform(p_lo, p_hi, ndim) for i in range(nwalk)]

# compute 1 model to set up GCF, corr caches
theta = p0[0]
foo = cube_parser(inc=theta[0], PA=theta[1], dist=dist, mstar=theta[2], r0=r0,
                  r_l=theta[3], z0=theta[4], zpsi=theta[5],
                  Tb0=theta[6], Tbq=theta[7], Tbmax=Tbmax, Tbmax_b=theta[8],
                  xi_nt=theta[9], FOV=FOV, Npix=Npix, mu_l=mu_l,
                  Vsys=theta[10], restfreq=restfreq, vel=vlsrk_native)

tvis, gcf, corr = vis_sample(imagefile=foo, uu=dvis_native.uu, 
                             vv=dvis_native.vv, return_gcf=True, 
                             return_corr_cache=True, mod_interp=False)

# object for passing information as global variable to posterior function
class passer:
    def __init__(self, dist, FOV, Tbmax, vels, Npix, mu_l, r0, nu0, gcf, corr, 
                 native_wgts, chpad, chbin, data_vis, data_wgts, M_inv):
        self.dist = dist
        self.FOV = FOV
        self.Tbmax = Tbmax
        self.vels = vels
        self.Npix = Npix
        self.mu_l = mu_l
        self.r0 = r0
        self.nu0 = nu0
        self.gcf = gcf
        self.corr = corr
        self.native_wgts = native_wgts
        self.chpad = chpad
        self.chbin = chbin
        self.data_vis = data_vis
        self.data_wgts = data_wgts
        self.M_inv = M_inv

global dd
dd = passer(dist, FOV, Tbmax, vlsrk_native, Npix, mu_l, r0, restfreq, gcf, 
            corr, dvis_native.wgts, chpad, chbin, dvis.VV, (16./5.)*dvis.wgts, 
            Mx2_inv)
     

# posterior sample function
def lnprob_globdata(theta):

    ### - PRIORS
    ptheta = np.empty_like(theta)

    # inc: p(i) = sin(i)
    if ((theta[0] > 0.) and (theta[0] < 90.)):
        ptheta[0] = 0
    else: return -np.inf

    # PA: p(PA) = uniform(0, 360)
    if ((theta[1] > 0.) and (theta[1] < 360.)): 
        ptheta[1] = 0.
    else: return -np.inf

    # Mstar: p(Mstar) = uniform(0, 5)
    if ((theta[2] > 0.) and (theta[2] < 5.)):
        ptheta[2] = 0.
    else: return -np.inf

    # r_l: p(r_l) = uniform(0, dist * FOV / 2)          
    if ((theta[3] > dd.r0) and (theta[3] < (dd.dist * dd.FOV / 2))):
        ptheta[3] = 0.
    else: return -np.inf

    # z0: p(z0) = uniform(0, 0.5)
    if ((theta[4] >= 0.0) and (theta[4] <= 10.)):
        ptheta[4] = 0.
    else: return -np.inf

    # zpsi: p(zsi) = uniform(0, 1.5)
    if ((theta[5] >= 0.0) and (theta[5] <= 1.5)):
        ptheta[5] = 0.
    else: return -np.inf

    # Tb0: p(Tb0) = uniform(0, Tbmax)		
    if ((theta[6] > 5.) and (theta[6] < dd.Tbmax)):
        ptheta[6] = 0.
    else: return -np.inf

    # Tbq: p(Tbq) = uniform(0, 2)
    if ((theta[7] >= 0) and (theta[7] < 2)):
        ptheta[7] = 0.
    else: return -np.inf

    # Tbmax_b: p(Tbmax_b) = uniform(5, 50)
    if ((theta[8] >= 5.) and (theta[8] <= 50.)):
        ptheta[8] = 0.
    else: return -np.inf

    # xi_nt: p(xi_nt) = uniform(0, 0.2)
    if ((theta[9] >= 0.0) and (theta[9] < 1.0)):
        ptheta[9] = 0.	#-0.5 * (theta[9] - 0.05)**2 / 0.001**2
    else: return -np.inf

    # V_sys: p(V_sys) = normal(0.0, 0.1)	# adjusted for each case
    if ((theta[10] > -0.2) & (theta[10] < 0.2)):
        ptheta[10] = 0.
    else: return -np.inf

    # dx: p(dx) = normal(0.0, 0.1)		# adjusted for each case
    if ((theta[11] > -0.2) & (theta[11] < 0.2)):
        ptheta[11] = 0.	#-0.5 * (theta[9] - 0.0)**2 / 0.08**2
    else: return -np.inf

    # dy: p(dy) = normal(0.0, 0.1)		# adjusted for each case
    if ((theta[12] > -0.2) & (theta[12] < 0.2)):
        ptheta[12] = 0.	#-0.5 * (theta[10] - 0.0)**2 / 0.08**2
    else: return -np.inf
    
    # generate a model cube
    model = cube_parser(inc=theta[0], PA=theta[1], dist=dd.dist, r0=dd.r0,
                        mstar=theta[2], r_l=theta[3], z0=theta[4], 
                        zpsi=theta[5], Tb0=theta[6], Tbq=theta[7], 
                        Tbmax=dd.Tbmax, Tbmax_b=theta[8], xi_nt=theta[9], 
                        FOV=dd.FOV, Npix=dd.Npix, mu_l=dd.mu_l, Vsys=theta[10], 
                        restfreq=dd.nu0, vel=dd.vels)

    # now sample the FT of the model onto the observed (u,v) points
    modl_vis = vis_sample(imagefile=model, mu_RA=theta[11], mu_DEC=theta[12], 
                          gcf_holder=dd.gcf, corr_cache=dd.corr, 
                          mod_interp=False)

    # window the visibilities
    hann = np.array([0.0, 0.25, 0.5, 0.25, 0.0])
    modl_vis_re = convolve1d(modl_vis.real, hann, axis=1, mode='nearest')
    modl_vis_im = convolve1d(modl_vis.imag, hann, axis=1, mode='nearest')
    modl_vis = modl_vis_re + 1.0j*modl_vis_im

    # excise the padded boundary channels to avoid edge effects
    modl_vis = modl_vis[:,dd.chpad:-dd.chpad].T
    modl_wgt = dd.native_wgts[dd.chpad:-dd.chpad,:]

    # clip for decimating
    modl_vis = modl_vis[:modl_vis.shape[0]-(modl_vis.shape[0] % dd.chbin),:]
    modl_wgt = modl_wgt[:modl_vis.shape[0]-(modl_vis.shape[0] % dd.chbin),:]

    # weighted decimated average
    modl_vis = np.average(modl_vis.reshape((-1, dd.chbin, modl_vis.shape[1])),
                          weights=modl_wgt.reshape((-1, dd.chbin, 
                                                   modl_wgt.shape[1])), axis=1)

    # compute the log-likelihood
    resid = np.absolute(dd.data_vis - modl_vis)
    logL = -0.5 * np.sum(resid * np.dot(dd.M_inv, dd.data_wgts * resid))

    # return the posterior
    return logL + np.sum(ptheta)



# set up and HDF5 backend
filename = 'posteriors/'+datafile+'.h5'
os.system('rm -rf '+filename)
backend = emcee.backends.HDFBackend(filename)
backend.reset(nwalk, ndim)

# run the sampler
max_steps = 10000
with Pool() as pool:
    sampler = emcee.EnsembleSampler(nwalk, ndim, lnprob_globdata, pool=pool,
                                    backend=backend)
    sampler.run_mcmc(p0, max_steps, progress=True)
