import os, sys, time
import numpy as np
from astropy.io import fits
from cube_parser import cube_parser
from vis_sample import vis_sample
from vis_sample.file_handling import import_data_uvfits
from scipy.ndimage import convolve1d
import emcee
from multiprocessing import Pool
os.environ["OMP_NUM_THREADS"] = "1"


# parse and package the DATA
data_set = 'simp3_std_medv_medr_10xHIGHV_hann_noiseless.shift'
data_file = 'fake_data/sim_uvfits/'+data_set+'.uvfits'
dvis = import_data_uvfits(data_file)

# perform windowing?
window_model = True
npad = 3

# extract the proper velocities from the data file
dat = fits.open(data_file)
hdr = dat[0].header
freq0 = hdr['CRVAL4']
indx0 = hdr['CRPIX4']
nchan = hdr['NAXIS4']
dfreq = hdr['CDELT4']
freqs = freq0 + (np.arange(nchan) - indx0 + 1) * dfreq
dvis.freqs = freqs

# extract only a subset of the velocities to fit
vidx_lo, vidx_hi = 38, 85
# if we're windowing, pad the desired channels by the window span on each side
if window_model:
    dvis.VV = dvis.VV[vidx_lo-npad:vidx_hi+npad, :]
    dvis.wgts = dvis.wgts[:, vidx_lo-npad:vidx_hi+npad]
    dvis.freqs = dvis.freqs[vidx_lo-npad:vidx_hi+npad]
    dvis.rfreq = np.mean(dvis.freqs)
else:
    dvis.VV = dvis.VV[vidx_lo:vidx_hi, :]
    dvis.wgts = dvis.wgts[:, vidx_lo:vidx_hi]
    dvis.freqs = dvis.freqs[vidx_lo:vidx_hi]
    dvis.rfreq = np.mean(dvis.freqs)

# fixed parameters
FOV = 8.0
dist = 150.
Npix = 256
Tbmax = 500.
r0 = 10.
mu_l = 28.
restfreq = 230.538e9
fixed = FOV, dist, Npix, Tbmax, r0, mu_l, restfreq, window_model

# calculate velocities (in m/s)
CC = 2.99792e10
vel = CC * (1. - dvis.freqs / restfreq) / 100.


# initialize walkers
p_lo = np.array([30., 120., 0.50, 100., 0., 0.5, 155., 0.2,  5., 0.0, -0.1, -0.1, -0.1])
p_hi = np.array([50., 140., 0.90, 300., 5., 1.5, 255., 0.8, 30., 0.1,  0.1,  0.1,  0.1])
ndim, nwalk = len(p_lo), 5 * len(p_lo)
p0 = [np.random.uniform(p_lo, p_hi, ndim) for i in range(nwalk)]

# compute 1 model to set up GCF, corr caches
theta = p0[0]
foo = cube_parser(inc=theta[0], PA=theta[1], dist=dist, mstar=theta[2], r0=r0,
                  r_l=theta[3], z0=theta[4], zpsi=theta[5],
                  Tb0=theta[6], Tbq=theta[7], Tbmax=Tbmax, Tbmax_b=theta[8],
                  xi_nt=theta[9], FOV=FOV, Npix=Npix, mu_l=mu_l,
                  Vsys=theta[10], restfreq=restfreq, vel=vel)

tvis, gcf, corr = vis_sample(imagefile=foo, uu=dvis.uu, vv=dvis.vv, 
                             return_gcf=True, return_corr_cache=True, 
                             mod_interp=False)


# package data and supplementary information
global dpassit
dpassit = dvis, gcf, corr, fixed

# posterior sample function
def lnprob_globdata(theta):

    # parse input arguments
    dvis, gcf, corr, fixed = dpassit
    FOV, dist, Npix, Tbmax, r0, mu_l, restfreq, window_model = fixed

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
    if ((theta[3] > r0) and (theta[3] < (dist * FOV / 2))):
        ptheta[3] = 0.
    else: return -np.inf

    # z0: p(z0) = uniform(0, 0.5)
    if ((theta[4] >= 0.0) and (theta[4] <= 50.)):
        ptheta[4] = 0.
    else: return -np.inf

    # zpsi: p(zsi) = uniform(0, 1.5)
    if ((theta[5] >= 0.0) and (theta[5] <= 1.5)):
        ptheta[5] = 0.
    else: return -np.inf

    # Tb0: p(Tb0) = uniform(0, Tbmax)		
    if ((theta[6] > 5.) and (theta[6] < Tbmax)):
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
    if ((theta[9] >= 0.0) and (theta[9] <= 0.2)):
        ptheta[9] = 0.
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
    
    # constants
    CC = 2.99792e10

    # convert to velocities
    vel = CC * (1. - dvis.freqs / restfreq) / 100.

    # generate a model cube
    model = cube_parser(inc=theta[0], PA=theta[1], dist=dist, mstar=theta[2], 
                        r0=r0, r_l=theta[3], z0=theta[4], zpsi=theta[5],
                        Tb0=theta[6], Tbq=theta[7], Tbmax=Tbmax, 
                        Tbmax_b=theta[8], xi_nt=theta[9], FOV=FOV, Npix=Npix, 
                        mu_l=mu_l, Vsys=theta[10], restfreq=restfreq, vel=vel)

    # now sample the FT of the model onto the observed (u,v) points
    modl_vis = vis_sample(imagefile=model, mu_RA=theta[11], mu_DEC=theta[12], 
                          gcf_holder=gcf, corr_cache=corr, mod_interp=False)

    # window the visibilities
    if window_model:
        # define and perform the window convolution
        hann = np.array([0.0, 0.25, 0.5, 0.25, 0.0])
        modl_vis_re = convolve1d(modl_vis.real, hann, axis=1, mode='nearest')
        modl_vis_im = convolve1d(modl_vis.imag, hann, axis=1, mode='nearest')
        modl_vis = modl_vis_re + 1.0j*modl_vis_im

        # excise the padded boundary channels to avoid edge effects
        modl_vis = modl_vis[:,npad:-npad]
        data_vis = dvis.VV.T[:,npad:-npad]
        weights = dvis.wgts[:,npad:-npad]
    else:
        data_vis = dvis.VV.T
        weights = dvis.wgts

    # compute the log-likelihood
    logL = -0.5 * np.sum(weights * np.absolute(data_vis - modl_vis)**2)

    # return the posterior
    return logL + np.sum(ptheta)


# set up and HDF5 backend
if window_model:
    filename = 'posteriors/'+data_set+'_windowed.h5'
else: filename = 'posteriors/'+data_set+'.h5'
os.system('rm -rf '+filename)
backend = emcee.backends.HDFBackend(filename)
backend.reset(nwalk, ndim)

max_steps = 10000
# perform the inference
with Pool() as pool:
    # set up sampler
    sampler = emcee.EnsembleSampler(nwalk, ndim, lnprob_globdata, pool=pool,
                                    backend=backend)

    # track autocorrelation time
    index = 0
    autocorr = np.empty(max_steps)
    old_tau = np.inf

    # sample for up to max_steps trials
    for sample in sampler.sample(p0, iterations=max_steps, progress=True):
        if sampler.iteration % 100:
            continue
        
        # compute the autocorrelation time 
        tau = sampler.get_autocorr_time(tol=0)
        autocorr[index] = np.mean(tau)
        index += 1

        # check convergence
        converged = np.all(tau * 100 < sampler.iteration)
        converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
        if converged:
            break
        old_tau = tau
