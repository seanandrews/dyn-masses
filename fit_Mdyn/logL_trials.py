import os, sys, time
import numpy as np
import copy as copy
from astropy.io import fits
from cube_parser import cube_parser
from vis_sample import vis_sample
from vis_sample.file_handling import import_data_uvfits
from scipy.ndimage import convolve1d
from scipy.interpolate import interp1d
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
        self.gcf = 0
        self.corr = 0
        self.native_wgts = native_wgts
        self.chpad = chpad
        self.chbin = chbin
        self.data_vis = data_vis
        self.data_wgts = data_wgts
        self.M_inv = M_inv

global dd
dd = passer(dist, FOV, Tbmax, vlsrk_native, Npix, mu_l, r0, restfreq, 0,
            0, dvis_native.wgts, chpad, chbin, dvis.VV, (16./5.)*dvis.wgts, 
            Mx2_inv)
     

# posterior sample function
def lnprob_globdata(theta):

    # generate a model cube
    model = cube_parser(inc=theta[0], PA=theta[1], dist=dd.dist, r0=dd.r0,
                        mstar=theta[2], r_l=theta[3], z0=theta[4], 
                        zpsi=theta[5], Tb0=theta[6], Tbq=theta[7], 
                        Tbmax=dd.Tbmax, Tbmax_b=theta[8], xi_nt=theta[9], 
                        FOV=dd.FOV, Npix=dd.Npix, mu_l=dd.mu_l, Vsys=theta[10], 
                        restfreq=dd.nu0, vel=dd.vels)

    # now sample the FT of the model onto the observed (u,v) points
    modl_vis = vis_sample(imagefile=model, uu=dvis_native.uu, 
                          vv=dvis_native.vv, mu_RA=theta[11], mu_DEC=theta[12], 
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
    return logL 


def lnprob_super_simple(theta):

    # generate a model cube
    model = cube_parser(inc=theta[0], PA=theta[1], dist=dd.dist, r0=dd.r0,
                        mstar=theta[2], r_l=theta[3], z0=theta[4],
                        zpsi=theta[5], Tb0=theta[6], Tbq=theta[7],
                        Tbmax=dd.Tbmax, Tbmax_b=theta[8], xi_nt=theta[9],
                        FOV=dd.FOV, Npix=dd.Npix, mu_l=dd.mu_l, Vsys=theta[10],
                        restfreq=dd.nu0, vel=vlsrk)

    # now sample the FT of the model onto the observed (u,v) points
    modl_vis = vis_sample(imagefile=model, uu=dvis_native.uu,
                          vv=dvis_native.vv, mu_RA=theta[11], mu_DEC=theta[12],
                          mod_interp=False)

    # compute the log-likelihood
    resid = np.absolute(dd.data_vis - modl_vis.T)
    logL = -0.5 * np.sum(resid * np.dot(dd.M_inv, dd.data_wgts * resid))

    # return the posterior
    return logL


def lnprob_dec_only(theta):
    # generate a model cube
    model = cube_parser(inc=theta[0], PA=theta[1], dist=dd.dist, r0=dd.r0,
                        mstar=theta[2], r_l=theta[3], z0=theta[4],
                        zpsi=theta[5], Tb0=theta[6], Tbq=theta[7],
                        Tbmax=dd.Tbmax, Tbmax_b=theta[8], xi_nt=theta[9],
                        FOV=dd.FOV, Npix=dd.Npix, mu_l=dd.mu_l, Vsys=theta[10],
                        restfreq=dd.nu0, vel=dd.vels)

    # now sample the FT of the model onto the observed (u,v) points
    modl_vis = vis_sample(imagefile=model, uu=dvis_native.uu,
                          vv=dvis_native.vv, mu_RA=theta[11], mu_DEC=theta[12],
                          mod_interp=False)

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
    return logL


df = np.load('fake_data/template_params/std_medr_medv10x.freq_conversions.npz')
freq_TOPO = df['freq_TOPO']
freq_LSRK = df['freq_LSRK']
freq_TOPO = freq_TOPO[::10].copy()
freq_LSRK = freq_LSRK[:,::10].copy()
v_LSRK = c_ * (1. - freq_LSRK / restfreq)
nvis = dvis_native.VV.shape[1]
nperstamp = np.int(nvis / 30)

def lnprob_almostcorrect(theta):

    # generate model cubes
    for i in range(v_LSRK.shape[0]):
        model = cube_parser(inc=theta[0], PA=theta[1], dist=dd.dist, r0=dd.r0,
                            mstar=theta[2], r_l=theta[3], z0=theta[4],
                            zpsi=theta[5], Tb0=theta[6], Tbq=theta[7],
                            Tbmax=dd.Tbmax, Tbmax_b=theta[8], xi_nt=theta[9],
                            FOV=dd.FOV, Npix=dd.Npix, mu_l=dd.mu_l, 
                            Vsys=theta[10], restfreq=dd.nu0, 
                            vel=v_LSRK[i,:])

        # now sample the FT of the model onto the observed (u,v) points
        mvis = vis_sample(imagefile=model, uu=dvis_native.uu,
                          vv=dvis_native.vv, mu_RA=theta[11], 
                          mu_DEC=theta[12], mod_interp=False)

        # populate the appropriate parts of the composite model
        if i == 0: modl_vis = mvis.copy()
        ix_lo, ix_hi = i * nperstamp, (i + 1) * nperstamp
        modl_vis.real[ix_lo:ix_hi,:] = mvis.real[ix_lo:ix_hi,:]
        modl_vis.imag[ix_lo:ix_hi,:] = mvis.imag[ix_lo:ix_hi,:]

    # window the visibilities
    hann = np.array([0.0, 0.25, 0.5, 0.25, 0.0])
    modl_vis_re = convolve1d(modl_vis.real, hann, axis=1, mode='nearest')
    modl_vis_im = convolve1d(modl_vis.imag, hann, axis=1, mode='nearest')
    modl_vis = modl_vis_re + 1.0j*modl_vis_im

    # LSRK frequency grid for easier interpolation
    freqgrid_LSRK = np.zeros((modl_vis.shape[0], modl_vis.shape[1]))
    for i in range(freq_LSRK.shape[0]):
        ix_lo, ix_hi = i * nperstamp, (i + 1) * nperstamp
        freqgrid_LSRK[ix_lo:ix_hi, :] = freq_LSRK[i,:]

    # model interpolation
    modl_visi = np.zeros((modl_vis.shape[0], len(dvis_native.freqs)),
                         dtype=complex)
    for i in range(modl_vis.shape[0]):
        re_int = interp1d(freqgrid_LSRK[i,:], modl_vis.real[i,:],
                          fill_value='extrapolate')
        im_int = interp1d(freqgrid_LSRK[i,:], modl_vis.imag[i,:],
                          fill_value='extrapolate')
        modl_visi[i,:] = re_int(dvis_native.freqs) + \
                         1j*im_int(dvis_native.freqs)

    # excise the padded boundary channels to avoid edge effects
    modl_viso = modl_visi[:,dd.chpad:-dd.chpad].T
    modl_wgto = dd.native_wgts[dd.chpad:-dd.chpad,:]

    # clip for decimating
    modl_visd = modl_viso[:modl_viso.shape[0]-(modl_viso.shape[0] % dd.chbin),:]
    modl_wgtd = modl_wgto[:modl_viso.shape[0]-(modl_viso.shape[0] % dd.chbin),:]

    # weighted decimated average
    modl_visf = np.average(modl_visd.reshape((-1, dd.chbin, modl_visd.shape[1])),
                           weights=modl_wgtd.reshape((-1, dd.chbin,
                                                   modl_wgtd.shape[1])), axis=1)

    # compute the log-likelihood
    resid = np.absolute(dd.data_vis - modl_visf)
    logL = -0.5 * np.sum(resid * np.dot(dd.M_inv, dd.data_wgts * resid))

    # return the posterior
    return logL


def lnprob_testcheat(theta):

    # generate model cubes
    model = cube_parser(inc=theta[0], PA=theta[1], dist=dd.dist, r0=dd.r0,
                        mstar=theta[2], r_l=theta[3], z0=theta[4],
                        zpsi=theta[5], Tb0=theta[6], Tbq=theta[7],
                        Tbmax=dd.Tbmax, Tbmax_b=theta[8], xi_nt=theta[9],
                        FOV=dd.FOV, Npix=dd.Npix, mu_l=dd.mu_l,
                        Vsys=theta[10], restfreq=dd.nu0,
                        vel=v_LSRK[15,:])

    # now sample the FT of the model onto the observed (u,v) points
    mvis = vis_sample(imagefile=model, uu=dvis_native.uu,
                      vv=dvis_native.vv, mu_RA=theta[11],
                      mu_DEC=theta[12], mod_interp=False)

    # window the visibilities
    hann = np.array([0.0, 0.25, 0.5, 0.25, 0.0])
    modl_vis_re = convolve1d(mvis.real, hann, axis=1, mode='nearest')
    modl_vis_im = convolve1d(mvis.imag, hann, axis=1, mode='nearest')
    modl_vis = modl_vis_re + 1.0j*modl_vis_im

    # model interpolation
    fint = interp1d(freq_LSRK[15,:], modl_vis, axis=1, 
                    fill_value='extrapolate')
    modl_visi = fint(dvis_native.freqs)

    # excise the padded boundary channels to avoid edge effects
    modl_viso = modl_visi[:,dd.chpad:-dd.chpad].T
    modl_wgto = dd.native_wgts[dd.chpad:-dd.chpad,:]

    # clip for decimating
    modl_visd = modl_viso[:modl_viso.shape[0]-(modl_viso.shape[0] % dd.chbin),:]
    modl_wgtd = modl_wgto[:modl_viso.shape[0]-(modl_viso.shape[0] % dd.chbin),:]

    # weighted decimated average
    modl_visf = np.average(modl_visd.reshape((-1, dd.chbin, modl_visd.shape[1])),
                           weights=modl_wgtd.reshape((-1, dd.chbin,
                                                   modl_wgtd.shape[1])), axis=1)

    # compute the log-likelihood
    resid = np.absolute(dd.data_vis - modl_visf)
    logL = -0.5 * np.sum(resid * np.dot(dd.M_inv, dd.data_wgts * resid))

    # return the posterior
    return logL






# true parameters
theta_true = [40, 130, 0.7, 200, 2.3, 1, 205, 0.5, 20, 0.05, 0, 0, 0]

# current method
t0 = time.time()
chi2 = -2. * lnprob_globdata(theta_true)
print('current:      chi2 =  %5.2f in  %3.2f s' % (chi2, time.time()-t0))

# super simple method (straight up calculation on binned velocities; no SSP)
t0 = time.time()
chi2 = -2. * lnprob_super_simple(theta_true)
print('super simple: chi2 = %5.2f in  %3.2f s' % (chi2, time.time()-t0))

# decimate-only method (calculation on native velocities; decimated average)
t0 = time.time()
chi2 = -2. * lnprob_dec_only(theta_true)
print('dec-avg only: chi2 = %5.2f in  %3.2f s' % (chi2, time.time()-t0))

# midpt-interp method 
t0 = time.time()
chi2 = -2. * lnprob_testcheat(theta_true)
print('midpt-interp: chi2 = %5.2f in  %3.2f s' % (chi2, time.time()-t0))

# almost-correct method (correct aside from over-sampling to do proper SRF)
#t0 = time.time()
#chi2 = -2. * lnprob_almostcorrect(theta_true)
#print('almost right: chi2 =   %5.2f in %3.2f s' % (chi2, time.time()-t0))



