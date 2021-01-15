#% matplotlib inline
#% config InlineBackend.figure_format = 'retina' # For mac users with Retina display
from matplotlib import pyplot as plt
import numpy as np
import pygtc
import os, sys
import matplotlib.pyplot as plt
#import matplotlib.gridspec as gridspec
import emcee
#import corner

wdir = '/Users/justinvega/Documents/GitHub/dyn-masses/fit_Mdyn/post/posteriors/postpack/'

# emcee backend files
postfiles = np.loadtxt(wdir+'posteriors.txt', dtype=str)

#fname = 'simp3_std_medr_highv_1024pix_noiseless'

# scale burn-in
burnin = 500

# calculate autocorrelation time as a function of step?
calc_tau = False
Ntau = 100

# set parameter labels, truths
lbls = [r'$i$', r'$PA$', r'$M$', r'$r_l$', r'$z0$', r'$z_{\psi}$', r'$Tb_{0}$', r'$Tb_q$', r'$T_{\rm{back}}$', r'$dV_{0}$', r'$v_{\rm{sys}}$', r'$dx$', r'$dy$']
theta = [40, 130, 0.7, 200, 2.3, 1.0, 205., 0.5, 20., 347.7, 4., 0., 0.]

# load the backend; parse the samples
def make_chain(filename):
    reader = emcee.backends.HDFBackend(wdir + filename)
    all_samples = reader.get_chain(discard=0, flat=False)
    samples = reader.get_chain(discard=burnin, flat=False)
    log_prob_samples = reader.get_log_prob(discard=burnin, flat=False)
    log_prior_samples = reader.get_blobs(discard=burnin, flat=False)
    maxlnprob = np.max(reader.get_log_prob(discard=0, flat=False))
    minlnprob = np.min(reader.get_log_prob(discard=0, flat=False))
    #print(minlnprob, maxlnprob)
    nsteps, nwalk, ndim = samples.shape[0], samples.shape[1], samples.shape[2]
    # corner plot to view covariances
    levs = 1. - np.exp(-0.5*(np.arange(3)+1)**2)
    flat_chain = samples.reshape(-1, ndim)
    return(flat_chain)

    # for trace plot
    for i in range(ndim):
        fig = plt.figure(figsize=(5, 5))
        plt.plot(np.arange(nsteps), samples[:,:,i], alpha=0.3)
        plt.ylabel(lbls[i], fontsize=12)
        plt.title('Trace %s' % lbls[i])
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.show()

noiseless = make_chain(postfiles[0])
regrid2x_noSSPnative = make_chain(postfiles[1])
SRFonly = make_chain(postfiles[2])
noSSPnative = make_chain(postfiles[3])
nobincov = make_chain(postfiles[4])
regridonly = make_chain(postfiles[5])

chainLabels= ['noiseless', 'regrid2x_noSSPnative', 'SRFonly', 'noSSPnative', 'nobincov', 'regridonly']

#plt.figure(figsize=())

#fig.tight_layout()

# GTC = pygtc.plotGTC(chains=[noiseless,noSSPnative, SRFonly], truths=theta, paramNames=lbls, chainLabels=(chainLabels[0], chainLabels[3], chainLabels[2]),
#                     truthLabels='Truth', legendMarker = 'All', customTickFont= {'family':'Arial', 'size':4}, figureSize=7.5)
# GTC.savefig(wdir + 'comparisons/cornerplot_%s_%s_%s.png' % (chainLabels[0], chainLabels[3], chainLabels[2]), dpi=300)


# # plot the traces
# fig = plt.figure(figsize=(5, 5))
# for idim in np.arange(ndim):
#     for iw in np.arange(nwalk):
#
#
#
#
#
#


# noiseless, regrid2x_noSSPnative, nobincov [0, 1, 4]
# noiseless, noSSPnative, SRFonly [0, 3, 2]


# cornerplots
a = np.stack((noiseless[:,6], noiseless[:,7], noiseless[:,8], noiseless[:,9]), axis=-1)
b = np.stack((noSSPnative[:,6], noSSPnative[:,7], noSSPnative[:,8], noSSPnative[:,9]), axis=-1)
c = np.stack((SRFonly[:,6], SRFonly[:,7], SRFonly[:,8], SRFonly[:,9]), axis=-1)


GTC = pygtc.plotGTC(chains=[a, b, c], truths=[205, 0.5, 20, 347.7], paramNames=[r'$Tb_{0}$', r'$Tb_q$', r'$T_{\rm{back}}$', r'$dV_{0}$'], chainLabels=(chainLabels[0], chainLabels[3], chainLabels[2]),
                    truthLabels='Truth', legendMarker = 'All', customTickFont= {'family':'Arial', 'size':9}, figureSize=7.5)
GTC.savefig(wdir + 'comparisons/cornerplot_%s_%s_%s_tb__tbq_tback_dV0.png' % (chainLabels[0], chainLabels[3], chainLabels[2]), dpi=300)

#plt.show()
