import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
import os
import sys

# emcee backend file
fname = 'simp3_std_medr_medv_test_noiseless'


# load the backend
reader = emcee.backends.HDFBackend('posteriors/'+fname+'.h5')

# set burn-in
burnin = 2000

# set parameter labels, truths
lbls = ['i', 'PA', 'M', 'r_l', 'z0', 'zpsi', 'Tb0', 'Tbq', 'Tback', 'xi_turb', 'vsys', 'dx', 'dy']
theta = [40., 130., 0.7, 200., 2.3, 1.0, 205., 0.5, 20., 0.05, 0., 0., 0.]

# parse the samples
all_samples = reader.get_chain(discard=0, flat=False)
samples = reader.get_chain(discard=burnin, flat=False)
remove_it = np.where(np.mean(samples[:,:,0], axis=0) < 35.)[0]
all_samples = np.delete(all_samples, remove_it, axis=1)
samples = np.delete(samples, remove_it, axis=1)
nsteps, nwalk, ndim = samples.shape[0], samples.shape[1],  samples.shape[2]


# plot the integrated autocorrelation time convergence every Ntau steps
Ntau = 100
Nmax = all_samples.shape[0]
if (Nmax > Ntau):
    tau_ix = np.empty(np.int(Nmax / Ntau))
    ix = np.empty(np.int(Nmax / Ntau))
    for i in range(len(tau_ix)):
        nn = (i + 1) * Ntau
        ix[i] = nn
        tau = emcee.autocorr.integrated_time(all_samples[:nn,:,:], tol=0)
        tau_ix[i] = np.mean(tau)

    fig = plt.figure()
    plt.plot(ix, tau_ix, '-o')
    plt.xlabel('steps')
    plt.ylabel('autocorr time (steps)')
    plt.xlim([0, Nmax])
    plt.ylim([0, tau_ix.max() + 0.1 * (tau_ix.max() - tau_ix.min())])
    fig.savefig('mcmc_analysis/'+fname+'.autocorr.png')
    fig.clf()

# plot the traces
fig = plt.figure(figsize=(8, 8))
for idim in np.arange(ndim):
    plt.subplot(4, 4, idim+1)
    for iw in np.arange(nwalk):
        plt.plot(np.arange(nsteps), samples[:, iw, idim], alpha=0.3)
        plt.ylabel(lbls[idim], fontsize=6)
        plt.xticks(fontsize=6)
        plt.yticks(fontsize=6)
fig.subplots_adjust(wspace=0.45, hspace=0.3)
fig.subplots_adjust(left=0.10, right=0.90, bottom=0.04, top=0.99)
fig.savefig('mcmc_analysis/'+fname+'.traces.png')
fig.clf()

# corner plot to view covariances
levs = 1. - np.exp(-0.5*(np.arange(3)+1)**2)
flat_chain = samples.reshape(-1, ndim)
fig = corner.corner(flat_chain, plot_datapoints=False, levels=levs, 
                    labels=lbls, truths=theta)
fig.savefig('mcmc_analysis/'+fname+'.corner.png')
fig.clf()
