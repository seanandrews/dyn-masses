import os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import emcee
import corner

# emcee backend file
fname = 'simp3_std_medr_highv_rms0_noisy'

# scale burn-in
burn_in = 0
burnin = 1000

# calculate autocorrelation time as a function of step?
calc_tau = False
Ntau = 100


# load the backend; parse the samples
reader = emcee.backends.HDFBackend('posteriors/'+fname+'.h5')
#if burn_in > 0:
#    tau = reader.get_autocorr_time()
#    burnin = int(burn_in * np.max(tau))
#    thin = int(np.min(tau) / burn_in)
#else:
#    burnin, thin = 0, 0
all_samples = reader.get_chain(discard=0, flat=False)
samples = reader.get_chain(discard=burnin, flat=False)
log_prob_samples = reader.get_log_prob(discard=burnin, flat=False)
log_prior_samples = reader.get_blobs(discard=burnin, flat=False)
print(log_prob_samples.shape)
print(np.mean(log_prob_samples, axis=0))
print(np.where(np.mean(log_prob_samples, axis=0) == np.min(np.mean(log_prob_samples, axis=0)))[0][0])

# remove the bad walker
#bad = np.where(np.mean(log_prob_samples, axis=0) == np.min(np.mean(log_prob_samples, axis=0)))[0]
#all_samples = np.delete(all_samples, bad, axis=1)
#samples = np.delete(samples, bad, axis=1)
#log_prob_samples = np.delete(log_prob_samples, bad, axis=1)
#log_prior_samples = np.delete(log_prior_samples, bad, axis=1)
nsteps, nwalk, ndim = samples.shape[0], samples.shape[1], samples.shape[2]
print(nsteps, nwalk, ndim)


# set parameter labels, truths
lbls = ['i', 'PA', 'M', 'r_l', 'z0', 'zpsi', 'Tb0', 'Tbq', 'Tback', 'dV0', 'vsys', 'dx', 'dy']
theta = [40, 130, 0.7, 200, 2.3, 1.0, 205., 0.5, 20., 347.7, 4., 0., 0.]


# plot the integrated autocorrelation time convergence every Ntau steps
if calc_tau:
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
fig = plt.figure(figsize=(5, 6))
gs = gridspec.GridSpec(8, 2)

# lnlike and lnprior at top
ax = fig.add_subplot(gs[0,0])
for iw in np.arange(nwalk):
    ax.plot(np.arange(nsteps), log_prob_samples[:, iw], color='k', alpha=0.3)
    ax.set_xlim([0, nsteps])
    ax.tick_params(which='both', labelsize=6)
    ax.set_ylabel('log likelihood', fontsize=6)
    ax.set_xticklabels([])

ax = fig.add_subplot(gs[0,1])
for iw in np.arange(nwalk):
    ax.plot(np.arange(nsteps), log_prior_samples[:, iw], color='k', alpha=0.03)
    ax.set_xlim([0, nsteps])
    ax.set_ylim([np.min(log_prior_samples[:, iw]) + 0.05,
                 np.max(log_prior_samples[:, iw]) - 0.05])
    ax.tick_params(which='both', labelsize=6)
    ax.set_ylabel('log prior', fontsize=6)
    ax.set_xticklabels([])

# now cycle through parameters
for idim in np.arange(ndim):
    ax = fig.add_subplot(gs[np.floor_divide(idim, 2) + 1, (idim % 2)])
    for iw in np.arange(nwalk):
        ax.plot(np.arange(nsteps), samples[:, iw, idim], color='k', alpha=0.03)
    ax.plot([0, nsteps], [theta[idim], theta[idim]], '-C0', lw=1)
    ax.set_xlim([0, nsteps])
    ax.tick_params(which='both', labelsize=6)
    ax.set_ylabel(lbls[idim], fontsize=6)
    if (np.floor_divide(idim, 2) + 1 != 7):
        ax.set_xticklabels([])
    else:
        ax.set_xlabel('steps', fontsize=6)

fig.subplots_adjust(wspace=0.40, hspace=0.05)
fig.subplots_adjust(left=0.15, right=0.90, bottom=0.05, top=0.99)

fig.savefig('mcmc_analysis/'+fname+'.traces.png')
fig.clf()

# corner plot to view covariances
levs = 1. - np.exp(-0.5*(np.arange(3)+1)**2)
flat_chain = samples.reshape(-1, ndim)
fig = corner.corner(flat_chain, plot_datapoints=False, levels=levs, 
                    labels=lbls, truths=theta)
fig.savefig('mcmc_analysis/'+fname+'.corner.png')
fig.clf()
