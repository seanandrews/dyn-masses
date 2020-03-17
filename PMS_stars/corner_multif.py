import numpy as np
import os
import sys
import corner
from post_summary import post_summary

name = 'IP_Tau'
Mdyn = 0.94
eMdyn = 0.05
prange = [(-0.5, 0.1), (5.5, 7.7)]
nbins = 30

fspot = ['f000', 'f017', 'f034', 'f051']

# defaults
levs = 1.-np.exp(-0.5*(np.arange(2)+1)**2)
levs0 = 1.-np.exp(-0.5*(np.arange(1)+1)**2)
cols = ['b', 'g', 'orange', 'r']
lbls = ['log Mstar / Msun', 'log age / yr']

all_logM = []
all_logAGE = []
for i in range(len(fspot)):

    # load posteriors
    dat = np.load('posteriors/'+name+'_'+fspot[i]+'.age-mass.posterior.npz')
    logM, logAGE = dat['logM'], dat['logAGE']
    posts = np.column_stack([logM, logAGE])
    all_logM = np.append(all_logM, logM)
    all_logAGE = np.append(all_logAGE, logAGE)

    # corner plot
    if i == 0:
        figA = corner.corner(posts, plot_datapoints=False, bins=nbins, 
                             levels=levs0, range=prange, no_fill_contours=True, 
                             plot_density=False, color=cols[i], labels=lbls,
                             truths=(np.log10(Mdyn), 7.8))
    else:
        corner.corner(posts, plot_datapoints=False, bins=nbins, levels=levs0,
                      range=prange, no_fill_contours=True, plot_density=False, 
                      color=cols[i], fig=figA, truths=(np.log10(Mdyn), 7.8))

    # posterior summary printout
    oname = name.replace('_', '')
    #print(post_summary(logM, mu='median', prec=0.01))
    #print(post_summary(logAGE, mu='median', prec=0.01))

# save corner plots
figA.savefig('corner_'+oname+'_fspots.png')
figA.clf()


# now make the composite distribution
all_posts = np.column_stack([all_logM, all_logAGE])

# weights based on dynamical mass
weights = np.exp(-0.5*((10**all_logM - Mdyn) / eMdyn)**2)

figB = corner.corner(all_posts, plot_datapoints=False, bins=40, levels=levs,
                     range=prange, no_fill_contours=True, plot_density=False, 
                     labels=lbls)

corner.corner(all_posts, weights=weights, plot_datapoints=False, bins=40, 
              levels=levs, range=prange, no_fill_contours=True,
              plot_density=False, labels=lbls, color='r', fig=figB)

quants = np.array([0.16, 0.84])
qs = corner.quantile(all_logAGE, quants, weights=weights)
print(qs)

figB.savefig('corner_'+oname+'_joint.png')
figB.clf()
