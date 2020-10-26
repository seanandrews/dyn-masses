import pickle
import dynesty
from dynesty import plotting as dyplot
import matplotlib.pyplot as plt

wdir = '/Users/justinvega/Documents/GitHub/dyn-masses/fit_Mdyn/pickles/'

picklefile = open(wdir + 'dynesty_results_90000_logL.pickle', 'rb')
dyresults = pickle.load(picklefile)

# set parameter labels, truths
theta_true = [40, 130, 0.7, 200, 2.3, 1, 205, 0.5, 20, 347.6, 4.0, 0, 0]
lbls = [r'$i$', r'$PA$', r'$M$', r'$r_l$', r'$z0$', r'$z_{\psi}$', r'$Tb_{0}$', r'$Tb_q$', r'$T_{\rm{back}}$', r'$dV_{0}$', r'$v_{\rm{sys}}$', r'$dx$', r'$dy$']

# plot cornerplot (not working yet)
# fig, axes = dyplot.cornerplot(db, truths=theta_true, labels=lbls)

# plot traces
fig, axes = dyplot.traceplot(dyresults, truths=theta_true,labels=lbls,
                             truth_color='black', show_titles=True,
                             trace_cmap='viridis', fig=plt.subplots(13, 2, figsize=(16, 64)))
fig.tight_layout()

#plt.show() # necessary to see plots
plt.savefig("./plots/traceplot_90000_logL.jpg")
#plt.close()
