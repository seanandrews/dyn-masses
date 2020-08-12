import numpy as np
import os
import sys
from scipy import stats

def post_summary(p, prec=0.1, mu='peak', CIlevs=[84.135, 15.865]):

    # deal with NaNs
    finite_p = p[np.isfinite(p)]
    if (len(finite_p) < len(p)):
        print('%i NaN points were removed (%.1f percent of total).' % \
              (len(p)-len(finite_p), 100.*(len(p)-len(finite_p)) / len(p)))
    p = finite_p

    # calculate percentiles as designated
    CI_p = np.percentile(p, CIlevs)

    # find peak of posterior
    if (mu == 'peak'):
        kde_p = stats.gaussian_kde(p)
        ndisc = np.int(np.round((CI_p[0] - CI_p[1]) / prec))
        x_p = np.linspace(CI_p[1], CI_p[0], ndisc)
        pk_p = x_p[np.argmax(kde_p.evaluate(x_p))]
    else:
        pk_p = np.percentile(p, 50.)

    # return the peak and upper, lower 1-sigma
    return (pk_p, CI_p[0]-pk_p, pk_p-CI_p[1])

