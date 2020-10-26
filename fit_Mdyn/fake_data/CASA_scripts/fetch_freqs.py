import os
import numpy

# Constants
c_ = 2.99792e5

# Load the template UVFITS file into a temporary MS file
which_temp = str(np.loadtxt('template_freqs.txt', dtype=str))
os.system('rm -rf temp.ms')
importuvfits(fitsfile='template_uvfits/'+which_temp+'.uvfits', vis='temp.ms')

# Open the MS table and extract relevant information
tb.open('temp.ms')
data = np.squeeze(tb.getcol("DATA"))
weight = tb.getcol("WEIGHT")
times = tb.getcol("TIME")
tb.close()
tb.open('temp.ms/SPECTRAL_WINDOW')
nchan = tb.getcol('NUM_CHAN').tolist()[0]
freqlist = np.squeeze(tb.getcol("CHAN_FREQ"))
tb.close()

# Find unique timestamps (in MJD)
tstamps = np.unique(times)

# Get a date/time string for start of observations
datetime0 = au.mjdsecToTimerangeComponent(tstamps[0])

# Load properties of template
io = np.loadtxt('template_params/'+which_temp+'.params.txt', dtype=str)
restfreq, RA, DEC = np.float(io[2]), str(io[6]), str(io[7])
ch_spacing, oversampling = np.float(io[1]), np.int(io[5])

# Set the *FIXED* TOPO frequencies we will convert to
freq_TOPO = au.restToTopo(restfreq, c_ * (1. - freqlist[0] / restfreq), 
                          datetime0, RA, DEC) - \
            (ch_spacing*1e3 / oversampling) * np.arange(nchan)

# Calculate the LSRK frequencies that correspond to these TOPO frequencies at
# each individual timestamp
freq_LSRK = np.empty((len(tstamps), nchan))
for i in range(len(tstamps)):
    for j in range(nchan):
        dt = au.mjdsecToTimerangeComponent(tstamps[i])
        freq_LSRK[i,j] = au.topoToLSRK(freq_TOPO[j], dt, RA, DEC)

# Save these frequency conversions
np.savez('template_params/'+which_temp+'.freq_conversions', 
         freq_TOPO=freq_TOPO, freq_LSRK=freq_LSRK)

# Clean up
os.system('rm -rf temp.ms')
os.system('rm -rf template_freqs.txt')
