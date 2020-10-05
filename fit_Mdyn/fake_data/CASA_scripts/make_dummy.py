import os
import numpy as np

# load information
io = np.loadtxt('dummy.txt', dtype=str)
template = io[0]
chanstart = io[1] + 'km/s'
chanwidth = io[2] + 'km/s'
nchan = np.int(io[3])
restfreq = io[4] + 'GHz'

print(chanstart, chanwidth, nchan, restfreq)

# import the template into MS format
os.system('rm -rf dummy_template.ms')
importuvfits(fitsfile=template, vis='dummy_template.ms')

# regrid
os.system('rm -rf dummy.ms')
mstransform(vis='dummy_template.ms', outputvis='dummy.ms',
            datacolumn='data', regridms=True, mode='velocity',
            start=chanstart, width=chanwidth, nchan=nchan, outframe='LSRK',
            veltype='radio', restfreq=restfreq)

# export the interpolated file into UVFITS
exportuvfits(vis='dummy.ms', fitsfile='dummy.uvfits', datacolumn='data',
             overwrite=True)

# clean up
os.system('rm -rf dummy*.ms')
os.system('rm -rf dummy.txt')
os.system('rm *.last')
