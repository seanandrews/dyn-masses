import numpy as np
import sys
import os
import time
import yaml
from sim_grid import sim_grid
from sim_disk import sim_disk
from plotutils import plotutils
from fitsconversion import convert_to_fits
from raytrace_maps import raytrace_maps


model_name = 'demo_wcyl'

print('making structure')
t0 = time.time()
di = sim_disk(model_name, cyl=True)
print(time.time()-t0)

_ = plotutils(model_name)

#T = di.temperature(r=1.496e15, z=np.array([0., 1.496e14]), **sim_disk.T_args)


#print(di.scaleheight(r=1.496e15, T=T) / 1.496e15)

#print('plotting structures')

# structure and plot in cylindrical coordinates
di = sim_disk(model_name, writestruct=False, cyl=True)
_ = plotutils(model_name, struct=di, cyl=True)	

sys.exit()

os.chdir(model_name)
os.system('radmc3d sed ' + \
          'incl %.2f ' % 60 + \
          'setthreads 4')


sys.exit()

# raytrace out a set of channel maps
print('raytracing')
t0 = time.time()
ch_maps = raytrace_maps(model_name)
print(time.time()-t0)
