#!/usr/bin/env python

import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib import gridspec
from environment import VrepEnvironment
from parameters import *
import math

# Performance graph scenario 3
# Fig. 5.13

env = VrepEnvironment()

x1 = env.d1_outer
x2 = x1 + env.d2_outer
x3 = x2 + env.d3_outer
x4 = x3 + env.d4_outer
x5 = x4 + env.d5_outer
x6 = x5 + env.d6_outer

lim = 0.2

path3 = "./data/session_xyz"

h5f4 = h5py.File(path3 + '/rstdp_performance_data.h5', 'r')

distance4 = np.array(h5f4['distance'], dtype=float)
position4 = np.array(h5f4['position'], dtype=float)

# distance2 = distance4[2:]
# position2 = position4[2:]
# distance4 = np.append(distance2, [-0.00003, -0.00002])
# position4 = np.append(position2, [31.88, 31.99])
# # Save performance data
# h5f = h5py.File(path + '/rstdp_performance_data_final.h5', 'w')
# h5f.create_dataset('distance', data=distance4)
# h5f.create_dataset('position', data=position4)
# h5f.close()



fig1 = plt.figure(figsize=(9, 1.6))

gs = gridspec.GridSpec(1, 2, width_ratios=[6, 1])

ax7 = plt.subplot(gs[0])
plt.axhline(y=0., linewidth=0.5, color='0.')
plt.axvline(x=x1, linestyle='--', color='0.75')
plt.axvline(x=x2, linestyle='--', color='0.75')
plt.axvline(x=x3, linestyle='--', color='0.75')
plt.axvline(x=x4, linestyle='--', color='0.75')
plt.axvline(x=x5, linestyle='--', color='0.75')

plt.plot(position4, distance4, color='g')
ax7.set_xlim((0, x6))
ax7.set_ylim((-lim, lim))
ax7.set_xlabel('Position [m]')
ax7.set_ylabel('Distance to\nLane-Center [m]')#, position=(0.,0.))
ax7.tick_params(axis='both', which='both', direction='in', bottom=True, top=True, left=True, right=True)

s4 = abs(distance4).mean()
mse = math.sqrt(sum([x ** 2 for x in distance4]) / len(distance4))  # mean square error, MSE
b = [x*0.01 for x in range(-23,24)]

ax8 = plt.subplot(gs[1])
ax8.set_xticklabels([])
ax8.set_yticklabels([])
ax8.set_ylim((-lim,lim))
ax8.tick_params(axis='both', which='both', direction='in', bottom=True, top=True, left=True)
ax8.set_title('R-STDP \ne = '+str('{:4.3f}'.format(s4)) + '\nMSE =' + str('{:4.3f}'.format(mse)), loc='left', size='medium', position=(1.1, 0.2))
plt.axhline(y=0, linewidth=0.5, color='0.')
ax8.set_xlabel('Histogram')
plt.hist(distance4, bins=b, normed=True, color='g', linewidth=2, orientation=u'horizontal')

plt.subplots_adjust(wspace=0., hspace=0.1, right=0.89, left=0.09, bottom=0.29)
plt.savefig('performancegraph_distance_way2.svg', format='svg', dpi=600)
plt.show()
