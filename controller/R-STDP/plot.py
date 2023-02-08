#!/usr/bin/env python


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

new_state = np.random.randint(0, 4, size=(8, 8), dtype='int')
print(new_state)
max_state = np.max(new_state)
print(max_state)
# fig = plt.figure(figsize=(4, 7.2))

fig = plt.figure()

gs = gridspec.GridSpec(2, 1, height_ratios=[8, 1])
ax1 = plt.subplot(gs[0])

# ax1 = plt.subplot(211)
plt.title('Input State')
plt.imshow(new_state, alpha=0.7, cmap='cool', vmax=max_state, interpolation="nearest")
plt.axis('off')
plt.colorbar(shrink=.5)

n_max = 15
n = 3
m = 10
output_spikes = np.array([[n, m]]).reshape((1,2))
# output_spikes = np.random.randint(0, 15, size=(1, 2), dtype='int')
print(output_spikes)
ax2 = plt.subplot(gs[1])
# ax2 = plt.subplot(212)
plt.title('Output Spikes')
plt.imshow(output_spikes, alpha=0.7, cmap='Wistia', vmax=n_max, aspect='auto')   #PuBu
plt.axis('off')

ax2.text(0,0,int(n),ha='center',va='center')
ax2.text(1,0,int(m),ha='center',va='center')

# fig.tight_layout()
#plt.savefig(path + '/' + sys.argv[1] + '.weights.pdf', dpi=300)
plt.show()

#
# import numpy as np
# n = 3
# m = 4
# a = np.array([n, m])
# print (a, a.size)
#
# import numpy as np
# a = np.array([[3,3,2,1],
#               [0,0,1,5],
#               [3,1,2,0],
#               [5,3,1,0]])
# b = np.rot90(a, 1)  # rotate 90
# print(b)




#!/usr/bin/env python

# import numpy as np
import sys
from environment import *
from network import *
from parameters import *
import h5py

snn = SpikingNeuralNetwork()
env = CarlaEnvironment()

# Read network weights
h5f = h5py.File(path + '/' + sys.argv[1], 'r')
w_l = np.array(h5f['w_l'], dtype=float)
w_r = np.array(h5f['w_r'], dtype=float)
w_i = np.array(h5f['w_i'], dtype=float)

if 'e_i' in h5f.keys():
	e_i = np.array(h5f['e_i'], dtype=int)
else:
	e_i_o = np.array(h5f['e_i_o'], dtype=int)
	e_i_i = np.array(h5f['e_i_i'], dtype=int)
	e_i = np.sort(np.concatenate((e_i_i, e_i_o), axis=0))

replay_weights = []

for i in e_i:
	idx = int((np.abs(w_i - i)).argmin())
	print idx
	replay_weights.append([w_l[idx], w_r[idx]])

# Initialize environment, get state, get reward
s, r = env.reset()

episode = 1
for weight in replay_weights:
	print "Weight: {}/{}".format(episode, len(replay_weights))
	# Set network weights
	snn.set_weights(weight[0], weight[1])


	for i in range(500):
		# Simulate network for 50 ms
		# Get left and right output spikes, get weights
		n_l, n_r, _, _ = snn.simulate(s, r)

		# Feed output spikes into steering wheel model
		# Get state, distance, position, reward, termination, step, lane
		s, d, p, r, t, n = env.step(n_l, n_r, episode)

		# Break episode if robot reaches starting position again
		# if p == env.d_outer or p == env.d_inner:

		if t:
			episode = episode + 1
			break



