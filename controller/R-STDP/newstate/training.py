#!/usr/bin/env python

# import numpy as np
from environment import *
from network import *
from parameters import *
import h5py


snn = SpikingNeuralNetwork()
env = VrepEnvironment()
weights_r = []
weights_l = []
weights_i = []
episode_position_o = []
episode_i_o = []
episode_position_i = []
episode_i_i = []
episode = 0
episode_step = []
# Initialize environment, get initial state, initial reward
s,r = env.reset()

for i in range(training_length):

	# Simulate network for 50 ms
	# get number of output spikes and network weights
	n_l, n_r, w_l, w_r = snn.simulate(s, r)

	# Feed output spikes in steering wheel model
	# get state, distance, position, reward, termination, steps, lane
	s, d, p, r, t, n, o = env.step(n_l, n_r, episode)

	# Save weights every 100 simulation steps
	if i % 100 == 0:
		weights_l.append(w_l)
		weights_r.append(w_r)
		weights_i.append(i)

	# Save last position if episode is terminated
	if t:
		if o:
			episode_position_o.append(p)
			episode_i_o.append(i)
		else:
			episode_position_i.append(p)
			episode_i_i.append(i)
		episode = episode + 1
		episode_step.append(n)
		print "Episode: " + str(episode) + "    Total steps: " + str(i) + "    Steps: " + str(n)


# Save data
h5f = h5py.File(path + '/rstdp_data.h5', 'w')
h5f.create_dataset('w_l', data=weights_l)
h5f.create_dataset('w_r', data=weights_r)
h5f.create_dataset('w_i', data=weights_i)
h5f.create_dataset('e_o', data=episode_position_o)
h5f.create_dataset('e_i_o', data=episode_i_o)
h5f.create_dataset('e_i', data=episode_position_i)
h5f.create_dataset('e_i_i', data=episode_i_i)
h5f.close()

# Save episode step
np.save('episodes.npy', episode_step)
