#!/usr/bin/env python

# import numpy as np
import sys
from environment import *
from network import *
from parameters import *
import h5py

snn = SpikingNeuralNetwork()
env = VrepEnvironment()

# Read network weights
path = "./data/session_xyz"
h5f = h5py.File(path + '/rstdp_data.h5', 'r')
w_l = np.array(h5f['w_l'], dtype=float)
w_r = np.array(h5f['w_r'], dtype=float)
w_i = np.array(h5f['w_i'], dtype=float)


e_i_o = np.array(h5f['e_i_o'], dtype=int)
e_i_i = np.array(h5f['e_i_i'], dtype=int)
e_i = np.sort(np.concatenate((e_i_i, e_i_o), axis=0))

print e_i
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

	for i in range(17000):
		# Simulate network for 50 ms
		# Get left and right output spikes, get weights
		n_l, n_r, _, _ = snn.simulate(s, r)

		# Feed output spikes into steering wheel model
		# Get state, distance, position, reward, termination, step, lane
		s, d, p, r, t, n, o = env.step(n_l, n_r, episode)

		# Break episode if robot reaches starting position again
		# if p == env.d_outer or p == env.d_inner:

		if t:
			episode = episode + 1
			break



