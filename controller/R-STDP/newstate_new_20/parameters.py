#!/usr/bin/env python

import numpy as np

path = "./data/session_xyz"			# Path for saving data

# Input image
dvs_resolution = [128,128]			# Original DVS frame resolution
crop_top = 40						# Crop at the top
crop_bottom = 24					# Crop at the bottom
resolution = [4,8]					# Resolution of reduced image
                                    # newstate = [4,4]; newstate_new = [4,8]; fcn = [8,8]; fcn_right = [4,4]

# Network parameters
sim_time = 50.0						# Length of network simulation during each step in ms
t_refrac = 2.						# Refractory period
time_resolution = 0.1				# Network simulation time resolution
iaf_params = {}						# IAF neuron parameters
poisson_params = {}					# Poisson neuron parameters
max_poisson_freq = 300.				# Maximum Poisson firing frequency for n_max  # 300.  #FCN_8shape: 225.
max_spikes = 20.	 				# number of events during each step for maximum poisson frequency
                                    # newstate = 130.; # newstate_new = [20., 22.]; # newstate_fcn = 5.3; # newstate_diff (X:160,200,250,130,90); newstate_fcn_right = 8.9
# R-STDP parameters
w_min = 0.							# Minimum weight value
w_max = 3000.						# Maximum weight value
w0_min = 200.						# Minimum initial random value
w0_max = 201.						# Maximum initial random value
tau_n = 200.						# Time constant of reward signal
tau_c = 1000.						# Time constant of eligibility trace
reward_factor = 0.01				# Reward factor modulating reward signal strength
A_plus = 1.							# Constant scaling strength of potentiation
A_minus = 1.						# Constant scaling strength of depression				

# Steering wheel model
v_max = 1.5							# Maximum speed
v_min = 1.							# Minimum speed
turn_factor= 0.5					# Factor controls turn radius
turn_pre = 0						# Initial turn speed
v_pre = v_max						# Initial speed
n_max = sim_time//t_refrac - 10.	# Maximum input activity (50//2-10 = 15)

# Other
reset_distance = 0.25				# Reset distance   # 0.25   FCN_right:0.15
rate = 20.							# ROS publication rate motor speed
training_length = 100000			# Lenth of training procedure (1 step ~ 50 ms)   #100000  #fcn:18000
max_step = 10000                 # Termination  # 10000  #fcn:1000/1300; FCN_8shape:2000; FCN_right: 2500/4500
