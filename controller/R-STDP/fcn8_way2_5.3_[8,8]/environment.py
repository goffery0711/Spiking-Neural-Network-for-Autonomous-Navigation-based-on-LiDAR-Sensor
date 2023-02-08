#!/usr/bin/env python

import sys
sys.path.append('/usr/lib/python2.7/dist-packages') # weil ROS nicht mit Anaconda installiert
import rospy

import math
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import sim
from collections import deque
# import pcl
# from pcl import PointCloud
# import open3d as o3d
# import re
# import pandas as pd
from preprocess import *
import os.path
import tensorflow as tf
import fcn8_helper


from std_msgs.msg import Int8MultiArray, Float32, Bool, Float32MultiArray
from sensor_msgs.msg import PointCloud
from geometry_msgs.msg import Transform

from parameters import *
import cv2


class VrepEnvironment():
	def __init__(self):
		self.preprocess = Radarpreprocess()

		# for saving .csv file
		self.pos = []
		self.ori = []
		self.count = 0

		# self.vld_sub = rospy.Subscriber('velodyneData', PointCloud, self.preprocess.vld_callback_new)
		## self.vld_sub = rospy.Subscriber('velodyneData', PointCloud, self.preprocess.vld_callback_diff)
		# self.vld_sub = rospy.Subscriber('velodyneData', PointCloud, self.preprocess.vld_callback)

		# -----  FCN8 Init  -----
		self.num_classes = 2
		self.image_shape = (160, 576)
		self.data_dir = './fcn_vgg/data'


		gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
		config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
		self.sess = tf.Session(config=config)

		# Path to vgg model
		vgg_path = os.path.join(self.data_dir, 'vgg')

		# Build NN using load_vgg, layers, and optimize function
		self.image_input, self.keep_prob, layer3_out, layer4_out, layer7_out = fcn8_helper.load_vgg(sess=self.sess, vgg_path=vgg_path)
		nn_last_layer = fcn8_helper.layers(vgg_layer3_out=layer3_out, vgg_layer4_out=layer4_out, vgg_layer7_out=layer7_out,
							   num_classes=self.num_classes)
		correct_label = tf.placeholder(dtype=tf.float32, shape=(None, None, None, self.num_classes))
		learning_rate = tf.placeholder(dtype=tf.float32)
		self.logits, train_op, cross_entropy_loss = fcn8_helper.optimize(nn_last_layer=nn_last_layer, correct_label=correct_label,
														learning_rate=learning_rate, num_classes=self.num_classes)

		saver = tf.train.Saver()
		saver.restore(self.sess, tf.train.latest_checkpoint('.'))

		self.vld_im_sub = rospy.Subscriber('velodyneData', PointCloud, self.vld_saveimg_callback)
		# -----  FCN8 Init  -----

		# self.vld_sub = rospy.Subscriber('vldData', Float32MultiArray, self.vld_callback)
		# self.dvs_sub = rospy.Subscriber('dvsData', Float32MultiArray, self.dvs_callback)
		self.pos_sub = rospy.Subscriber('transformData', Transform, self.pos_callback)
		self.left_pub = rospy.Publisher('leftMotorSpeed', Float32, queue_size=1)
		self.right_pub = rospy.Publisher('rightMotorSpeed', Float32, queue_size=1)
		self.reset_pub = rospy.Publisher('resetRobot', Bool, queue_size=None)
		self.vld_queue = 4   # collect 4 times to get the whole data
		self.fifo = deque(np.zeros((self.vld_queue, )))
		# self.vld_data = np.array(())
		self.dvs_data = np.array([0,0])
		self.pos_data = []
		self.distance = 0
		self.steps = 0
		self.v_pre = v_pre
		self.turn_pre = turn_pre
		self.resize_factor = [dvs_resolution[0]//resolution[0], (dvs_resolution[1]-crop_bottom-crop_top)//resolution[1]]
		# (128/8, (128-40-24)/4)=(16,16)
		self.outer = False
		rospy.init_node('dvs_controller')
		self.rate = rospy.Rate(rate)
		self.clientID = sim.simxStart('127.0.0.1', 19997, True, True, 5000, 5)
		
		#Some values for calculating distance to center of lane
		self.v1 = 2.5
		self.v2 = 7.5
		self.scale = 1.0
		self.c1 = np.array([self.scale*self.v1,self.scale*self.v1])
		self.c2 = np.array([self.scale*self.v2,self.scale*self.v2])
		self.c3 = np.array([self.scale*self.v2,self.scale*self.v1])
		self.c4 = np.array([self.scale*self.v1,self.scale*self.v2])
		self.r1_outer = self.scale*(self.v1-0.25)
		self.r2_outer = self.scale*(self.v1+0.25)
		self.l1_outer = self.scale*(self.v1+self.v2-0.25)
		self.l2_outer = self.scale*0.25
		self.r1_inner = self.scale*(self.v1-0.75)
		self.r2_inner = self.scale*(self.v1+0.75)
		self.l1_inner = self.scale*(self.v1+self.v2-0.75)
		self.l2_inner = self.scale*0.75
		self.d1_outer = 5.0
		self.d2_outer = 2*math.pi*self.r1_outer*0.25
		self.d3_outer = 5.0
		self.d4_outer = 2*math.pi*self.r1_outer*0.5
		self.d5_outer = 2*math.pi*self.r2_outer*0.25
		self.d6_outer = 2*math.pi*self.r1_outer*0.5
		self.d1_inner = 5.0
		self.d2_inner = 2*math.pi*self.r1_inner*0.25
		self.d3_inner = 5.0
		self.d4_inner = 2*math.pi*self.r1_inner*0.5
		self.d5_inner = 2*math.pi*self.r2_inner*0.25
		self.d6_inner = 2*math.pi*self.r1_inner*0.5
		self.d_outer = self.d1_outer + self.d2_outer + self.d3_outer + self.d4_outer + self.d5_outer + self.d6_outer
		self.d_inner = self.d1_inner + self.d2_inner + self.d3_inner + self.d4_inner + self.d5_inner + self.d6_inner
		# self.n = 1
		# for square scene
		self.d1_outer_square = 5.0
		self.d2_outer_square = 2 * math.pi * self.r1_outer * 0.25
		self.d3_outer_square = 5.0
		self.d4_outer_square = 2 * math.pi * self.r1_outer * 0.25
		self.d5_outer_square = 5.0
		self.d6_outer_square = 2 * math.pi * self.r1_outer * 0.25
		self.d7_outer_square = 5.0
		self.d8_outer_square = 2 * math.pi * self.r1_outer * 0.25
		self.d1_inner_square = 5.0
		self.d2_inner_square = 2 * math.pi * self.r1_inner * 0.25
		self.d3_inner_square = 5.0
		self.d4_inner_square = 2 * math.pi * self.r1_inner * 0.25
		self.d5_inner_square = 5.0
		self.d6_inner_square = 2 * math.pi * self.r1_inner * 0.25
		self.d7_inner_square = 5.0
		self.d8_inner_square = 2 * math.pi * self.r1_inner * 0.25
		self.d_outer_square = self.d1_outer_square + self.d2_outer_square + self.d3_outer_square + self.d4_outer_square + self.d5_outer_square + self.d6_outer_square + self.d7_outer_square + self.d8_outer_square
		self.d_inner_square = self.d1_inner_square + self.d2_inner_square + self.d3_inner_square + self.d4_inner_square + self.d5_inner_square + self.d6_inner_square + self.d7_inner_square + self.d8_inner_square
		# for eightshape scene
		self.horizontal = 0
		self.d1_outer_eightshape = 2 * math.pi * (self.r1_outer - 0.25) * 0.25
		self.d2_outer_eightshape = 4.0
		self.d3_outer_eightshape = 2 * math.pi * (self.r1_outer - 0.25) * 0.5
		self.d4_outer_eightshape = 2 * math.pi * (self.r1_outer - 0.25) * 0.25
		self.d5_outer_eightshape = 4.0
		self.d6_outer_eightshape = 2 * math.pi * (self.r1_outer - 0.25) * 0.5
		self.c5 = np.array([6.5, 6.5])
		self.d_outer_eightshape = self.d1_outer_eightshape + self.d2_outer_eightshape + self.d3_outer_eightshape + self.d4_outer_eightshape + self.d5_outer_eightshape + self.d6_outer_eightshape

		# ########################PLOT STATE##################################
		# plt.ion()
		# self.fig = plt.figure(figsize=(2, 4.5))
		#
		# spec = gridspec.GridSpec(ncols=1, nrows=2, height_ratios=[3, 1])
		# ax1 = self.fig.add_subplot(spec[0])
		# plt.title('Lane Feature')
		# self.state_plot = ax1.imshow(np.zeros((resolution[1], resolution[0])), alpha=1, cmap='PuBu', vmax=1, aspect='equal')
		# plt.axis('off')
		#
		# self.motor_ax2 = self.fig.add_subplot(spec[1])
		# plt.title('Actuator Output')
		# self.motor_plot = self.motor_ax2.imshow(np.zeros((2, 1)), alpha=1, cmap='PuBu', vmax=1, aspect='auto')
		# self.motor_l_value_plot = self.motor_ax2.text(-0.25, 1.0, 0, ha='center', va='center')
		# self.motor_r_value_plot = self.motor_ax2.text(0.25, 1.0, 0, ha='center', va='center')
		# plt.axis('off')
		# #####################################################################

	def vld_saveimg_callback(self, msg):
		# print(type(msg), 'type')   # (<class 'sensor_msgs.msg._PointCloud.PointCloud'>, 'type')
		# list = msg.points
		# print(list[:], 'right')
		# points = np.asarray(msg.points)
		# print(points.shape)
		# print(points[0], 'x')
		# print(points[1], 'y')
		# print(type(points[1]), 'type')

		x_points = []
		y_points = []
		z_points = []

		points = np.asarray(msg.points)

		for point in points:
			x_points.append(point.x)   # (16646,)
			y_points.append(point.y)   # (16646,)
			z_points.append(point.z)   # (16646,)

		x_points = np.array(x_points)
		y_points = np.array(y_points)
		z_points = np.array(z_points)

		side_range = (-1, 1)  # left-most to right-most
		fwd_range = (0, 4)  # back-most to forward-most

		# FILTER - To return only indices of points within desired cube
		# Three filters for: Front-to-back, side-to-side, and height ranges
		# Note left side is positive y axis in LIDAR coordinates

		s_filt = np.logical_and((x_points > -side_range[1]), (x_points < -side_range[0]))
		f_filt = np.logical_and((y_points > fwd_range[0]), (y_points < fwd_range[1]))
		filter = np.logical_and(f_filt, s_filt)
		indices = np.argwhere(filter).flatten()

		# KEEPERS
		x_points = x_points[indices]  # (4952,)
		y_points = y_points[indices]  # (4952,)
		z_points = z_points[indices]  # (4952,)

		res = 0.01
		# CONVERT TO PIXEL POSITION VALUES - Based on resolution
		x_img = (x_points / res).astype(np.int32)  # x axis is -y in LIDAR
		y_img = (y_points / res).astype(np.int32)  # y axis is -x in LIDAR

		# SHIFT PIXELS TO HAVE MINIMUM BE (0,0)
		# floor and ceil used to prevent anything being rounded to below 0 after shift
		x_img -= int(np.floor(side_range[0] / res))
		y_img -= int(np.floor(fwd_range[0] / res))

		height_range = (-0.3, -0.2)  # bottom-most to upper-most

		# CLIP HEIGHT VALUES - to between min and max heights
		pixel_values = np.clip(a=z_points, a_min=height_range[0], a_max=height_range[1])

		def scale_to_255(a, min, max, dtype=np.uint8):
			""" Scales an array of values from specified min, max range to 0-255
                Optionally specify the data type of the output (default is uint8)
            """
			return (((a - min) / float(max - min)) * 255).astype(dtype)

		# RESCALE THE HEIGHT VALUES - to be between the range 0-255
		pixel_values = scale_to_255(pixel_values, min=height_range[0], max=height_range[1])

		# INITIALIZE EMPTY ARRAY - of the dimensions we want
		x_max = int((side_range[1] - side_range[0]) / res)  # 200
		y_max = int((fwd_range[1] - fwd_range[0]) / res)  # 400
		self.vld_im = np.zeros([y_max, x_max], dtype=np.uint8)

		# FILL PIXEL VALUES IN IMAGE ARRAY
		self.vld_im[y_img, x_img] = pixel_values
		self.vld_im = np.rot90(self.vld_im, 2)  # rotate 180
		return

	def dvs_callback(self, msg):
		# Store incoming DVS data
		self.dvs_data = msg.data
		return

	def pos_callback(self, msg):
		# # Store incoming position data
		# if self.preprocess.i % 10 == 0:
		# 	if self.count == 0:
		# 		self.pos.append(msg.translation.x)
		# 		self.pos.append(msg.translation.y)
		# 		self.ori.append(msg.rotation.x)
		# 		self.ori.append(msg.rotation.y)
		# 		self.ori.append(msg.rotation.z)
		# 		self.ori.append(msg.rotation.w)
		# 		self.count = self.count + 1
		# else:
		# 	self.count = 0
		#
		# if self.preprocess.i <= 1200:
		# 	position = pd.DataFrame(self.pos)
		# 	position.to_csv('pos.csv')
		# 	orientation = pd.DataFrame(self.ori)
		# 	orientation.to_csv('ori.csv')

		self.pos_data = np.array([msg.translation.x, msg.translation.y, time.time()])
		return

	def allreset(self):
		if self.clientID != -1:
			sim.simxStopSimulation(self.clientID, sim.simx_opmode_oneshot)
			time.sleep(2)
			sim.simxStartSimulation(self.clientID, sim.simx_opmode_oneshot)
			time.sleep(1)
			print('Server is reset correctly!')
		# Reset steering wheel model
		self.left_pub.publish(0.)
		self.right_pub.publish(0.)
		self.v_pre = v_min
		self.turn_pre = 0.
		# Change lane
		self.outer = not self.outer
		self.reset_pub.publish(Bool(self.outer))
		time.sleep(1)
		return np.zeros((resolution[0], resolution[1]), dtype=int), 0.

	def reset(self):
		# Reset steering wheel model
		self.left_pub.publish(0.)
		self.right_pub.publish(0.)
		self.v_pre = v_min
		self.turn_pre = 0.
		# Change lane
		self.outer = not self.outer
		self.reset_pub.publish(Bool(self.outer))
		time.sleep(1)
		return np.zeros((resolution[0], resolution[1]), dtype=int), 0.

	def step(self, n_l, n_r, episode):

		self.steps += 1
		t = False # terminal state

		self.n_l = n_l
		self.n_r = n_r  # for plot

		# Steering wheel model
		m_l = n_l/n_max
		m_r = n_r/n_max
		a = m_l - m_r
		v_cur = - abs(a)*(v_max - v_min) + v_max
		turn_cur = turn_factor * a
		c = math.sqrt((m_l**2 + m_r**2)/2.0)
		self.v_pre = c*v_cur + (1-c)*self.v_pre
		self.turn_pre = c*turn_cur + (1-c)*self.turn_pre
		
		# Publish motor speeds
		# if self.v_pre+self.turn_pre >= 0:
		# 	self.left_pub.publish(self.v_pre+self.turn_pre)
		# else:
		# 	self.left_pub.publish(0)
		#
		# if self.v_pre-self.turn_pre >= 0:
		# 	self.right_pub.publish(self.v_pre-self.turn_pre)
		# else:
		# 	self.right_pub.publish(0)
		self.left_pub.publish(self.v_pre + self.turn_pre)
		self.right_pub.publish(self.v_pre - self.turn_pre)
		self.rate.sleep()
		
		# Get position and distance
		d, p = self.getDistance(self.pos_data)
		# d, p = self.getDistance_right(self.pos_data)
		# d, p = self.getDistance_square(self.pos_data)
		# d, p = self.getDistance_eightshape(self.pos_data)

		# Set reward signal
		if self.outer == (d > 0):
			r = abs(d)
		else:
			r = -abs(d)

		self.distance = d

		## s = self.preprocess.getNewState_diff()
		# s = self.preprocess.getNewState_new()
		# s = self.preprocess.getNewState()
		# self.preprocess.showradarplot_new()
		# self.preprocess.showradarplot()
		# self.showradarplot()

		# FCN8
		s = fcn8_helper.save_inference_samples(self.vld_im, self.sess, self.image_shape, self.logits, self.keep_prob, self.image_input)
		# self.state = s
		# self.showradarplot_right()

		# ################PLOT###################
		# self.state_plot.set_data(np.flipud(s.T))
		# self.motor_plot.set_data(np.array([[m_l, m_r]]))
		# self.motor_l_value_plot.set_text('%.2f' % m_l)
		# self.motor_r_value_plot.set_text('%.2f' % m_r)
		# self.fig.canvas.draw()
		# self.fig.canvas.flush_events()
		# ########################################
		# cv2.imshow("img", self.vld_im)
		# cv2.waitKey(1)
		# ########################################

		n = self.steps
		lane = self.outer

		# Terminate episode of robot reaches start position again
		# or reset distance
		if abs(d) > reset_distance or n >= max_step:   # or p < 0.49
			if n >= max_step:
				if self.outer:
					p = self.d_outer
					# p = self.d_outer_square
					# p = self.d_outer_eightshape
				else:
					p = self.d_inner
					# p = self.d_inner_square
			self.steps = 0
			t = True
			# print('steps:', n)
			# print('distance:', d)
			print('position:', p)
			if episode % 30 == 0:
				self.allreset()
				time.sleep(8)
			else:
				self.reset()
				time.sleep(2)

		# Return state, distance, position, reward, termination, steps, lane
		return s,d,p,r,t,n,lane

	def getDistance(self,p):
		# 180 turn for x < 2.5
		if p[0] < self.scale*self.v1:
			r = np.linalg.norm(p[:2]-self.c1)
			delta_y = p[1] - self.c1[1]
			if self.outer:
				a = abs(math.acos(delta_y / r)/math.pi)
				position = self.d1_outer + self.d2_outer + self.d3_outer + self.d4_outer + self.d5_outer + a * self.d6_outer
				distance = r - (self.r1_outer - 0.25)
				return distance, position
			else:
				a = 1. - abs(math.acos(delta_y / r)/math.pi)
				position = self.d1_inner + self.d2_inner + self.d3_inner + a * self.d4_inner
				distance = r - (self.r1_inner + 0.25)
				return distance, position
		# 180 turn for y > 7.5
		elif p[1] > self.scale*self.v2:
			r = np.linalg.norm(p[:2]-self.c2)
			delta_x = p[0] - self.c2[0]
			if self.outer:
				a = abs(math.acos(delta_x / r)/math.pi)
				position = self.d1_outer + self.d2_outer + self.d3_outer + a * self.d4_outer
				distance = r - (self.r1_outer - 0.25)
				return distance, position
			else:
				a = 1. - abs(math.acos(delta_x / r)/math.pi)
				position = self.d1_inner + self.d2_inner + self.d3_inner + self.d4_inner + self.d5_inner + a * self.d6_inner
				distance = r - (self.r1_inner + 0.25)
				return distance, position
		# x > 7.5
		elif p[0] > self.scale*self.v2:
			# 90 turn for y < 2.5
			if p[1] < self.scale*self.v1:
				r = np.linalg.norm(p[:2]-self.c3)
				delta_x = p[0] - self.c3[0]
				if self.outer:
					a = abs(math.asin(delta_x / r)/(0.5*math.pi))
					position = self.d1_outer + a * self.d2_outer
					distance = r - (self.r1_outer - 0.25)
					return distance, position
				else:
					a = 1. - abs(math.asin(delta_x / r)/(0.5*math.pi))
					position = self.d1_inner + a * self.d2_inner
					distance = r - (self.r1_inner + 0.25)
					return distance, position
			# straight for 2.5 < y < 7.5
			else:
				if self.outer:
					distance = (p[0] - (self.l1_outer - 0.25))
					position = self.d1_outer + self.d2_outer + abs(p[1] - self.v1)
					return distance, position
				else:
					distance = (p[0] - (self.l1_inner + 0.25))
					position = abs(p[1] - self.v2)
					return distance, position
		
		else:
			# straight for y < 2.5
			if p[1] < self.scale*self.v1:
				if self.outer:
					distance = (p[1] - (self.l2_outer + 0.25))*(-1)
					position = abs(p[0] - self.v1)
					return distance, position
				else:
					distance = (p[1] - (self.l2_inner - 0.25))*(-1)
					position = self.d1_inner + self.d2_inner + abs(p[0] - self.v2)
					return distance, position
			# 90 turn for x,y between 2.5 and 7.5
			else:
				r = np.linalg.norm(p[:2]-self.c4)
				delta_y = p[1] - self.c4[1]
				if self.outer:
					a = abs(math.asin(delta_y / r)/(0.5*math.pi))
					position = self.d1_outer + self.d2_outer + self.d3_outer + self.d4_outer + a * self.d5_outer
					distance = (r - (self.r2_outer + 0.25))*(-1)
					return distance, position
				else:
					a = 1. - abs(math.asin(delta_y / r)/(0.5*math.pi))
					position = self.d1_inner + self.d2_inner + self.d3_inner + self.d4_inner + a * self.d5_inner
					distance = (r - (self.r2_inner - 0.25))*(-1)
					return distance, position

	def getDistance_right(self, p):
		# 180 turn for x < 2.5
		if p[0] < self.scale * self.v1:
			r = np.linalg.norm(p[:2] - self.c1)
			delta_y = p[1] - self.c1[1]
			if self.outer:
				a = abs(math.acos(delta_y / r) / math.pi)
				position = self.d1_outer + self.d2_outer + self.d3_outer + self.d4_outer + self.d5_outer + a * self.d6_outer
				distance = r - self.r1_outer
				return distance, position
			else:
				a = 1. - abs(math.acos(delta_y / r) / math.pi)
				position = self.d1_inner + self.d2_inner + self.d3_inner + a * self.d4_inner
				distance = r - self.r1_inner
				return distance, position
		# 180 turn for y > 7.5
		elif p[1] > self.scale * self.v2:
			r = np.linalg.norm(p[:2] - self.c2)
			delta_x = p[0] - self.c2[0]
			if self.outer:
				a = abs(math.acos(delta_x / r) / math.pi)
				position = self.d1_outer + self.d2_outer + self.d3_outer + a * self.d4_outer
				distance = r - self.r1_outer
				return distance, position
			else:
				a = 1. - abs(math.acos(delta_x / r) / math.pi)
				position = self.d1_inner + self.d2_inner + self.d3_inner + self.d4_inner + self.d5_inner + a * self.d6_inner
				distance = r - self.r1_inner
				return distance, position
		# x > 7.5
		elif p[0] > self.scale * self.v2:
			# 90 turn for y < 2.5
			if p[1] < self.scale * self.v1:
				r = np.linalg.norm(p[:2] - self.c3)
				delta_x = p[0] - self.c3[0]
				if self.outer:
					a = abs(math.asin(delta_x / r) / (0.5 * math.pi))
					position = self.d1_outer + a * self.d2_outer
					distance = r - self.r1_outer
					return distance, position
				else:
					a = 1. - abs(math.asin(delta_x / r) / (0.5 * math.pi))
					position = self.d1_inner + a * self.d2_inner
					distance = r - self.r1_inner
					return distance, position
			# straight for 2.5 < y < 7.5
			else:
				if self.outer:
					distance = (p[0] - self.l1_outer)
					position = self.d1_outer + self.d2_outer + abs(p[1] - self.v1)
					return distance, position
				else:
					distance = (p[0] - self.l1_inner)
					position = abs(p[1] - self.v2)
					return distance, position

		else:
			# straight for y < 2.5
			if p[1] < self.scale * self.v1:
				if self.outer:
					distance = (p[1] - self.l2_outer) * (-1)
					position = abs(p[0] - self.v1)
					return distance, position
				else:
					distance = (p[1] - self.l2_inner) * (-1)
					position = self.d1_inner + self.d2_inner + abs(p[0] - self.v2)
					return distance, position
			# 90 turn for x,y between 2.5 and 7.5
			else:
				r = np.linalg.norm(p[:2] - self.c4)
				delta_y = p[1] - self.c4[1]
				if self.outer:
					a = abs(math.asin(delta_y / r) / (0.5 * math.pi))
					position = self.d1_outer + self.d2_outer + self.d3_outer + self.d4_outer + a * self.d5_outer
					distance = (r - self.r2_outer) * (-1)
					return distance, position
				else:
					a = 1. - abs(math.asin(delta_y / r) / (0.5 * math.pi))
					position = self.d1_inner + self.d2_inner + self.d3_inner + self.d4_inner + a * self.d5_inner
					distance = (r - self.r2_inner) * (-1)
					return distance, position

	def getDistance_square(self, p):
		# x < 2.5
		if p[0] < self.scale * self.v1:
			# 90 turn for y < 2.5
			if p[1] < self.scale * self.v1:
				r = np.linalg.norm(p[:2] - self.c1)  # get radius
				delta_x = p[0] - self.c1[0]
				if self.outer:
					a = abs(math.asin(delta_x / r) / (0.5 * math.pi))  # get ratio of 90 degrees
					position = self.d1_outer_square + self.d2_outer_square + self.d3_outer_square + self.d4_outer_square + self.d5_outer_square + self.d6_outer_square + self.d7_outer_square + a * self.d8_outer_square
					distance = r - (self.r1_outer - 0.25)
					return distance, position
				else:
					a = 1. - abs(math.asin(delta_x / r) / (0.5 * math.pi))
					position = self.d1_inner_square + self.d2_inner_square + self.d3_inner_square + a * self.d4_inner_square
					distance = r - (self.r1_inner + 0.25)
					return distance, position
			# 90 turn for y > 7.5
			elif p[1] > self.scale * self.v2:
				r = np.linalg.norm(p[:2] - self.c4)  # get radius
				delta_x = p[1] - self.c4[1]
				if self.outer:
					a = abs(math.asin(delta_x / r) / (0.5 * math.pi))  # get ratio of 90 degrees
					position = self.d1_outer_square + self.d2_outer_square + self.d3_outer_square + self.d4_outer_square + self.d5_outer_square + a * self.d6_outer_square
					distance = r - (self.r1_outer - 0.25)
					return distance, position
				else:
					a = 1. - abs(math.asin(delta_x / r) / (0.5 * math.pi))
					position = self.d1_inner_square + self.d2_inner_square + self.d3_inner_square + self.d4_inner_square + self.d5_inner_square + a * self.d6_inner_square
					distance = r - (self.r1_inner + 0.25)
					return distance, position
			# straight for 2.5 < y < 7.5
			else:
				if self.outer:
					distance = (p[0] - (self.l2_outer + 0.25)) * (-1)
					position = self.d1_outer_square + self.d2_outer_square + self.d3_outer_square + self.d4_outer_square + self.d5_outer_square + self.d6_outer_square + abs(p[1] - self.v2)
					return distance, position
				else:
					distance = (p[0] - (self.l2_inner - 0.25)) * (-1)
					position = self.d1_inner_square + self.d2_inner_square + self.d3_inner_square + self.d4_inner_square + abs(p[1] - self.v1)
					return distance, position
		# x > 7.5
		elif p[0] > self.scale * self.v2:
			# 90 turn for y < 2.5
			if p[1] < self.scale * self.v1:
				r = np.linalg.norm(p[:2] - self.c3)  # get radius
				delta_x = p[0] - self.c3[0]
				if self.outer:
					a = abs(math.asin(delta_x / r) / (0.5 * math.pi))  # get ratio of 90 degrees
					position = self.d1_outer_square + a * self.d2_outer_square
					distance = r - (self.r1_outer - 0.25)
					return distance, position
				else:
					a = 1. - abs(math.asin(delta_x / r) / (0.5 * math.pi))
					position = self.d1_inner_square + a * self.d2_inner_square
					distance = r - (self.r1_inner + 0.25)
					return distance, position
			# 90 turn for y > 7.5
			elif p[1] > self.scale * self.v2:
				r = np.linalg.norm(p[:2] - self.c2)  # get radius
				delta_x = p[1] - self.c2[1]
				if self.outer:
					a = abs(math.asin(delta_x / r) / (0.5 * math.pi))  # get ratio of 90 degrees
					position = self.d1_outer_square + self.d2_outer_square + self.d3_outer_square + a * self.d4_outer_square
					distance = r - (self.r1_outer - 0.25)
					return distance, position
				else:
					a = 1. - abs(math.asin(delta_x / r) / (0.5 * math.pi))
					position = self.d1_inner_square + self.d2_inner_square + self.d3_inner_square + self.d4_inner_square + self.d5_inner_square + self.d6_inner_square + self.d7_inner_square + a * self.d8_inner_square
					distance = r - (self.r1_inner + 0.25)
					return distance, position
			# straight for 2.5 < y < 7.5
			else:
				if self.outer:
					distance = (p[0] - (self.l1_outer - 0.25))
					position = self.d1_outer_square + self.d2_outer_square + abs(p[1] - self.v1)
					return distance, position
				else:
					distance = (p[0] - (self.l1_inner + 0.25))
					position = abs(p[1] - self.v2)
					return distance, position
		# 2.5 <= x <= 7.5
		else:
			# straight for y < 2.5
			if p[1] < self.scale * self.v1:
				if self.outer:
					distance = (p[1] - (self.l2_outer + 0.25)) * (-1)
					position = abs(p[0] - self.v1)
					return distance, position
				else:
					distance = (p[1] - (self.l2_inner - 0.25)) * (-1)
					position = self.d1_inner_square + self.d2_inner_square + abs(p[0] - self.v2)
					return distance, position
			# straight for y > 7.5
			elif p[1] > self.scale * self.v2:
				if self.outer:
					distance = (p[1] - (self.l1_outer - 0.25))
					position = self.d1_outer_square + self.d2_outer_square + self.d3_outer_square + self.d4_outer_square + abs(p[0] - self.v2)
					return distance, position
				else:
					distance = (p[1] - (self.l1_inner + 0.25))
					position = self.d1_inner_square + self.d2_inner_square + self.d3_inner_square + self.d4_inner_square + self.d5_inner_square + self.d6_inner_square + abs(p[0] - self.v1)
					return distance, position

	def getDistance_eightshape(self, p):
		# 180 turn for x < 2.5
		if p[0] < self.scale * self.v1:
			r = np.linalg.norm(p[:2] - self.c1)    # get radius
			delta_y = p[1] - self.c1[1]
			# if self.outer:
			a = abs(math.acos(delta_y / r) / math.pi)   # get ratio of 180 degrees
			position = self.d1_outer_eightshape + self.d2_outer_eightshape + self.d3_outer_eightshape + self.d4_outer_eightshape + self.d5_outer_eightshape + a * self.d6_outer_eightshape
			distance = r - (self.r1_outer - 0.25)
			return distance, position

		# 180 turn for y > 6.5
		elif p[1] > 6.5:
			r = np.linalg.norm(p[:2] - self.c5)
			delta_x = self.c5[0] - p[0]
			# if self.outer:
			a = abs(math.acos(delta_x / r) / math.pi)
			position = self.d1_outer_eightshape + self.d2_outer_eightshape + a * self.d3_outer_eightshape
			distance = r - (self.r1_outer - 0.25)
			return distance, position

		# x > 6.5
		elif p[0] > 6.5:
			# 90 turn for y < 6.5 & x > 6.5
			if p[1] < 6.5:
				self.horizontal = 1
				r = np.linalg.norm(p[:2] - self.c5)
				delta_x = p[0] - self.c5[0]
				# if self.outer:
				a = abs(math.acos(delta_x / r) / (0.5 * math.pi))
				position = self.d1_outer_eightshape + self.d2_outer_eightshape + self.d3_outer_eightshape + a * self.d4_outer_eightshape
				distance = r - (self.r1_outer - 0.25)
				return distance, position

		# y <= 2.5
		elif p[1] <= 2.5:
			# 90 turn for y < 2.5 & 2.5 <= x <= 5.0
			if 2.5 <= p[0] <= 5.0:
				self.horizontal = 0
				r = np.linalg.norm(p[:2] - self.c1)
				delta_y = self.c1[1] - p[1]
				# if self.outer:
				a = abs(math.acos(delta_y / r) / (0.5 * math.pi))
				position = a * self.d1_outer_eightshape
				distance = r - (self.r1_outer - 0.25)
				return distance, position

		else:
			if self.horizontal == 0:
				if 4.0 <= p[0] <= 5.0:
					# straight for y --> [2.5, 4.0] & [5.0, 6.5] (including square)
					if 2.5 <= p[1] <= 6.5:
						# if self.outer:
						distance = p[0] - 4.5
						position = self.d1_outer_eightshape + abs(p[1] - 2.5)
						return distance, position

			# straight for x --> [2.5, 4.0] & [5.0, 6.5]
			elif self.horizontal == 1:
				if 2.5 <= p[0] <= 6.5:
					if 4.0 <= p[1] <= 5.0:
						# if self.outer:
						distance = p[1] - 4.5
						position = self.d1_outer_eightshape + self.d2_outer_eightshape + self.d3_outer_eightshape + self.d4_outer_eightshape + abs(p[0] - 6.5)
						return distance, position


	def getState(self):   # 22.09 change: [8,4]--[4,4]--[2,4](0,2.5)
		new_state = np.zeros((resolution[0],resolution[1]),dtype=int)   #(8,4)
		for i in range(len(self.dvs_data)//2): # len(self.dvs_data)//2=128
			try:
				if self.dvs_data[i*2] >= 0:
					state_y = 8*self.dvs_data[i*2]       # y:[0,5]--[0,40]
					state_x = 2*(self.dvs_data[i*2+1]+5) # x:[-5,5]--[0,40]   Then:[40,40]
					if 0 <= state_y <= 8*5//2:    # now only half of them[40,20];  original:24 <= y < (128-40)
						idx = (int(state_x//5), int(state_y//5))  # [40,20]--[8,4]
						# idx = (x//16, (y-24)//16)
						new_state[idx] += 1
			except:
				pass
		# print(new_state)
		return new_state

	def showradarplot(self):
		plt.clf()
		plt.ion()
		# new_state = np.zeros((resolution[0], resolution[1]), dtype=int)  # (4,4)
		vld_data = self.vld_data
		for i in range(len(vld_data) // 2):  # len(self.dvs_data)//2=925
			if -1 <= vld_data[i * 2] <= 1 and 0 <= vld_data[i * 2 + 1] <= 4:
				state_x = round(2 * (vld_data[i * 2] + 1))  # x:[-1,1]--[0,4]
				state_y = round(vld_data[i * 2 + 1])  # y:[0,4]
				if state_x < 4 and state_y < 4:
					# idx = (int(state_x), int(state_y))
					# new_state[idx] += 1
					plt.scatter(int(state_x), int(state_y))
		plt.xlim([0,4])
		plt.ylim([0,4])
		plt.pause(0.001)
		plt.clf()
		plt.ioff()

	def showradarplot_right(self):
		# plt.clf()
		plt.ion()

		max_state = np.max(self.state)

		# fig = plt.figure()
		self.fig = plt.figure(figsize=(2, 4.5))
		gs = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[4, 1])

		# ax1 = plt.subplot(gs[0])
		ax1 = self.fig.add_subplot(gs[0])
		plt.title('Input State')
		ax1.imshow(np.rot90(self.state, 1), alpha=0.7, cmap='PuBu', vmax=max_state, aspect='equal')
		plt.axis('off')

		output_spikes = np.array([[self.n_l, self.n_r]]).reshape((1, 2))
		# ax2 = plt.subplot(gs[1])
		ax2 = self.fig.add_subplot(gs[1])
		plt.title('Output Spikes')
		ax2.imshow(output_spikes, alpha=0.7, cmap='PuBu', vmax=n_max, aspect='auto')
		plt.axis('off')
		ax2.text(0, 0, int(self.n_l), ha='center', va='center')
		ax2.text(1, 0, int(self.n_r), ha='center', va='center')

		plt.pause(0.001)
		# plt.clf()
		plt.ioff()
