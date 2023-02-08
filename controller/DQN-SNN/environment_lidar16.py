#!/usr/bin/env python

import sys
sys.path.append('/usr/lib/python2.7/dist-packages') # weil ROS nicht mit Anaconda installiert

import rospy
import math
import time
import sim
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from collections import deque
from std_msgs.msg import Int8MultiArray, Float32, Bool, Float32MultiArray
from geometry_msgs.msg import Transform
from sensor_msgs.msg import PointCloud

import os.path
import tensorflow as tf
import fcn8_helper

from parameters import *
from preprocess import *


def normpdf(x, mean=0, sd=0.15):
    var = float(sd)**2
    denom = (2*math.pi*var)**.5
    num = math.exp(-(float(x)-float(mean))**2/(2*var))
    return num/denom

class VrepEnvironment():
	# def __init__(self, sess1, speed, turn, resolution, reset_distance, rate, dvs_queue, resize_factor, crop):
	def __init__(self,speed,turn,resolution,reset_distance, rate, dvs_queue, resize_factor, crop):

		self.preprocess = Radarpreprocess()
		self.vld_sub = rospy.Subscriber('velodyneData', PointCloud, self.preprocess.vld_callback)



		# # -----  FCN8 Init  -----
		# self.num_classes = 2
		# self.image_shape = (160, 576)
		# self.data_dir = './fcn_vgg/data'
		#
		# self.sess = sess1
		#
		# # Path to vgg model
		# vgg_path = os.path.join(self.data_dir, 'vgg')
		#
		# # Build NN using load_vgg, layers, and optimize function
		# self.image_input, self.keep_prob, layer3_out, layer4_out, layer7_out = fcn8_helper.load_vgg(sess=self.sess, vgg_path=vgg_path)
		# nn_last_layer = fcn8_helper.layers(vgg_layer3_out=layer3_out, vgg_layer4_out=layer4_out, vgg_layer7_out=layer7_out, num_classes=self.num_classes)
		# correct_label = tf.placeholder(dtype=tf.float32, shape=(None, None, None, self.num_classes))
		# learning_rate = tf.placeholder(dtype=tf.float32)
		# self.logits, train_op, cross_entropy_loss = fcn8_helper.optimize(nn_last_layer=nn_last_layer, correct_label=correct_label, learning_rate=learning_rate, num_classes=self.num_classes)
		#
		# saver = tf.train.Saver()
		# saver.restore(self.sess, tf.train.latest_checkpoint('.'))
		#
		# self.vld_im_sub = rospy.Subscriber('velodyneData', PointCloud, self.vld_saveimg_callback)
		# # -----  FCN8 Init  -----


		self.pos_sub = rospy.Subscriber('transformData', Transform, self.pos_callback)
		self.left_pub = rospy.Publisher('leftMotorSpeed', Float32, queue_size=1)
		self.right_pub = rospy.Publisher('rightMotorSpeed', Float32, queue_size=1)
		self.reset_pub = rospy.Publisher('resetRobot', Bool, queue_size=None)
		self.v_forward = speed
		self.v_turn = turn
		self.resolution = resolution
		self.reset_distance = reset_distance
		self.dvs_queue = dvs_queue   #dvs_queue=10;
		self.fifo = deque(np.zeros((dvs_queue, resolution[0]*resolution[1]))) 
		self.resize_factor = resize_factor
		self.crop = crop
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
		self.l2_outer = self.scale*(0.25)
		self.r1_inner = self.scale*(self.v1-0.75)
		self.r2_inner = self.scale*(self.v1+0.75)
		self.l1_inner = self.scale*(self.v1+self.v2-0.75)
		self.l2_inner = self.scale*(0.75)
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

		# ########################PLOT STATE##################################
		# self.motor_value = np.array([[0, 1, 0]])
		# plt.ion()
		# self.fig = plt.figure(figsize=(2, 4.5))
		#
		# spec = gridspec.GridSpec(ncols=1, nrows=2, height_ratios=[3, 1])
		# ax1 = self.fig.add_subplot(spec[0])
		# plt.title('Lane Feature')
		# # self.state_plot = ax1.imshow(np.zeros((resolution[1], resolution[0])), alpha=1, cmap='PuBu', vmax=1, aspect='equal')
		# self.state_plot = ax1.imshow(np.zeros((13, 40)), alpha=1, cmap='PuBu', vmax=1, aspect='equal')
		# plt.axis('off')
		#
		# self.motor_ax2 = self.fig.add_subplot(spec[1])
		# plt.title('Actuator Output')
		# self.motor_plot = self.motor_ax2.imshow(np.zeros((1, 3)), alpha=1, cmap='PuBu', vmax=1, aspect='auto')
		# self.motor_l_value_plot = self.motor_ax2.text(-0.25, 1.0, 0, ha='center', va='center')
		# self.motor_m_value_plot = self.motor_ax2.text(0, 1.0, 0, ha='center', va='center')
		# self.motor_r_value_plot = self.motor_ax2.text(0.25, 1.0, 0, ha='center', va='center')
		# plt.axis('off')


	def vld_saveimg_callback(self, msg):
		x_points = []
		y_points = []
		z_points = []

		points = np.asarray(msg.points)

		for point in points:
			x_points.append(point.x)  # (16646,)
			y_points.append(point.y)  # (16646,)
			z_points.append(point.z)  # (16646,)

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

	def pos_callback(self, msg):
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
		# Change lane
		self.outer = not self.outer
		self.reset_pub.publish(Bool(self.outer))
		time.sleep(1)
		return np.zeros((self.resolution[0]*self.resolution[1]),dtype=int)

	def reset(self):
		# Reset robot, change lane and return empty state
		self.outer = not self.outer
		self.reset_pub.publish(Bool(self.outer))
		return np.zeros((self.resolution[0]*self.resolution[1]),dtype=int)

	def step(self, action):
		# Publish action
		if action==0:
			self.left_pub.publish(self.v_forward-self.v_turn)
			self.right_pub.publish(self.v_forward+self.v_turn)
			self.rate.sleep()
			####################
			self.motor_value = np.array([[1, 0, 0]])
			####################
		elif action==1:
			self.left_pub.publish(self.v_forward)
			self.right_pub.publish(self.v_forward)
			self.rate.sleep()
			####################
			self.motor_value = np.array([[0, 1, 0]])
			####################
		elif action==2:
			self.left_pub.publish(self.v_forward+self.v_turn)
			self.right_pub.publish(self.v_forward-self.v_turn)
			self.rate.sleep()
			####################
			self.motor_value = np.array([[0, 0, 1]])
			####################
		# Calculate robot position and distance
		d, p = self.getDistance(self.pos_data)
		# Calculate reward
		r = normpdf(d)
		# print(r)

		# # Get State from FCN8
		# s = fcn8_helper.save_inference_samples(self.vld_im, self.sess, self.image_shape, self.logits, self.keep_prob, self.image_input)
		# Get State
		s, s_plot = self.preprocess.getNewState()
		# self.preprocess.showradarplot_vld()

		# ###############Plot#####################
		# # self.state_plot.set_data(np.flipud(s.reshape((resolution[0], resolution[1])).T))
		# self.state_plot.set_data(s_plot[0:520].reshape((13, 40)))
		# self.motor_plot.set_data(self.motor_value)
		# self.motor_l_value_plot.set_text('%.2f' % self.motor_value[0][0])
		# self.motor_m_value_plot.set_text('%.2f' % self.motor_value[0][1])
		# self.motor_r_value_plot.set_text('%.2f' % self.motor_value[0][2])
		# self.fig.canvas.draw()
		# self.fig.canvas.flush_events()
		# ########################################

		# Check if distance causes reset
		if abs(d) > self.reset_distance:
			return s, r, True, d, p
		else:
			return s, r, False, d, p

	def getDistance(self,p): # Calculate robot position and distance
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

