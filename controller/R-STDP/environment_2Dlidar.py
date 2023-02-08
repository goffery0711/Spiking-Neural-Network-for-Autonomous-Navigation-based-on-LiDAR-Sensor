#!/usr/bin/env python

import sys
sys.path.append('/usr/lib/python2.7/dist-packages') # weil ROS nicht mit Anaconda installiert
import rospy

import math
import time
import numpy as np
import matplotlib.pyplot as plt
import sim
from collections import deque
# import pcl
# from pcl import PointCloud
import open3d as o3d
# import re
import pandas as pd


from std_msgs.msg import Int8MultiArray, Float32, Bool, Float32MultiArray
from sensor_msgs.msg import PointCloud
from geometry_msgs.msg import Transform

from parameters import *


class VrepEnvironment():
	def __init__(self):
		self.vld_sub = rospy.Subscriber('velodyneData', PointCloud, self.vld_callback)
		# self.vld_sub = rospy.Subscriber('vldData', Float32MultiArray, self.vld_callback)
		self.dvs_sub = rospy.Subscriber('dvsData', Float32MultiArray, self.dvs_callback)
		self.pos_sub = rospy.Subscriber('transformData', Transform, self.pos_callback)
		self.left_pub = rospy.Publisher('leftMotorSpeed', Float32, queue_size=1)
		self.right_pub = rospy.Publisher('rightMotorSpeed', Float32, queue_size=1)
		self.reset_pub = rospy.Publisher('resetRobot', Bool, queue_size=None)
		self.vld_queue = 4   # collect 4 times to get the whole data
		self.fifo = deque(np.zeros((self.vld_queue, )))
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
		self.i = 1

	def vld_callback(self, msg):
		# PCL data
		self.x_data = []
		self.y_data = []
		self.z_data = []
		path = "/Thesis/Training-NN/Controller/R-STDP/test.pcd"

		o3d.io.write_point_cloud(path, msg)

		pcd = o3d.io.read_point_cloud(path)
		value = 16
		pcd_new = o3d.geometry.PointCloud.uniform_down_sample(pcd, value)
		cloud = pcl.PointCloud()

		# pointcloud = np.array(msg, dtype=np.float32)
		# cloud.from_array(pointcloud)
		# pcl.save(cloud, "cloud.pcd", format='pcd')


		self.vld_data = pcd_new.points
		points = np.asarray(self.vld_data)


		# self.vld_data = msg.points
		# points = np.asarray(self.vld_data)


		for point in points:
			self.x_data.append(point.x)
			self.y_data.append(point.y)
			self.z_data.append(point.z)

		if self.i == 1:
			df_x = pd.DataFrame(self.x_data)
			df_x.to_excel('x_data.xlsx')
			df_y = pd.DataFrame(self.y_data)
			df_y.to_excel('y_data.xlsx')
			df_z = pd.DataFrame(self.z_data)
			df_z.to_excel('z_data.xlsx')
			self.i = self.i+1
		# plt.scatter(range(len(self.z_data)), self.z_data)
		# plt.show()
		# print(len(self.x_data))   # <class 'geometry_msgs.msg._Point32.Point32'>
		return

	def dvs_callback(self, msg):
		# Store incoming DVS data
		self.dvs_data = msg.data
		return

	def pos_callback(self, msg):
		# Store incoming position data
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
		return np.zeros((resolution[0],resolution[1]),dtype=int), 0.

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
		return np.zeros((resolution[0],resolution[1]),dtype=int), 0.

	def step(self, n_l, n_r, episode):

		self.steps += 1
		t = False # terminal state
		
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
		# Set reward signal
		if self.outer == (d > 0):
			r = abs(d)
		else:
			r = -abs(d)

		self.distance = d
		s = self.getState()
		# self.showradarplot()
		n = self.steps
		lane = self.outer

		# Terminate episode of robot reaches start position again
		# or reset distance
		if abs(d) > reset_distance or n >= max_step:   # or p < 0.49
			# if p < 0.49:
			if self.outer:
				p = self.d_outer
			else:
				p = self.d_inner
			self.steps = 0
			t = True
			if episode % 30 == 0:
				self.allreset()
				time.sleep(8)
			else:
				self.reset()

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

	def getNewState(self):
		# self.x_data  # find more times data
		# self.y_data
		# self.z_data
		pass


	def showradarplot(self):
		plt.clf()
		plt.ion()
		new_state = np.zeros((resolution[0], resolution[1]), dtype=int)  # (8,4)
		for i in range(len(self.dvs_data) // 2):  # len(self.dvs_data)//2=128
			if self.dvs_data[i * 2] >= 0:
				state_y = 8 * self.dvs_data[i * 2]
				state_x = 4 * (self.dvs_data[i * 2 + 1] + 5)
				if 0 <= state_y <= 8 * 5 // 2:  # 24 <= y < (128-40)
					idx = (int(state_x // 5), int(state_y // 5))
					# idx = (x//16, (y-24)//16)
					plt.scatter(int(state_x // 5), int(state_y // 5))
		plt.xlim([0,8])
		plt.ylim([0,4])
		plt.pause(0.01)
		plt.clf()
		plt.ioff()
