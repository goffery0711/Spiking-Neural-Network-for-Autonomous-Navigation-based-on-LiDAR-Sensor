#!/usr/bin/env python

import sys
sys.path.append('/usr/lib/python2.7/dist-packages') # weil ROS nicht mit Anaconda installiert
import rospy
import math
import time
import numpy as np
from std_msgs.msg import Int8MultiArray, Float32, Bool, Float32MultiArray
from geometry_msgs.msg import Transform
from parameters import *

class VrepEnvironment():
	def __init__(self):
		self.dvs_sub = rospy.Subscriber('dvsData', Float32MultiArray, self.dvs_callback)
		self.pos_sub = rospy.Subscriber('transformData', Transform, self.pos_callback)
		self.left_pub = rospy.Publisher('leftMotorSpeed', Float32, queue_size=1)
		self.right_pub = rospy.Publisher('rightMotorSpeed', Float32, queue_size=1)
		self.reset_pub = rospy.Publisher('resetRobot', Bool, queue_size=None)
		self.dvs_data = np.array([0,0])
		self.pos_data = []
		self.v_pre = v_pre
		self.turn_pre = turn_pre
		self.resize_factor = [dvs_resolution[0]//resolution[0], (dvs_resolution[1]-crop_bottom-crop_top)//resolution[1]]
		self.outer = False
		rospy.init_node('dvs_controller')
		self.rate = rospy.Rate(rate)
		
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

	def dvs_callback(self, msg):
		# Store incoming DVS data
		self.dvs_data = msg.data
		return

	def pos_callback(self, msg):
		# Store incoming robot position data
		self.pos_data = np.array([msg.translation.x, msg.translation.y, time.time()])
		return

	def reset(self):
		# Reset robot and change lane
		self.outer = not self.outer
		self.reset_pub.publish(Bool(self.outer))
		return np.zeros((resolution[0],resolution[1]),dtype=int)

	def step(self, n_l, n_r):
		# Steering wheel model
		m_l = n_l/n_max
		m_r = n_r/n_max
		print m_l, m_r
		a = m_l - m_r
		v_cur = - abs(a)*(v_max - v_min) + v_max
		turn_cur = turn_factor * a
		c = math.sqrt((m_l**2 + m_r**2)/2.0)
		self.v_pre = c*v_cur + (1-c)*self.v_pre
		self.turn_pre = c*turn_cur + (1-c)*self.turn_pre
		
		# Publish motor speeds
		self.left_pub.publish(self.v_pre+self.turn_pre)
		self.right_pub.publish(self.v_pre-self.turn_pre)
		self.rate.sleep()
		
		# Return distance, position, state
		d, p = self.getDistance(self.pos_data)
		s = self.getState()
		if abs(d) > reset_distance:
			self.reset()
		return s,d,p

	def getDistance(self,p): # Calculate robot position and distance
		# 180 turn for x < 2.5
		if p[0] < self.scale*self.v1:
			r = np.linalg.norm(p[:2]-self.c1)
			delta_y = p[1] - self.c1[1]
			if self.outer:
				a = abs(math.acos(delta_y / r)/math.pi)
				position = self.d1_outer + self.d2_outer + self.d3_outer + self.d4_outer + self.d5_outer + a * self.d6_outer
				distance = r - self.r1_outer
				return distance, position
			else:
				a = 1. - abs(math.acos(delta_y / r)/math.pi)
				position = self.d1_inner + self.d2_inner + self.d3_inner + a * self.d4_inner
				distance = r - self.r1_inner
				return distance, position
		# 180 turn for y > 7.5
		elif p[1] > self.scale*self.v2:
			r = np.linalg.norm(p[:2]-self.c2)
			delta_x = p[0] - self.c2[0]
			if self.outer:
				a = abs(math.acos(delta_x / r)/math.pi)
				position = self.d1_outer + self.d2_outer + self.d3_outer + a * self.d4_outer
				distance = r - self.r1_outer
				return distance, position
			else:
				a = 1. - abs(math.acos(delta_x / r)/math.pi)
				position = self.d1_inner + self.d2_inner + self.d3_inner + self.d4_inner + self.d5_inner + a * self.d6_inner
				distance = r - self.r1_inner
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
					distance = r - self.r1_outer
					return distance, position
				else:
					a = 1. - abs(math.asin(delta_x / r)/(0.5*math.pi))
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
		# 2.5 < x < 7.5 and y < 7.5
		else:
			# straight for y < 2.5
			if p[1] < self.scale*self.v1:
				if self.outer:
					distance = (p[1] - self.l2_outer)*(-1)
					position = abs(p[0] - self.v1)
					return distance, position
				else:
					distance = (p[1] - self.l2_inner)*(-1)
					position = self.d1_inner + self.d2_inner + abs(p[0] - self.v2)
					return distance, position
			# 90 turn for x,y between 2.5 and 7.5
			else:
				r = np.linalg.norm(p[:2]-self.c4)
				delta_y = p[1] - self.c4[1]
				if self.outer:
					a = abs(math.asin(delta_y / r)/(0.5*math.pi))
					position = self.d1_outer + self.d2_outer + self.d3_outer + self.d4_outer + a * self.d5_outer
					distance = (r - self.r2_outer)*(-1)
					return distance, position
				else:
					a = 1. - abs(math.asin(delta_y / r)/(0.5*math.pi))
					position = self.d1_inner + self.d2_inner + self.d3_inner + self.d4_inner + a * self.d5_inner
					distance = (r - self.r2_inner)*(-1)
					return distance, position

	def getState(self): # Translate DVS data from FIFO queue into state image
		new_state = np.zeros((resolution[0], resolution[1]), dtype=int)  # (8,4)
		for i in range(len(self.dvs_data) // 2):  # len(self.dvs_data)//2=128
			if self.dvs_data[i * 2] >= 0:
				state_y = 8 * self.dvs_data[i * 2]
				state_x = 4 * (self.dvs_data[i * 2 + 1] + 5)
				try:
					if 0 <= state_y <= 8 * 5 // 2:  # 24 <= y < (128-40)
						idx = (int(state_x // 5), int(state_y // 5))
					# idx = (x//16, (y-24)//16)
					new_state[idx] += 1
				except:
					pass
		return new_state
