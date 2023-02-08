#!/usr/bin/env python

import sys
sys.path.append('/usr/lib/python2.7/dist-packages') # weil ROS nicht mit Anaconda installiert

import rospy
import math
import time
import sim
import random
import numpy as np
from collections import deque
from std_msgs.msg import Int8MultiArray, Float32, Bool, Float32MultiArray
from geometry_msgs.msg import Transform
import matplotlib.pyplot as plt


def normpdf(x, mean=0, sd=0.15):
    var = float(sd)**2
    denom = (2*math.pi*var)**.5
    num = math.exp(-(float(x)-float(mean))**2/(2*var))
    return num/denom

class VrepEnvironment():
	def __init__(self,speed,turn,resolution,reset_distance, rate, dvs_queue, resize_factor, crop):
		self.dvs_sub = rospy.Subscriber('dvsData', Float32MultiArray, self.dvs_callback)
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


	def dvs_callback(self, msg):
		# FIFO queue storing DVS data during one step
		self.fifo.append(msg.data)
		self.fifo.popleft() 
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
		elif action==1:
			self.left_pub.publish(self.v_forward)
			self.right_pub.publish(self.v_forward)
			self.rate.sleep()
		elif action==2:
			self.left_pub.publish(self.v_forward+self.v_turn)
			self.right_pub.publish(self.v_forward-self.v_turn)
			self.rate.sleep()
		# Get transform data
		p = rospy.wait_for_message('transformData', Transform).translation
		p = np.array([p.x,p.y])
		# Calculate robot position and distance
		d, p = self.getDistance(p)
		# Calculate reward
		r = normpdf(d)
		# Translate DVS data from FIFO queue into state image
		s = self.getState()
		# self.showradarplot()
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

	def getState(self): # Translate DVS data from FIFO queue into state image
		# Initialize array in new size
		new_state = np.zeros(40*20,dtype=int)
		state = list(self.fifo)[-1]
		# for state in list(self.fifo): # For every DVS frame in FIFO queue
		for i in range(len(state)//2): # For every event in DVS frame
			if state[i*2] != -99 and state[i*2] >= 0:
				state_y = 16*state[i*2]          # [0,5] --> [0,80]  --  [0,20]
				state_x = 16*(state[i*2+1]+5)	 # [-5,5] --> [0,160]  --  [0,40]
				idx = (state_y//self.resize_factor)*40 + state_x//self.resize_factor   # resize_factor = 4
				idx = int(idx)
				if idx >= 800:
					idx = 799
				new_state[idx] += 1
		# print(new_state[0:512])
		return new_state[0:512]  #Here only first part  # Original DVS(middle): (6*32 : -10*32)=(6*32 : 22*32)

	def showradarplot(self):
		plt.clf()
		plt.ion()
		s = list(self.fifo)[-1]
		# for s in list(self.fifo): # For every DVS frame in FIFO queue
		for i in range(len(s)//2): # For every event in DVS frame
			if s[i*2] != -99 and s[i*2] >= 0:
				# x = 10*(s[i*2] + 1)
				# y = 25*(s[i*2+1]+2)
				# plt.scatter(x, y)
				plt.scatter(s[i*2+1], s[i*2])
		plt.xlim([-2.5, 2.5])  # just keep the same as ylim
		# plt.xlim([-5,5])
		plt.ylim([0,5])
		plt.pause(0.01)
		plt.clf()
		plt.ioff()
