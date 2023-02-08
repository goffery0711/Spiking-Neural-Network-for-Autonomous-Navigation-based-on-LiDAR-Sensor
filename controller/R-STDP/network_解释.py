#!/usr/bin/env python

import nest
import numpy as np
import pylab
import parameters as p

# 参考：https://blog.csdn.net/qq_39295220/article/details/108960005
# nest文档：https://nest-simulator.readthedocs.io/en/nest-2.20.1

class SpikingNeuralNetwork():
	def __init__(self):
		# NEST options
		np.set_printoptions(precision=1)
		nest.set_verbosity('M_WARNING')
		nest.ResetKernel()  # 重置内核
		nest.SetKernelStatus({"local_num_threads" : 1, "resolution" : p.time_resolution})  # 线程数设置为零；将分辨率设置为指定值0.1
		# Create Poisson neurons
		self.spike_generators = nest.Create("poisson_generator", p.resolution[0]*p.resolution[1], params=p.poisson_params) #泊松尖峰发生器8*4=32个
		self.neuron_pre = nest.Create("parrot_neuron", p.resolution[0]*p.resolution[1]) #"parrot_neuron"(鹦鹉神经元)重复传入尖峰的神经元；作用：为一组神经元提供相同的泊松尖峰序列
		# Create motor IAF neurons 
		self.neuron_post = nest.Create("iaf_psc_alpha", 2, params=p.iaf_params)    # "iaf_psc_alpha"具有α形突触后电流的整合并发射的神经元
		# Create Output spike detector  尖峰检测器 （如果神经元出现尖峰，它将向尖峰检测器发送事件）
		self.spike_detector = nest.Create("spike_detector", 2, params={"withtime": True})   #"withtime" True:记录触发事件的时间点
		# Create R-STDP synapses
		self.syn_dict = {"model": "stdp_dopamine_synapse",
						"weight": {"distribution": "uniform", "low": p.w0_min, "high": p.w0_max}}
		# 具有多巴胺调制的尖峰定时依赖可塑性的突触的连接；多巴胺神经元池发出的尖峰通过指定的体积发射器传递到突触。
		self.vt = nest.Create("volume_transmitter")
		nest.SetDefaults("stdp_dopamine_synapse", {"vt": self.vt[0], "tau_c": p.tau_c, "tau_n": p.tau_n, "Wmin": p.w_min, "Wmax": p.w_max, "A_plus": p.A_plus, "A_minus": p.A_minus}) 
		# https://nest-test.readthedocs.io/en/pynestapi_test/models/synapses/stdp_dopamine_synapse.html
		# 常见属性：    
		# SetDefaults(model, params): 将model的默认参数设置为params字典中指定的值。
		# vt long- volume_transmitter 的 ID，从多巴胺释放神经元池收集尖峰并将尖峰传输到突触。 值 -1 表示尚未分配体积传送器。
		# A_plus double - 促进重量变化的幅度
		# A_minus double - 抑郁症的体重变化幅度
		# tau_plus double - 促进的 STDP 时间常数（以毫秒为单位）
		# tau_c double - 资格跟踪的时间常数（以毫秒为单位）
		# tau_n double - 多巴胺能跟踪的时间常数，以毫秒为单位
		# b double - 多巴胺能基线浓度
		# Wmin double - 最小突触权重
		# Wmax double - 最大突触权重
		
		# 个人属性：
		# c double - 资格追踪
		# n double - 神经调节剂浓度 
		
		nest.Connect(self.spike_generators, self.neuron_pre, "one_to_one")
		nest.Connect(self.neuron_pre, self.neuron_post, "all_to_all", syn_spec=self.syn_dict)
		nest.Connect(self.neuron_post, self.spike_detector, "one_to_one")
		# Create connection handles for left and right motor neuron
		self.conn_l = nest.GetConnections(target=[self.neuron_post[0]])
		self.conn_r = nest.GetConnections(target=[self.neuron_post[1]])

	def simulate(self, dvs_data, reward):
		# Set reward signal for left and right network
		nest.SetStatus(self.conn_l, {"n": -reward*p.reward_factor})  # 神经调节剂浓度    n:二项式,binomial？？？
		nest.SetStatus(self.conn_r, {"n": reward*p.reward_factor})
		# Set poisson neuron firing time span
		time = nest.GetKernelStatus("time")  # 获取仿真内核的参数
		nest.SetStatus(self.spike_generators, {"origin": time})
		nest.SetStatus(self.spike_generators, {"stop": p.sim_time}) # 在特定时间段开始和停止的尖峰发生器
		# Set poisson neuron firing frequency
		dvs_data = dvs_data.reshape(dvs_data.size) 
		for i in range(dvs_data.size):   # .size: 8*4=32
			rate = dvs_data[i]/p.max_spikes
			rate = np.clip(rate,0,1)*p.max_poisson_freq  # np.clip(a_min, a_max):将数组中的元素限制在a_min, a_max之间
			nest.SetStatus([self.spike_generators[i]], {"rate": rate})  # 设置尖峰发生器的速率
		# Simulate network
		nest.Simulate(p.sim_time)   # 仿真内核模拟运行的时间(单位ms)
		# Get left and right output spikes 
		n_l = nest.GetStatus(self.spike_detector,keys="n_events")[0]  # 获取事件发生时的数据
		n_r = nest.GetStatus(self.spike_detector,keys="n_events")[1]
		# Reset output spike detector
		nest.SetStatus(self.spike_detector, {"n_events": 0})
		# Get network weights
		weights_l = np.array(nest.GetStatus(self.conn_l, keys="weight")).reshape(p.resolution)
		weights_r = np.array(nest.GetStatus(self.conn_r, keys="weight")).reshape(p.resolution)
		return n_l, n_r, weights_l, weights_r

	def set_weights(self, weights_l, weights_r):
		# Translate weights into dictionary format
		w_l = []
		for w in weights_l.reshape(weights_l.size):
			w_l.append({'weight': w})
		w_r = []
		for w in weights_r.reshape(weights_r.size):
			w_r.append({'weight': w})
		# Set left and right network weights
		nest.SetStatus(self.conn_l, w_l)
		nest.SetStatus(self.conn_r, w_r)
		return
