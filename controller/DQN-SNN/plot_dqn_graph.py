#!/usr/bin/env python

import numpy as np
import math
import matplotlib.pyplot as plt

# DQN training progress figure (Fig. 5.1)
# CSV files downloaded from tensorboard


data1 = np.load('episodes.npy')
data2 = np.load('reward.npy')
x1 = np.arange(len(data1))
y1 = data1
x2 = np.arange(len(data2))
y2 = data2

fig = plt.figure(figsize=(7,4))

ax1 = plt.subplot(211)
ax1.set_ylabel('Time Steps')
plt.grid(linestyle=':')
ax1.set_xlim((0,max(x1)))
ax1.set_ylim([0,1200])
plt.plot(x1,y1,linewidth=1.0)

ax2 = plt.subplot(212, sharex=ax1)
ax2.set_xlabel('Episode')
ax2.set_ylabel('Reward')
ax2.set_xlim((0,max(x1)))
ax2.set_ylim([0,2900])
plt.grid(linestyle=':')
plt.plot(x2,y2,linewidth=1.0)

fig.tight_layout()
plt.savefig('dqn_graph.svg', format='svg', dpi=600)
plt.show()
