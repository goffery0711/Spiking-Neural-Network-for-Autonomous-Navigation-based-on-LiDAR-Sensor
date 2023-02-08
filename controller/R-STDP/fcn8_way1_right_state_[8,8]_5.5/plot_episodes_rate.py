import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

episode = np.load('episodes.npy')

num = np.arange(len(episode))

plt.plot(num, episode)

plt.xlabel("episode_num")
plt.ylabel("num")
plt.title('episode number')

ax = plt.gca()
x_major_locator=MultipleLocator(2)
ax.xaxis.set_major_locator(x_major_locator)

plt.savefig('episode_number_right.svg', format='svg', dpi=600)
plt.show()
