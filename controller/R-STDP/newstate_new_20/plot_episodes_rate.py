import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

episode = np.load('episodes.npy')
# episode = np.array([ 1188, 370, 250, 275, 753, 669, 250, 915, 813, 1152, 10000, 1297, 7112, 7120, 10000, 10000, 10000, 10000, 10000, 10000])
# np.save('episodes.npy', episode)
num = np.arange(len(episode))

plt.plot(num, episode)

plt.xlabel("episode_num")
plt.ylabel("num")
plt.title('episode number')

ax = plt.gca()
x_major_locator = MultipleLocator(2)
ax.xaxis.set_major_locator(x_major_locator)

plt.savefig('episode_number_grid.svg', format='svg', dpi=600)
plt.show()
