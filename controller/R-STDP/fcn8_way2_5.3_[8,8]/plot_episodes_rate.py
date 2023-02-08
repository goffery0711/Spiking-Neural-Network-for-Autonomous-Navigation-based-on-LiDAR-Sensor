import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

# episode = np.load('episodes.npy')
episode = np.array([266, 109, 64, 108, 235, 223, 265, 224, 273, 218, 268, 199, 287, 199, 700, 218, 848, 230, 787, 236, 767, 529, 762, 499, 1300, 1300, 1300, 1300, 1300, 1300, 1300])
np.save('episodes.npy', episode)
num = np.arange(len(episode))

plt.plot(num, episode)

plt.xlabel("episode_num")
plt.ylabel("num")
plt.title('episode number')

ax = plt.gca()
x_major_locator=MultipleLocator(2)
ax.xaxis.set_major_locator(x_major_locator)
plt.savefig('episode_number_way2.svg', format='svg', dpi=600)
plt.show()
