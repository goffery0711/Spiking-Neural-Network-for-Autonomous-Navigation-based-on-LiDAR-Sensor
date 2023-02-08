import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

episode = np.load('episodes.npy')
# episode = np.array([283, 125, 60, 88, 220, 237, 325, 224, 307, 224, 299, 227, 287, 230, 308, 218, 965, 207, 1300, 1300, 1300, 1300, 1300, 1300, 1300, 1300])
# np.save('episodes.npy', episode)
num = np.arange(len(episode))

plt.plot(num, episode)

plt.xlabel("episode_num")
plt.ylabel("num")
plt.title('episode number')

ax = plt.gca()
x_major_locator=MultipleLocator(2)
ax.xaxis.set_major_locator(x_major_locator)

# path = './'   #path+
plt.savefig('episode_num_way1.svg', format='svg', dpi=600)
plt.show()
