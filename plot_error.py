import matplotlib.pyplot as plt
import numpy as np
import sys

x, y = np.loadtxt(sys.argv[1], delimiter=' ', unpack=True)

plt.scatter(x,y,s=3)
plt.show()
