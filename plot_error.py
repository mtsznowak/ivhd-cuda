import matplotlib.pyplot as plt
import numpy as np
import sys

i = 1
while i < len(sys.argv):
    x, y = np.loadtxt(sys.argv[i], delimiter=' ', unpack=True)

    scaling = False
    if scaling:
        for j in range(1, len(y)):
            y[j] = y[j] / y[0]
        y[0] = 1.0;

    plt.scatter(x,y,s=3,label=sys.argv[i])
    i = i + 1

plt.legend()
plt.show()
