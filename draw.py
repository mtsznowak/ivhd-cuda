import matplotlib.pyplot as plt
import numpy as np
import sys

x, y, labels = np.loadtxt(sys.argv[1], delimiter=' ', unpack=True)

# plt.scatter(x,y, c=labels)
# for i, txt in enumerate(labels):
#     if i % 50 == 0:
#         plt.annotate(str(txt), (x[i], y[i]))

unique = list(set(labels))
colors = [plt.cm.jet(float(i)/max(unique)) for i in unique]
for i, u in enumerate(unique):
    xi = [x[j] for j  in range(len(x)) if labels[j] == u]
    yi = [y[j] for j  in range(len(x)) if labels[j] == u]
    plt.scatter(xi, yi, c=colors[i], label=str(u))
plt.legend()

plt.show()
