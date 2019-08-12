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
    plt.scatter(xi, yi, c=colors[i], label=str(u), s=2)
plt.legend()



# auto annotations
sums = {}
freqs = {}

for i in range(0, len(x)):
    if labels[i] in freqs:
        freqs[labels[i]] = freqs[labels[i]] + 1
    else:
        freqs[labels[i]] = 1

    if (labels[i] in sums):
        (ax, ay) = sums[labels[i]]
        sums[labels[i]] = (x[i] + ax, y[i] + ay)
    else:
        sums[labels[i]] = (x[i], y[i])


for i in range(0, len(unique)):
    (avx, avy) = sums[unique[i]]
    avx = avx / freqs[unique[i]]
    avy = avy / freqs[unique[i]]

    plt.annotate(unique[i],
            [avx, avy],
            horizontalalignment='center',
            verticalalignment='center',
            size=15, weight='bold',
            color='white',
            backgroundcolor=colors[i])

plt.show()
