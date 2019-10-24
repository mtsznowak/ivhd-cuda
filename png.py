import matplotlib.pyplot as plt
import random
import numpy as np
import sys
from sklearn.neighbors import KDTree
import glob

K_METRIC = 10

def loadPositions(exp_name):
    allPositions = glob.glob(exp_name + "_*_positions")
    allPositions = [pos.split("_")[1] for pos in allPositions]
    allPositions.sort(key=int)
    allPositions = [ exp_name + "_" + str(middle) + "_positions" for middle in allPositions]

    i = 0
    colors = None

    for positionFile in allPositions:
        print(positionFile)

        X, Y, L = np.loadtxt(positionFile, delimiter=' ', unpack=True)

        if colors == None:
            unique = list(set(L))
            print(unique)
            colors = list(range(0, int(max(unique))+3))
            random.seed(212)
            random.shuffle(colors)

# colors = [plt.cm.gist_rainbow(float(i)/max(unique)) for i in unique]

            colors = [plt.cm.Set1(float(colors[int(cl)])/len(unique)) for cl in L]

        plt.scatter(X, Y, c=colors, label=L, s=0.3)

        plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0)
        plt.axis('off')
        zerostoadd = 5 - len(str(i))
        plt.savefig("0"*zerostoadd + str(i) + ".png")
        plt.clf()
        i = i + 1

exp_name = sys.argv[1]
loadPositions(exp_name)


