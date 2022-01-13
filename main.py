# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import matplotlib.pyplot as plt
import itertools
from GA_skeleton import *
from calcTSP import *
def print_hi(name):
    n = 150
    depots = {}
    graph = np.zeros((n,n))
    with open ('tokyo.dat', 'r') as tab :
        for i in range (1,151) :
            line = tab.readline().split()
            depots[i] = (float(line[1]), float(line[2]))
        for i in range (1,151) :
            for j in range (1,151) :
                if j != i :
                    graph[i-1][j-1] = np.sqrt((depots[i][0] - depots[j][0]) ** 2 + (depots[i][1] - depots[j][1]) ** 2)

    tourStat = []
    NTrials = 10 ** 8
    xmin,fmin,history = GA(n,NTrials, computeTourLength, graph, 151)

    # for k in range(NTrials):
    #     tourStat.append(computeTourLength(np.random.permutation(n), gr aph))
    plt.hist(history, bins=50)
    print('best TSP res', fmin)
    plt.show()
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
