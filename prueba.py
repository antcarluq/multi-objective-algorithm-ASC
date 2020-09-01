import numpy as numpy
import random
import math
import matplotlib.pyplot as plt
from tqdm import tqdm

from main import *
from subproblem import Subproblem
from subproblem import Individual


g = 250
n = 40
t = 12
type = "zdt3"
dimension = 30
search_space = [0, 1]
seed = 30


subproblems = algorithm(g, n, t, search_space, dimension, type, seed)

nsgaii = numpy.genfromtxt('results/ZDT3/EVAL10000/P40G250/final_pop_seed1.out')
pareto_front = numpy.genfromtxt('ZDT3_PF.dat')
plt.plot(pareto_front[:, 0], pareto_front[:, 1], 'bo', markersize=4, color="black")
plt.plot(nsgaii[:, 0], nsgaii[:, 1], 'bo', markersize=4, color="red")
for subproblem in subproblems:
        plt.plot(subproblem.individual.solution[0], subproblem.individual.solution[1], 'go')
plt.axis((-0.05, 1, -1, 5))
plt.show(block=False)


f = open("results/MyAlgorithm/EVAL10000/P"+ str(n) +"G"+ str(g) +"/final_pop_seed"+ str(seed) +".out","w")
for subproblem in subproblems:
    f.write(subproblem.individual.solution[0] + " " + subproblem.individual.solution[1] + "\n")
f.close()
