import numpy as numpy
import random
import math
import matplotlib.pyplot as plt
from tqdm import tqdm

from subproblem import Subproblem
from subproblem import Individual


# Metodo inicilizar una distribucion uniforme de vectores cuyas componenetes sumen la unidad
def initialize_subproblems_old_version(n):
    i = 0
    subproblems = []
    a = 0.5
    b = 0.5
    if n % 2 == 1:
        i = 1
        subproblems.append(Subproblem(a, b, None, []))
    while i < n:
        aux = 1 / n
        if i % 2 == 0:
            a = a + aux
            subproblems.append(Subproblem(round(a, 2), round(1 - a, 2), None, []))
        else:
            b = b + aux
            subproblems.append(Subproblem(round(1 - b, 2), round(b, 2), None, []))
        i = i + 1

    return subproblems

# Operador evolutivo: La media entre dos individuos
def operador_evolutivo_old(neighbours, search_space):
    list_aux = list(range(len(neighbours)))
    i = random.choice(list_aux)
    list_aux.remove(i)
    j = random.choice(list_aux)
    neighbour_1 = neighbours[i]
    neighbour_2 = neighbours[j]
    gen = []
    k = 0
    while k < len(neighbour_1.individual.gen):
        if random.choice(([0, 1, 2, 3, 4, 5])) == 0:
            gen.append((neighbour_1.individual.gen[k] + neighbour_2.individual.gen[k]) / 2)
        else:
            gen.append(random.uniform(search_space[0], search_space[1]))
        k = k + 1

    individual = Individual(gen, None)
    return individual

# Operador evolutivo: Operadores mutación y cruce DE
def evolutive_operator_2(subproblem, search_space):
    neighbours = subproblem.neighbours
    f = 0.5
    list_aux = list(range(len(neighbours)))
    i = random.choice(list_aux)
    list_aux.remove(i)
    j = random.choice(list_aux)
    neighbour_1 = subproblem
    neighbour_2 = neighbours[i]
    neighbour_3 = neighbours[j]
    gen = []
    k = 0
    while k < len(neighbour_1.individual.gen):
        if random.choice(([1, 0])) == 0:
            aux = (neighbour_1.individual.gen[k] + f * (neighbour_2.individual.gen[k] - neighbour_3.individual.gen[k]))
            if aux < 0:
                aux = 0
            elif aux > 1:
                aux = 1
            gen.append(aux)
        else:
            gen.append(random.uniform(search_space[0], search_space[1]))
        k = k + 1

    individual = Individual(gen, None)
    return individual
########################################################################################################

# Operador evolutivo: Operadores mutación y cruce DE
def evolutive_operator_3(subproblem):
    neighbours = subproblem.neighbours
    f = 0.5
    list_aux = list(range(len(neighbours)))
    i = random.choice(list_aux)
    list_aux.remove(i)
    j = random.choice(list_aux)
    neighbour_1 = subproblem
    neighbour_2 = neighbours[i]
    neighbour_3 = neighbours[j]
    gen = []
    k = 0
    while k < len(neighbour_1.individual.gen):
        gen.append((neighbour_1.individual.gen[k] + neighbour_2.individual.gen[k]) / 2)
        k = k + 1

    individual = Individual(gen, None)
    return individual
########################################################################################################

# Operador evolutivo: Operadores mutación y cruce DE
def evolutive_operator_4(subproblem, search_space):
    neighbours = subproblem.neighbours
    f = 0.5
    list_aux = list(range(len(neighbours)))
    i = random.choice(list_aux)
    list_aux.remove(i)
    j = random.choice(list_aux)
    neighbour_1 = subproblem
    neighbour_2 = neighbours[i]
    neighbour_3 = neighbours[j]
    gen = []
    k = 0
    while k < len(neighbour_1.individual.gen):
        if random.choice(([0, 1])) == 0:
            gen.append(random.uniform(search_space[0], search_space[1]))
        else:
            gen.append(neighbours[0].individual.gen[k])
        k = k + 1

    individual = Individual(gen, None)
    return individual
################################################################

# Operador evolutivo: Mutacion uniforme
def operador_evolutivo_2(neighbours, search_space):
    gen = []
    k = 0
    while k < len(neighbours[0].individual.gen):
        if random.choice(([0, 1])) == 0:
            gen.append(random.uniform(search_space[0], search_space[1]))
        else:
            gen.append(neighbours[0].individual.gen[k])
        k = k + 1

    individual = Individual(gen, None)
    return individual

def selection_operator_old(neighbour, solution, reference_point):
    best_solution_point = numpy.array((neighbour.individual.solution[0], neighbour.individual.solution[1]))
    solution_point = numpy.array((solution[0], solution[1]))

    rp_point = numpy.array((reference_point[0], reference_point[1]))
    dist_best_solution_to_rp = numpy.linalg.norm(best_solution_point - rp_point)
    dist_solution_to_rp = numpy.linalg.norm(solution_point - rp_point)

    subproblem_point = numpy.array((neighbour.x, neighbour.y))
    dist_best_solution_to_subproblem = numpy.linalg.norm(best_solution_point - subproblem_point)
    dist_solution_to_subproblem = numpy.linalg.norm(solution_point - subproblem_point)

    if dist_solution_to_rp < dist_best_solution_to_rp:
        setattr(neighbour.individual, "solution", [solution[0], solution[1]])
    elif dist_solution_to_subproblem < dist_best_solution_to_subproblem:
        setattr(neighbour.individual, "solution", [solution[0], solution[1]])

def visualization2(subproblems, reference_point, type, solutions):
    if type == 'zdt3':
        pareto_front = numpy.genfromtxt('ZDT3_PF.dat')
    elif type == 'cf6':
        pareto_front = numpy.genfromtxt('CF6_PF.dat')
    else:
        raise Exception("The type of problem must be zdt3 or cf6")
    for solution in solutions:
        plt.plot(solution[0], solution[1], 'ro')

    plt.plot(pareto_front[:, 0], pareto_front[:, 1], 'bo', markersize=4, color="black")
    plt.axis((-0.05, 1, -1, 5))
    plt.show(block=False)