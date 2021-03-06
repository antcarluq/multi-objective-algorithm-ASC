import numpy as numpy
import random
import math
import matplotlib.pyplot as plt
from tqdm import tqdm

from subproblem import Subproblem
from subproblem import Individual


# Metodo mejorado inicilizar una distribucion uniforme de vectores cuyas componenetes sumen la unidad
def initialize_subproblems(n):
    i = 1
    subproblems = []
    a = 1 / (n + 1)
    subproblems.append(Subproblem(a, 1 - a, None, []))
    while i < n:
        a = round(a + 1 / (n + 1), 10)
        subproblems.append(Subproblem(round(a, 10), round(1 - a, 10), None, []))
        i = i + 1

    return subproblems
#######################################################################################################


# Metodo para encontrar los T vectores vecinos mas cercanos
def calculate_neighbours(t, subproblems):
    for subproblem in subproblems:
        list_subproblem_dist = []
        for potential_neighbour in subproblems:
            # Calculo la distancia euclidea de los pares de vectores
            a = numpy.array((subproblem.x, subproblem.y))
            b = numpy.array((potential_neighbour.x, potential_neighbour.y))
            dist = numpy.linalg.norm(a - b)

            # Guardo en una matriz los potenciales vectores vecinos y su distancia al vector estudiado
            list_subproblem_dist.append(numpy.array((potential_neighbour, dist)))

            # Los ordenado de menor a mayor segun la distancia y me quedo con los t primeros
        list_subproblem_dist.sort(key=lambda tup: tup[1])
        matrix = numpy.array(list_subproblem_dist[0:t])

        # Guardo los vecinos mas cercanos en el vector estudiado
        setattr(subproblem, "neighbours", matrix[:, 0].tolist())

#######################################################################################################


# Metodo para generar la población inicial
def generate_population(subproblems, search_space, dimension, seed):
    population = []
    random.seed(seed)
    for subproblem in subproblems:
        gen = []
        for j in range(dimension):
            gen.append(random.uniform(search_space[0], search_space[1]))

        individual = Individual(gen, None)
        setattr(subproblem, "individual", individual)

        population.append(gen)
    return population


########################################################################################################


# Metodo para evaluar individualmente a un individuo
def evaluate_individual(individual, type):
    if type == 'zdt3':
        y = test_zdt3(individual)
    elif type == 'cf6':
        y = test_cf6(individual)
    else:
        raise Exception("The type of problem must be zdt3 or cf6")

    setattr(individual, "solution", [y[0], y[1]])
    return y


########################################################################################################


# Metodo para inicializar el punto de rerefencia con la poblacion inicial
def initialize_reference_point(subproblems, type):
    reference_point = []
    y0min = 10000000
    y1min = 10000000

    if type == 'zdt3':
        i = 0
        for subproblem in subproblems:
            y = test_zdt3(subproblem.individual)
            i = i + 1
            setattr(subproblem.individual, "solution", y)
            if y[0] < y0min:
                y0min = y[0]
            if y[1] < y1min:
                y1min = y[1]
    elif type == 'cf6':
        for subproblem in subproblems:
            y = test_cf6(subproblem.individual)
            setattr(subproblem.individual, "solution", y)
            if y[0] < y0min:
                y0min = y[0]
            if y[1] < y1min:
                y1min = y[1]

    reference_point.insert(0, y0min)
    reference_point.insert(1, y1min)
    return reference_point


########################################################################################################

# Formula de ZDT3
def test_zdt3(individual):
    gen = individual.gen
    n = len(gen)
    y = []
    f1 = gen[0]
    y.insert(0, f1)
    g = 1 + (9 / (n - 1)) * sum(gen[1:len(gen)])
    h = 1 - math.sqrt(f1 / g) - (f1 / g) * math.sin(10 * math.pi * f1)
    f2 = g * h
    y.insert(1, f2)

    return y


# Formula de CF6
def test_cf6(individual):
    gen = individual.gen
    n = len(gen)
    sum1 = 0
    sum2 = 0
    i = 2
    while i <= n:
        if i % 2 == 1:
            yi = gen[i - 1] - 0.8 * gen[0] * math.cos(6.0 * math.pi * gen[0] + i * math.pi / n)
            sum1 = sum1 + yi * yi
        else:
            yi = gen[i - 1] - 0.8 * gen[0] * math.sin(6.0 * math.pi * gen[0] + i * math.pi / n)
            sum2 = sum2 + yi * yi

        i = i + 1

    y = []
    y0 = gen[0] + sum1
    y1 = (1.0 - gen[0]) * (1.0 - gen[0]) + sum2

    y.insert(0, y0)
    y.insert(1, y1)
    # TODO en la formula final falta algo de restricciones que todavia no estoy implementando

    return y

########################################################################################################


# Operador evolutivo: Operadores mutación y cruce DE
def evolutive_operator(subproblem, probability, search_space):
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
        if random.choice(probability) == 0:
            aux = (neighbour_1.individual.gen[k] + f * (neighbour_2.individual.gen[k] - neighbour_3.individual.gen[k]))
            if aux < 0:
                aux = 0
            elif aux > 1:
                aux = 1
            gen.append(aux)
        elif random.choice(probability) == 1:
            gen.append(random.uniform(search_space[0], search_space[1]))
        else:
            gen.append(neighbour_1.individual.gen[k])
        k = k + 1

    individual = Individual(gen, None)
    return individual
########################################################################################################


# Metodo de seleccion de mejores soluciones
def selection_operator(neighbour, reference_point, individual):
    alpha_1 = neighbour.x
    alpha_2 = neighbour.y
    y_1 = individual.solution[0]
    y_2 = individual.solution[1]
    z_1 = reference_point[0]
    z_2 = reference_point[1]
    x_1 = neighbour.individual.solution[0]
    x_2 = neighbour.individual.solution[1]
    gte_x = max([alpha_1 * abs(x_1 - z_1), alpha_2 * abs(x_2 - z_2)])
    gte_y = max([alpha_1 * abs(y_1 - z_1), alpha_2 * abs(y_2 - z_2)])

    if gte_y <= gte_x:
        setattr(neighbour, "individual", individual)
########################################################################################################

# Metodo para guardar los datos
def save_data(subproblems, type, g, n , dimension):
    f = open(str(type) + "_D" + str(dimension) + "_P" + str(n)+ "_G" + str(g) + "_all_pop_seed" + str(seed) + ".out", "w")
    for subproblem in subproblems:
        f.write(str(subproblem.individual.solution[0]) + " " + str(subproblem.individual.solution[1]) + "\n")
    f.close()
########################################################################################################

# Metodo para visualizar los resultados
def visualization(subproblems, reference_point, type, g, n, dimension):
    if type == 'zdt3':
        pareto_front = numpy.genfromtxt('ZDT3_PF.dat')
    elif type == 'cf6':
        pareto_front = numpy.genfromtxt('CF6_PF.dat')
    else:
        raise Exception("The type of problem must be zdt3 or cf6")


    plt.plot(pareto_front[:, 0], pareto_front[:, 1], 'bo', markersize=4, color="black")
    plt.plot(reference_point[0], reference_point[1], 'bo')
    for subproblem in subproblems:
        plt.plot(subproblem.individual.solution[0], subproblem.individual.solution[1], 'go')
    plt.axis((-0.05, 1, -1, 5))
    plt.show()
##############################################################

# Algoritmo multiobjetivo basado en agregacion
def algorithm(g, n, t, search_space, dimension, type, seed, visual):
    # Apartado: Inicializacion
    subproblems = initialize_subproblems(n)
    calculate_neighbours(t, subproblems)
    generate_population(subproblems, search_space, dimension, seed)
    reference_point = initialize_reference_point(subproblems, type)

    #visualization(subproblems, reference_point, type)

    # Actualización por cada iteración
    k = 0  # Unicamente cuenta las evaluaciones totales
    for i in tqdm(range(g)):
        for subproblem in subproblems:
            # Reproduccion
            if i < g * 0.25:
                individual = evolutive_operator(subproblem, ([0, 1]), search_space)
            else:
                individual = evolutive_operator(subproblem, ([0, 2]), search_space)

            # Evaluacion
            solution = evaluate_individual(individual, type)
            k = k + 1
            # Actualizacion del punto de referencia

            if reference_point[0] > solution[0]:
                reference_point.pop(0)
                reference_point.insert(0, solution[0])
            if reference_point[1] > solution[1]:
                reference_point.pop(1)
                reference_point.insert(1, solution[1])

            # Actualizacion de vecinos: Por cada vecino del subproblema estudiado vemos si la solucion obtenida es mejor
            # que la existente
            for neighbour in subproblem.neighbours:
                selection_operator(neighbour, reference_point, individual)

    if visual == 1:
        visualization(subproblems, reference_point, type, g, n, dimension)
        save_data(subproblems, type, g, n, dimension)
        print("Numero de evaluaciones: " + str(k))
    return subproblems
########################################################################################################


# Ejecucion
########################################################################################################

input_type = int(input("Seleccione el problema a resolver (0 para zdt3) (1 para cf6): "))
if input_type == 0:
    type = "zdt3"
    dimension = 30
    search_space = [0, 1]
elif input_type == 1:
    type = "cf6"
    search_space = [-2, 2]
    input_dimension = int(input("Seleccione la dimension para cf6 (16 o 4): "))
    if input_dimension == 16:
        dimension = 16
    elif input_dimension == 4:
        dimension = 4
    else:
        raise Exception("La dimension para CF6 debe ser 16 o 4")
else:
    raise Exception("El tipo de problema debe ser zdt3 o cf6")

g = int(input("Seleccione el numero de generaciones: "))
n = int(input("Seleccione el numero de subproblemas: "))
t = int(input("Seleccione el numero de vecinos (recomendado el 30% del numero de subproblemas): "))

input_seed = int(input("Seleccione la semilla (semilla aleatoria = -1): "))

if input_seed == -1:
    seed = random.randint(0, 50)
elif input_seed > 0:
    seed = input_seed
else:
    raise Exception("Semilla incorrecta")

try:
    algorithm(g, n, t, search_space, dimension, type, seed, 1)
except ValueError:
    print("No se ha podido completar la ejecucion, comprueba que los parametros introducidos son correctos")