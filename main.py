import numpy as numpy
import random
import math
import matplotlib.pyplot as plt
from tqdm import tqdm

from subproblem import Subproblem
from subproblem import Individuo



# Metodo inicilizar una distribucion uniforme de vectores cuyas componenetes sumen la unidad
def initialize_subproblems_old_version(n):
    i = 0
    subproblems = []
    a = 0.5
    b = 0.5
    if n % 2 == 1:
        i = 1
        subproblems.append(Subproblem(a, b, []))
    while i < n:
        aux = 1 / n
        if i % 2 == 0:
            a = a + aux
            subproblems.append(Subproblem(round(a, 2), round(1 - a, 2), []))
        else:
            b = b + aux
            subproblems.append(Subproblem(round(1 - b, 2), round(b, 2), []))
        i = i + 1

    return subproblems


#######################################################################################################


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
def calcular_vecinos(t, subproblems):
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
def generar_poblacion(subproblems, search_space, dimension):
    poblacion = []
    random.seed(30)
    for subproblem in subproblems:
        gen = []
        for j in range(dimension):
            gen.append(random.uniform(search_space[0], search_space[1]))

        individuo = Individuo(gen, None)
        setattr(subproblem, "individuo", individuo)

        poblacion.append(gen)
    return poblacion


########################################################################################################


# Metodo para evaluar individualmente a un individuo
def evaluar_individuo(individuo, type):
    if type == 'zdt3':
        y = test_zdt3(individuo)
    elif type == 'cf6':
        y = test_cf6(individuo)
    return y


########################################################################################################


# Metodo para inicializar el punto de rerefencia con la poblacion inicial
def initialize_reference_point(subproblems, type):
    reference_point = []
    y0min = 10000000
    y1min = 10000000

    if type == 'zdt3':
        for subproblem in subproblems:
            y = test_zdt3(subproblem.individuo)
            setattr(subproblem.individuo, "solution", y)
            if y[0] < y0min:
                y0min = y[0]
            if y[1] < y1min:
                y1min = y[1]
    elif type == 'cf6':
        for subproblem in subproblems:
            y = test_cf6(subproblem.individuo)
            setattr(subproblem.individuo, "solution", y)
            if y[0] < y0min:
                y0min = y[0]
            if y[1] < y1min:
                y1min = y[1]

    reference_point.insert(0, y0min)
    reference_point.insert(1, y1min)
    return reference_point


########################################################################################################


# Formula de ZDT3
def test_zdt3(individuo):
    sum = 0
    gen = individuo.gen
    n = len(gen)
    i = 1
    while i < n:
        sum = sum + gen[i]
        i = i + 1

    y = []
    y.insert(0, gen[0])
    g = 1 + ((9 * sum) / (n - 1))
    h = 1 - math.sqrt(gen[0] / g) - (gen[0] / g) * math.sin(10 * math.pi * gen[0])
    y.insert(1, (g * h))

    return y

# Formula de CF6
def test_cf6(individuo):
    gen = individuo.gen
    n = len(gen)
    sum1 = 0
    sum2 = 0
    i = 2
    while i <= n:
        if i % 2 == 1:
            yi = gen[i - 1] - 0.8 * gen[0] * math.cos(6.0 * math.pi * gen[0] + i * math.pi/n)
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


# Operador evolutivo: La media entre dos individuos
def operador_evolutivo_old(neighbours):
    list_aux = list(range(len(neighbours)))
    i = random.choice(list_aux)
    list_aux.remove(i)
    j = random.choice(list_aux)
    neighbour_1 = neighbours[i]
    neighbour_2 = neighbours[j]
    gen = []
    k = 0
    while k < len(neighbour_1.individuo.gen):
        if random.choice(([0, 1, 2, 3, 4, 5])) == 0:
            gen.append((neighbour_1.individuo.gen[k] + neighbour_2.individuo.gen[k]) / 2)
        else:
            gen.append(random.uniform(search_space[0], search_space[1]))
        k = k + 1

    individuo = Individuo(gen, None)
    return individuo

# Operador evolutivo: Operadores mutación y cruce DE
def operador_evolutivo(neighbours):
    f = 0.5 # TODO Poner como variable
    list_aux = list(range(len(neighbours)))
    i = random.choice(list_aux)
    list_aux.remove(i)
    j = random.choice(list_aux)
    list_aux.remove(j)
    z = random.choice(list_aux)
    neighbour_1 = neighbours[i]
    neighbour_2 = neighbours[j]
    neighbour_3 = neighbours[z]
    gen = []
    k = 0
    while k < len(neighbour_1.individuo.gen):
        if random.choice(([0])) == 0:
            aux = (neighbour_1.individuo.gen[k] + f * (neighbour_2.individuo.gen[k] - neighbour_3.individuo.gen[k]))
            if aux < 0:
                aux = 0
            elif aux > 1:
                aux = 1
            gen.append(aux)
        else:
            gen.append(random.uniform(search_space[0], search_space[1]))
        k = k + 1

    individuo = Individuo(gen, None)
    return individuo
########################################################################################################

# Operador evolutivo: Mutacion uniforme
def operador_evolutivo_2(neighbours):
    gen = []
    k = 0
    while k < len(neighbours[0].individuo.gen):
        if random.choice(([0, 1])) == 0:
            gen.append(random.uniform(search_space[0], search_space[1]))
        else:
            gen.append(neighbours[0].individuo.gen[k])
        k = k + 1

    individuo = Individuo(gen, None)
    return individuo
########################################################################################################


# Algoritmo multiobjetivo basado en agregacion
def operador_seleccion(neighbour, solution, reference_point):
    alpha_1 = neighbour.x
    alpha_2 = neighbour.y
    y_1 = solution[0]
    y_2 = solution[1]
    z_1 = reference_point[0]
    z_2 = reference_point[1]
    x_1 = neighbour.individuo.solution[0]
    x_2 = neighbour.individuo.solution[1]
    gte_x = max([alpha_1 * abs(x_1 - z_1), alpha_2 * abs(x_2 - z_2)])
    gte_y = max([alpha_1 * abs(y_1 - z_1), alpha_2 * abs(y_2 - z_2)])

    if gte_y <= gte_x:
        setattr(neighbour.individuo, "solution", [y_1, y_2])


def operador_seleccion_old(neighbour, solution, reference_point):
    best_solution_point = numpy.array((neighbour.individuo.solution[0], neighbour.individuo.solution[1]))
    solution_point = numpy.array((solution[0], solution[1]))

    rp_point = numpy.array((reference_point[0], reference_point[1]))
    dist_best_solution_to_rp = numpy.linalg.norm(best_solution_point - rp_point)
    dist_solution_to_rp = numpy.linalg.norm(solution_point - rp_point)

    subproblem_point = numpy.array((neighbour.x, neighbour.y))
    dist_best_solution_to_subproblem = numpy.linalg.norm(best_solution_point - subproblem_point)
    dist_solution_to_subproblem = numpy.linalg.norm(solution_point - subproblem_point)

    if dist_solution_to_rp < dist_best_solution_to_rp:
        setattr(neighbour.individuo, "solution", [solution[0], solution[1]])
    elif dist_solution_to_subproblem < dist_best_solution_to_subproblem:
        setattr(neighbour.individuo, "solution", [solution[0], solution[1]])


def visualization(subproblems, reference_point, type):
    if type == 'zdt3':
        pareto_front = numpy.genfromtxt('ZDT3_PF.dat')
    elif type == 'cf6':
        pareto_front = numpy.genfromtxt('CF6_PF.dat')

    plt.plot(pareto_front[:, 0], pareto_front[:, 1], 'bo', markersize=4, color="black")
    plt.plot(reference_point[0], reference_point[1], 'bo')
    for subproblem in subproblems:
        plt.plot(subproblem.individuo.solution[0], subproblem.individuo.solution[1], 'go')
    plt.axis((-0.05, 1, -1, 5))
    plt.show(block=False)


def algorithm(g, n, t, search_space, dimension, type):
    # Apartado: Inicializacion
    subproblems = initialize_subproblems(n)
    calcular_vecinos(t, subproblems)
    generar_poblacion(subproblems, search_space, dimension)
    reference_point = initialize_reference_point(subproblems, type)

    visualization(subproblems, reference_point, type)

    # Actualización por cada iteración
    z = 0 # Unicamente cuenta las evaluaciones totales
    i = 0
    for i in tqdm(range(g)):
        for subproblem in subproblems:
            # Reproduccion
            individuo = operador_evolutivo_2(subproblem.neighbours)

            # Evaluacion
            solution = evaluar_individuo(individuo, type)
            z = z + 1
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
                operador_seleccion(neighbour, solution, reference_point)

    #Visualizacion
        i = i + 1

    visualization(subproblems, reference_point, type)
    print("Iteraciones: " + str(z))

########################################################################################################


# Ejecucion
########################################################################################################
g = 100
n = 100
t = 25
type = "cf6"
dimension = 16
search_space = [-2, 2]

algorithm(g, n, t, search_space, dimension, type)

