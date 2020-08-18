import numpy as numpy
import random
import math
import matplotlib.pyplot as plt

from subproblem import Subproblem


# Metodo inicilizar una distribucion uniforme de vectores cuyas componenetes sumen la unidad
def initialize_subproblems(n):
    i = 0
    subproblems = []
    a = 0.5
    b = 0.5
    if n % 2 == 1:
        i = 1
        subproblems.append(Subproblem(a, b, []))
    while i < n:
        aux = 1/n
        if i % 2 == 0:
            a = a + aux
            subproblems.append(Subproblem(round(a, 2), round(1-a, 2), []))
        else:
            b = b + aux
            subproblems.append(Subproblem(round(1-b, 2), round(b, 2), []))
        i = i + 1

    return subproblems
#######################################################################################################


# Metodo mejorado inicilizar una distribucion uniforme de vectores cuyas componenetes sumen la unidad
def initialize_subproblems_old_version(n):
    i = 1
    subproblems = []
    a = 1/n
    subproblems.append(Subproblem(a, 1-a, []))
    while i < n:
        a = round(a + 1/n, 10)
        subproblems.append(Subproblem(round(a, 10), round(1-a, 10), []))
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
        subproblem.neighbours.extend(matrix[:, 0].tolist())
#######################################################################################################


# Metodo para generar la población inicial
def generar_poblacion(n, search_space):
    poblacion = []
    random.seed(30)
    for i in range(n):
        individuo = []
        for j in range(30):
            individuo.append(random.uniform(search_space[0], search_space[1]))
        poblacion.append(individuo)
    return poblacion
########################################################################################################


# Metodo para evaluar individualmente a un individuo
def evaluar_individuo(individuo):
    return test_zdt3(individuo)
########################################################################################################


# Metodo para inicializar el punto de rerefencia con la poblacion inicial
def initialize_reference_point(poblacion):
    reference_point = []
    y0min = 10000000
    y1min = 10000000
    for individuo in poblacion:
        y = test_zdt3(individuo)
        if y[0] < y0min:
            y0min = y[0]
        if y[1] < y1min:
            y1min = y[1]
    reference_point.insert(0, y0min)
    reference_point.insert(1, y1min)
    return reference_point
#######################################################################################################


# Formula de ZDT3
def test_zdt3(individuo): # FIXME esto tiene que estar mal porque da unos resultados grandes en plan (0.2, 4.1)
    sum = 0
    n = len(individuo)
    i = 1
    while i < n:
        sum = sum + individuo[i]
        i = i + 1

    y = []
    y.insert(0, individuo[0])
    g = 1 + ((9 * sum) / (n - 1))
    h = 1 - math.sqrt(individuo[0] / g) - (individuo[0] / g) * math.sin(10 * math.pi * individuo[0])
    y.insert(1, g * h)
    print(y)
    return y
########################################################################################################


# Operador evolutivo
def operador_evolutivo(neighbours):
    # TODO: Hacerlo bien, ahora mismo es un individuo aleatorio
    individuo = []
    for i in range(30):
        individuo.append(random.uniform(search_space[0], search_space[1]))

    return individuo
########################################################################################################


# Algoritmo multiobjetivo basado en agregacion
def algorithm(g, n, t, search_space):
    # Apartado: Inicializacion
    subproblems = initialize_subproblems(n)
    calcular_vecinos(t, subproblems)
    poblacion = generar_poblacion(n, search_space)
    reference_point = initialize_reference_point(poblacion)
    # ¿Las soluciones que salgan de evaluar a la poblacion inicial se guardan como resultado de cada subproblema?

    # Actualización por cada iteración
    i = 0
    while i < g:
        for subproblem in subproblems:
            # Reproduccion
            individuo = operador_evolutivo(subproblem.neighbours)
            # Evaluacion
            solution = evaluar_individuo(individuo)
            # Actualizacion del punto de referencia: Si solucion es mejor actualizo el punto de referencia, entiendo por
            # el enunciado que se compara componente a componete
            if reference_point[0] > solution[0]:
                reference_point.pop(0)
                reference_point.insert(0, solution[0])
            if reference_point[1] > solution[1]:
                reference_point.pop(1)
                reference_point.insert(1, solution[1])
            # Actualizacion de vecinos: Por cada vecino del subproblema estudiado vemos si la solucion obtenida es mejor
            # que la existente
            for neighbour in subproblem.neighbours:
                best_solution_point = numpy.array((neighbour.best_solution[0], neighbour.best_solution[1]))
                solution_point = numpy.array((solution[0], solution[1]))
                rp_point = numpy.array((reference_point[0], reference_point[1]))
                dist_neighbour_to_rp = numpy.linalg.norm(best_solution_point - rp_point)
                dist_solution_to_rp = numpy.linalg.norm(solution_point - rp_point)

                if dist_solution_to_rp < dist_neighbour_to_rp:
                    neighbour.best_solution.pop(0)
                    neighbour.best_solution.insert(0, solution[0])
                    neighbour.best_solution.pop(1)
                    neighbour.best_solution.insert(1, solution[1])

    i = i +1
    # Visualizacion
    plt.plot(reference_point[0], reference_point[1], 'bo')
    for subproblem in subproblems:
        plt.plot(subproblem.x, subproblem.y, 'ro')
        plt.plot(subproblem.best_solution[0], subproblem.best_solution[1], 'go')
    plt.show()
########################################################################################################


# Ejecucion
########################################################################################################
g = 100
n = 20
t = 2
search_space = [0, 1]

algorithm(g, n, t, search_space)
