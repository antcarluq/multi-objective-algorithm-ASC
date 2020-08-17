import numpy as numpy
import random
import math
import matplotlib.pyplot as plt

from subproblem import Subproblem

# Metodo inicilizar una distribucion uniforme de vectores cuyas componenetes sumen la unidad
def initialize_subproblems(N):
    i = 0
    subproblems = []
    a = 0.5
    b = 0.5
    if N%2 == 1:
        i=1
        subproblems.append(Subproblem(a, b, []))
    while i < N:
        aux = 1/N
        if i%2 == 0:
            a = a + aux
            subproblems.append(Subproblem(round(a, 2), round(1-a, 2), []))
        else:
            b = b + aux
            subproblems.append(Subproblem(round(1-b, 2), round(b, 2), []))
        i = i + 1
    for aux in subproblems:
        None
        #print("Vector: " + str(aux.x) + " " + str(aux.y) + " y sus vecinos son: " + str(aux.neighbours))
    return subproblems
#######################################################################################################


# Metodo mejorado inicilizar una distribucion uniforme de vectores cuyas componenetes sumen la unidad
def initialize_subproblems_old_version(N):
    i = 1
    subproblems = []
    a = 1/N
    subproblems.append(Subproblem(a, 1-a, []))
    while i < N:
        a = round(a + 1/N, 10)
        subproblems.append(Subproblem(round(a, 10), round(1-a, 10), []))
        i = i + 1
    for aux in subproblems:
        None
        #print("Vector: " + str(aux.x) + " " + str(aux.y) + " y sus vecinos son: " + str(aux.vecinos))
    return subproblems
#########################################################################


# Metodo para encontrar los T vectores vecinos mas cercanos
def calcular_vecinos(self, T, subproblems):
    for subproblem in subproblems:
        list_subproblem_dist = []
        for potential_neighbour in subproblems:
            # Calculo la distancia euclidea de los pares de vectores
            a = numpy.array((subproblem.x, subproblem.y))
            b = numpy.array((potential_neighbour.x, potential_neighbour.y))
            dist = numpy.linalg.norm(a - b)

            # Guardo en una matriz los potenciales vectores vecinos y su distancia al vector estudiado
            list_subproblem_dist.append(numpy.array((potential_neighbour, dist)))

            # Los ordenado de menor a mayor segun la distancia y me quedo con los T primeros
        list_subproblem_dist.sort(key=lambda tup: tup[1])
        matrix = numpy.array(list_subproblem_dist[0:T])

        # Guardo los vecinos mas cercanos en el vector estudiado
        subproblem.neighbours.extend(matrix[:,0].tolist())
        break

        for aux in subproblems:
            None
            #print("Vector: " + str(aux.x) +" "+ str(aux.y) + " y sus vecinos son: " + str(aux.neighbours))
########################################################################3


# Metodo para generar la población inicial
def generar_poblacion(N, search_space):
    poblacion = []
    #TODO random.seed(30) usar seed y que siempre sean los mismos los numeros aleatorios
    for i in range(N):
        poblacion.append(random.uniform(search_space[0], search_space[1]))#TODO Seguir investigando random uniform
    return poblacion


# Algoritmo multiobjetivo basado en agregacion
#######################################################################

def algorithm(N, T, search_space):
    #Apartado: Inicializacion
    subproblems = initialize_subproblems(N)
    calcular_vecinos(N, T, subproblems)
    poblacion = generar_poblacion(N, search_space)
    reference_point = initialize_reference_point(poblacion, N) #Las soluciones que salgan de evaluar a la poblacion inicial se guardan como resultado de cada subproblema?

    #Actualización por cada iteración
    for subproblem in subproblems:
        #Reproduccion
            individuo = operador_evolutivo(subproblems)
        #Evaluacion
            solution = evaluar_individuo(individuo, N)
        #Actualizacion del punto de referencia: Si solucion es mejor actualizo el punto de referencia, entiendo por el enunciado que se compara componente a componete
            if reference_point[0] > solution[0]:
                reference_point.insert(0, solution[0])
            if reference_point[1] > solution[1]:
                reference_point.insert(1, solution[1])
        # Actualizacion de vecinos: Por cada vecino del subproblema estudiado vemos si la solucion obtenida es mejor que la existente
            for neighbour in subproblem.neighbours:
                n_vector = numpy.array((neighbour.best_solution[0], neighbour.best_solution[1]))
                s_vector = numpy.array((solution[0], solution[1]))
                rp_vector = numpy.array((reference_point[0], reference_point[1]))
                dist_neighbour_to_rp = numpy.linalg.norm(n_vector - rp_vector)
                dist_solution_to_rp = numpy.linalg.norm(s_vector - rp_vector)

                if dist_solution_to_rp < dist_neighbour_to_rp:
                    neighbour.best_solution.insert(0, solution[0])
                    neighbour.best_solution.insert(1, solution[1])

    ############################## Visualizacion
    plt.plot(reference_point[0], reference_point[1], 'bo')
    for subproblem in subproblems:
        plt.plot(subproblem.x, subproblem.y, 'ro')
        plt.plot(subproblem.best_solution[0], subproblem.best_solution[1], 'go')
    plt.show()
########################################################################
def evaluar_individuo(individuo, N):
    poblacion = []
    poblacion.append(individuo)
    return test_zdt3(individuo, poblacion, N)



def initialize_reference_point(poblacion, N):
    reference_point =[]
    y0min = 10000000
    y1min = 10000000
    for individuo in poblacion:
        y = test_zdt3(individuo, poblacion, N)
        if y[0] < y0min:
            y0min = y[0]
        if y[1] < y1min:
            y1min = y[1]
    reference_point.insert(0, y0min)
    reference_point.insert(1, y1min)
    return reference_point

#FIXME Algo mal estoy haciendo fijo
def test_zdt3(individuo, poblacion, N): # y[0] es el eje x, y[1] es el eje y
    sum = 0
    i = 0
    #i = 2 Esto lo pone en la formula pero no lo entiendo
    #while i < N: #TODO Optimizar esto
    while i < len(poblacion):
        sum = sum + poblacion[i]
        i = i + 1

    y = []
    y.insert(0, individuo)
    g = 1 + ((9 * sum) / (N - 1));
    h = 1 - math.sqrt(individuo / g) - (individuo / g) * math.sin(10 * math.pi * individuo)
    y.insert(1, g * h)

    return y
########################################################################################################


# Operador evolutivo
def operador_evolutivo(subproblems):
    individuo = 4
    #TODO: Hacerlo bien, ahora mismo es un numero aleatorio
    return random.uniform(search_space[0], search_space[1])
########################################################################################################

N = 5
T = 3
search_space = [0, 1]

algorithm(N, T, search_space)
