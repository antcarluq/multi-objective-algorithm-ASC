import numpy as numpy
import random
import math
import matplotlib.pyplot as plt

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
def initialize_subproblems(n):
    i = 1
    subproblems = []
    a = 1/(n+1)
    subproblems.append(Subproblem(a, 1-a, None, []))
    while i < n:
        a = round(a + 1/(n+1), 10)
        subproblems.append(Subproblem(round(a, 10), round(1-a, 10), None, []))
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
        #subproblem.neighbours.extend(matrix[:, 0].tolist())
#######################################################################################################


# Metodo para generar la población inicial
def generar_poblacion(subproblems, search_space):
    poblacion = []
    random.seed(30)
    for subproblem in subproblems:
        gen = []
        for j in range(30): #TODO Poner como variable, estas son las dimensiones
            gen.append(random.uniform(search_space[0], search_space[1]))

        individuo = Individuo(gen, None)
        setattr(subproblem, "individuo", individuo)

        poblacion.append(gen)
    return poblacion
########################################################################################################


# Metodo para evaluar individualmente a un individuo
def evaluar_individuo(individuo):
    return test_zdt3(individuo)
########################################################################################################


# Metodo para inicializar el punto de rerefencia con la poblacion inicial
def initialize_reference_point(subproblems):
    reference_point = []
    y0min = 10000000
    y1min = 10000000
    for subproblem in subproblems:
        y = test_zdt3(subproblem.individuo)
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
def test_zdt3(individuo): # FIXME esto tiene que estar mal porque da unos resultados grandes en plan (0.2, 4.1)
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
    y.insert(1, g * h)

    return y
########################################################################################################


# Operador evolutivo: La media entre dos individuos
def operador_evolutivo(neighbours):
    list_aux = list(range(len(neighbours)))
    i = random.choice(list_aux)
    list_aux.remove(i)
    j = random.choice(list_aux)
    neighbour_1 = neighbours[i]
    neighbour_2 = neighbours[j]
    gen = []
    k = 0
    while k < len(neighbour_1.individuo.gen):
        if random.choice(([0, 1])) == 0:
            gen.append((neighbour_1.individuo.gen[k] + neighbour_2.individuo.gen[k]) / 2)
        else:
            gen.append(random.uniform(search_space[0], search_space[1]))
        k = k + 1

    individuo = Individuo(gen, None)
    return individuo
########################################################################################################


# Algoritmo multiobjetivo basado en agregacion
def algorithm(g, n, t, search_space):
    # Apartado: Inicializacion
    subproblems = initialize_subproblems(n)
    calcular_vecinos(t, subproblems)
    generar_poblacion(subproblems, search_space)
    reference_point = initialize_reference_point(subproblems)

    plt.plot(reference_point[0], reference_point[1], 'bo')
    for subproblem in subproblems:
        plt.plot(subproblem.x, subproblem.y, 'ro')
        plt.plot(subproblem.individuo.solution[0], subproblem.individuo.solution[1], 'go')
    plt.show(block=False)
    plt.pause(0.01)

    # Actualización por cada iteración
    i = 0
    z = 0 # Fixme no vale pa na
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
                best_solution_point = numpy.array((neighbour.individuo.solution[0], neighbour.individuo.solution[1]))
                solution_point = numpy.array((solution[0], solution[1]))
                rp_point = numpy.array((reference_point[0], reference_point[1]))
                subproblem_point = numpy.array((neighbour.x, neighbour.y))
                #subproblem_point = numpy.array(((0, 0))) # FIXME Trampa
                #dist_neighbour_to_rp = numpy.linalg.norm(best_solution_point - rp_point)
                #dist_solution_to_rp = numpy.linalg.norm(solution_point - rp_point)
                dist_best_solution_to_subproblem = numpy.linalg.norm(best_solution_point - subproblem_point)
                dist_solution_to_subproblem = numpy.linalg.norm(solution_point - subproblem_point)


                if dist_solution_to_subproblem < dist_best_solution_to_subproblem:
                    z = z + 1
                    print("Se ha actualizado la solucion " + str(z) + " veces, en la generacion numero " + str(i))
                    setattr(neighbour.individuo, "solution", [solution[0], solution[1]])
                #if dist_solution_to_rp < dist_neighbour_to_rp:
                    #setattr(neighbour.individuo, "solution", [solution[0], solution[1]])

                # rp_point = numpy.array((reference_point[0], reference_point[1]))
                # dist_neighbour_to_rp = numpy.linalg.norm(best_solution_point - rp_point)
                # dist_solution_to_rp = numpy.linalg.norm(solution_point - rp_point)

                # if dist_solution_to_rp < dist_neighbour_to_rp:
                # setattr(neighbour.individuo, "solution", [solution[0], solution[1]])

                if subproblem == subproblems[0]:
                    if False:
                        print("Vuelta " + str(i))
                        print("Solucion mejor actual: " + str(best_solution_point)  +" y distancia al 0,0: " + str(dist_best_solution_to_subproblem))
                        print("Potencial Solucion: " + str(solution_point)  +" y distancia al 0,0: " + str(dist_solution_to_subproblem))
                        print("Solucion real: " + str(subproblems[0].individuo.solution))
                        print("Solucion real del vecino: " + str(subproblems[0].neighbours[1].individuo.solution))
                        print("###########################################################")
        i = i + 1

    plt.plot(reference_point[0], reference_point[1], 'bo')
    for subproblem in subproblems:
        plt.plot(subproblem.x, subproblem.y, 'ro')
        plt.plot(subproblem.individuo.solution[0], subproblem.individuo.solution[1], 'go')
    plt.show(block = False)
    plt.pause(0.01)

########################################################################################################


# Ejecucion
########################################################################################################
g = 500
n = 20
t = 3
search_space = [0, 1]

algorithm(g, n, t, search_space)