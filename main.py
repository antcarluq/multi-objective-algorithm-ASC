import numpy as numpy
import random
import math
from vector import Vector_Peso

# Metodo inicilizar una distribucion uniforme de vectores cuyas componenetes sumen la unidad
def inicializar_vectores_peso(N):
    i = 0
    lista_vectores_peso = []
    a = 0.5
    b = 0.5
    if N%2 == 1:
        i=1
        lista_vectores_peso.append(Vector_Peso(a, b, []))
    while i < N:
        aux = 1/N
        if i%2 == 0:
            a = a + aux
            lista_vectores_peso.append(Vector_Peso(round(a, 2), round(1-a, 2), []))
        else:
            b = b + aux
            lista_vectores_peso.append(Vector_Peso(round(1-b, 2), round(b, 2), []))
        i = i + 1
    for aux in lista_vectores_peso:
        print("Vector: " + str(aux.x) + " " + str(aux.y) + " y sus vecinos son: " + str(aux.vecinos))
    return lista_vectores_peso
#######################################################################################################


# Metodo mejorado inicilizar una distribucion uniforme de vectores cuyas componenetes sumen la unidad
def inicializar_vectores_peso_old_version(N):
    i = 1
    lista_vectores_peso = []
    a = 1/N
    lista_vectores_peso.append(Vector_Peso(a, 1-a, []))
    while i < N:
        a = round(a + 1/N, 10)
        lista_vectores_peso.append(Vector_Peso(round(a, 10), round(1-a, 10), []))
        i = i + 1
    for aux in lista_vectores_peso:
        print("Vector: " + str(aux.x) + " " + str(aux.y) + " y sus vecinos son: " + str(aux.vecinos))
    return lista_vectores_peso
#########################################################################


# Metodo para encontrar los T vectores vecinos mas cercanos
def calcular_vecinos(self, T, lista_vectores_peso):
    T = 2
    for vector in lista_vectores_peso:
        lista_vector_distancia = []
        for potential_neighbour in lista_vectores_peso:
            # Calculo la distancia euclidea de los pares de vectores
            a = numpy.array((vector.x, vector.y))
            b = numpy.array((potential_neighbour.x, potential_neighbour.y))
            dist = numpy.linalg.norm(a - b)

            # Guardo en una matriz los potenciales vectores vecinos y su distancia al vector estudiado
            lista_vector_distancia.append(numpy.array((potential_neighbour, dist)))

            # Los ordenado de menor a mayor segun la distancia y me quedo con los T primeros
            lista_vector_distancia.sort(key=lambda tup: tup[1])
            lista_vector_distancia = lista_vector_distancia[0:T]
            matriz = numpy.array(lista_vector_distancia)

        # Guardo los vecinos mas cercanos en el vector estudiado
        vector.vecinos.append(matriz[:,0].tolist())

    for aux in lista_vectores_peso:
        print("Vector: " + str(aux.x) +" "+ str(aux.y) + " y sus vecinos son: " + str(aux.vecinos))
########################################################################3

# TODO creo que estos no son numeros sino que son componentes
# Metodo para generar la población inicial #TODO Posibilidad de utilizar otro tipo de generacion de numeros aleatorios
def generar_poblacion(N):
    poblacion = []
    for i in range(N): #Tiene que estar entre 1 y 0 usar seed y que siempre sean los mismos los numeros aleatorios
        poblacion.append(random.randint(0, 10)) #TODO ¿Tienen que estar limitados estos numeros? ¿Quizas por el espacio de busqueda? ¿Se pueden repetir los numeros?
    return poblacion

def evaluar_poblacion(poblacion):
    for individuo in poblacion:
        None # Aqui hay que evaluar pero entiendo que es en funcion del problema ¿De donde recibo la funcion para evaluar?


# Algoritmo multiobjetivo basado en agregacion
########################################################################
def algorithm(N, T):
    lista_vectores_peso = inicializar_vectores_peso(N)
    calcular_vecinos(T, N, lista_vectores_peso)
    poblacion = generar_poblacion(N)
    evaluar_poblacion(poblacion)
    # Inicializa la lista de puntos de referencia donde zi es el mejor valor de valor del objetivo fi encontrado. Se ira actualizando
    i=0
    while i < N:
        #Reproduccion
        #Evaluacion
        #Actualizacion del punto de referencia
        #Actualizacion de vecinos
        i = i + 1

########################################################################

def test_zdt3(poblacion, N):
    aux = 0
    for i in range(N):
        aux = aux + poblacion[i]

    y = []
    y[0] = poblacion[0]
    g = 1 + ((9 * aux) / (N - 1));
    h = 1 - math.sqrt(poblacion[0] / g) - (poblacion[0] / g) * math.sin(10 * math.pi * poblacion[0])
    y[1] = g * h

    return y