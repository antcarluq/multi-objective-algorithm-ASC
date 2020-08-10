import numpy as numpy
from vector import Vector_Peso

# Tengo que crear N Vectores Peso
# Debiéndose cumplir que la suma de los componentes de cada vector es la unidad
# Dichos vectores peso están distribuidos uniformemente, simplemente significa que los vectores están equiespaciados,
# donde la distancia se define de forma euclídea

# Basicamente aqui quiero que se hay 4 subproblemas tener 4 vectores de la siguiente forma:
#a = numpy.array((1, 2))
#b = numpy.array((2, 3))
#dist = numpy.linalg.norm(a-b)
#print(dist)

#T = 2
#N = 4
#i = 0
#matriz_vector_peso = []
#while i < N:
#    matriz_vector_peso.append(Vector_Peso(0.2, 0.8, []))
#    i = i + 1

#print(matriz_vector_peso)
# V1 = (0.1,0.9) | V2 = (0.9,0.1) | V3 = (0.4,0.6) | V4 = (0.6,0.4) *Suponiendo que estos vectores estan equiespaciados

# De momento solo quiero el mas cercano del primero

#matriz_vector_peso = []
#matriz_vector_peso.append(Vector_Peso(1, 1, []))
#matriz_vector_peso.append(Vector_Peso(1, 2, []))
#matriz_vector_peso.append(Vector_Peso(5, 2, []))
#matriz_vector_peso.append(Vector_Peso(4, 3, []))

# Metodo inicilizar una distribucion uniforme de vectores cuyas componenetes sumen la unidad
def inicializar_vectores_peso_old_version(N=10):
    i = 1
    lista_vectores_peso = []
    a = 0.5
    b = 0.5
    lista_vectores_peso.append(Vector_Peso(a, b, []))
    while i < N:
        aux = 1/N
        if i%2 == 0:
            a = a + aux
            lista_vectores_peso.append(Vector_Peso(a, 1-a, []))
        else:
            b = b + aux
            lista_vectores_peso.append(Vector_Peso(1-b, b, []))
        i = i + 1
    for aux in lista_vectores_peso:
        print("Vector: " + str(aux.x) + " " + str(aux.y) + " y sus vecinos son: " + str(aux.vecinos))

#inicializar_vectores_peso_old_version()
# Metodo mejorado inicilizar una distribucion uniforme de vectores cuyas componenetes sumen la unidad
def inicializar_vectores_peso(N=2):
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

inicializar_vectores_peso()
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
