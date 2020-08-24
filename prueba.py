import random
import math
from subproblem import Individuo

def test_zdt3(individuo): # FIXME esto tiene que estar mal porque da unos resultados grandes en plan (0.2, 4.1) y he visto que la funcion zdt3 no alcanza valores para 1 mayores que 1
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
    print(g)
    h = 1 - math.sqrt(gen[0] / g) - (gen[0] / g) * math.sin(10 * math.pi * gen[0])
    print(h)
    y.insert(1, (g * h))

    return y



i = 0
while True:
    gen = []
    for j in range(30):
        gen.append(random.uniform(0, 1))

    individuo = Individuo(gen, None)
    y = test_zdt3(individuo)
    print(y)
    i = i + 1
    if y[1] < 1:
        print("Despues de " + str(i) + " iteraciones")
        break