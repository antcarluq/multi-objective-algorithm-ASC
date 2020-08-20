lista1 = [2,3,4,5,6]

import random

listauax = []

list = list(range(len(lista1)))
print(list)
i = random.choice(list)
list.remove(i)
j = random.choice(list)


print(i)
print(list)
print(j)