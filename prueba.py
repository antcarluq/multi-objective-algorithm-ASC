from subproblem import Subproblem

subproblem1 = Subproblem(0, 0, [])
subproblem1.x = 1
subproblem1.y = 2

subproblem2 = Subproblem(0, 0, [])
subproblem2.x = 3
subproblem2.y = 4

subproblem4 = Subproblem(0, 0, [])
subproblem4.x = 6
subproblem4.y = 7
subproblem4.neighbours = [subproblem1, subproblem2]


print(subproblem4.neighbours)