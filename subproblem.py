class Subproblem:

    def __init__(self, x, y, best_solution, neighbours=[]):
        self.x = x
        self.y = y
        self.best_solution = best_solution
        self.neighbours = neighbours

