class Subproblem:

    def __init__(self, x, y, neighbours, best_solution=[10000, 10000]):
        self.x = x
        self.y = y
        self.best_solution = best_solution
        self.neighbours = neighbours

    def __repr__(self):
        return "(" + str(self.x) + ", " + str(self.y) + "). " + str(self.best_solution) + ". [ " + str(self.neighbours) + "]"