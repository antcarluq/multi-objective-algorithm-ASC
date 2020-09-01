class Subproblem:

    def __init__(self, x, y, individual, neighbours):
        self.x = x
        self.y = y
        self.individual = individual
        self.neighbours = neighbours

    def __repr__(self):
        return "(" + str(self.x) + ", " + str(self.y) + "). " + str(self.individual) + ". [ " + str(self.neighbours) + "]"

class Individual:

    def __init__(self, gen, solution):
        self.gen = gen
        self.solution = solution

    def __repr__(self):
        return str(self.gen)