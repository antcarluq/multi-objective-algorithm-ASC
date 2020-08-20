class Subproblem:

    def __init__(self, x, y, individuo, neighbours):
        self.x = x
        self.y = y
        self.individuo = individuo
        self.neighbours = neighbours

    def __repr__(self):
        return "(" + str(self.x) + ", " + str(self.y) + "). " + str(self.individuo) + ". [ " + str(self.neighbours) + "]"

class Individuo:

    def __init__(self, gen, solution):
        self.gen = gen
        self.solution = solution

    def __repr__(self):
        return str(self.gen)