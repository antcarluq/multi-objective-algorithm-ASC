from main import *

#evaluations = [(subproblemas, generaciones),...]
evaluations = [(40, 100), (80, 50), (100, 40), (40, 250), (100, 100), (200, 50)]
seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]



#ZDT3
##########################################
for evaluation in evaluations:
    for seed in seeds:
        generations = evaluation[1]
        population = evaluation[0]
        iterations = generations * population
        neigbours = int(evaluation[0] * 0.4)
        subproblems = algorithm(generations, population, neigbours, [0, 1], 30, "zdt3", seed, 0)
        f = open("results/MyAlgorithm/ZDT3/EVAL" + str(iterations) + "/P" + str(population) + "G" + str(generations) + "/final_pop_seed" + str(seed) + ".out", "w")
        for subproblem in subproblems:
            f.write(str(subproblem.individual.solution[0]) + " " + str(subproblem.individual.solution[1]) + "\n")
        f.close()
##########################################

#CF6 4D
##########################################
for evaluation in evaluations:
    for seed in seeds:
        generations = evaluation[1]
        population = evaluation[0]
        iterations = generations * population
        neigbours = int(evaluation[0] * 0.4)
        subproblems = algorithm(generations, population, neigbours, [-2, 2], 4, "cf6", seed, 0)
        f = open("results/MyAlgorithm/CF64D/EVAL" + str(iterations) + "/P" + str(population) + "G" + str(generations) + "/final_pop_seed" + str(seed) + ".out", "w")
        for subproblem in subproblems:
            f.write(str(subproblem.individual.solution[0]) + " " + str(subproblem.individual.solution[1]) + "\n")
        f.close()
##########################################

#CF6 16D
##########################################
for evaluation in evaluations:
    for seed in seeds:
        generations = evaluation[1]
        population = evaluation[0]
        iterations = generations * population
        neigbours = int(evaluation[0] * 0.4)
        subproblems = algorithm(generations, population, neigbours, [-2, 2], 16, "cf6", seed, 0)
        f = open("results/MyAlgorithm/CF616D/EVAL" + str(iterations) + "/P" + str(population) + "G" + str(generations) + "/final_pop_seed" + str(seed) + ".out", "w")
        for subproblem in subproblems:
            f.write(str(subproblem.individual.solution[0]) + " " + str(subproblem.individual.solution[1]) + "\n")
        f.close()
##########################################