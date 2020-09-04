evaluations = [(40, 100), (80, 50), (100, 40), (40, 250), (100, 100), (200, 50)]
seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
problems = ["ZDT3", "CF64D", "CF616D"]

for problem in problems:
    for evaluation in evaluations:
        sum_hyper_1 = 0
        sum_hyper_2 = 0
        sum_spacing_1 = 0
        sum_spacing_2 = 0
        sum_c_1 = 0
        sum_c_2 = 0
        for seed in seeds:
            population = evaluation[0]
            generations = evaluation[1]
            iterations = population * generations

            file = open("results/comparation/"+ problem +"/EVAL" + str(iterations) + "/P" + str(population) + "G" + str(
                generations) + "/seed" + str(seed) + ".txt", "r").read()

            lines = file.split("\n")

            sum_hyper_1 = sum_hyper_1 + float(lines[7].split(":")[1].strip())
            sum_spacing_1 = sum_spacing_1 + float(lines[8].split(":")[1].strip())
            sum_hyper_2 = sum_hyper_2 + float(lines[9].split(":")[1].strip())
            sum_spacing_2 = sum_spacing_2 + float(lines[10].split(":")[1].strip())
            sum_c_2 = sum_c_2 + float(lines[11].split("=")[1].strip())
            sum_c_1 = sum_c_1 + float(lines[12].split("=")[1].strip())


        print("\n")
        print("#####################################################")
        print("Media para " + str(problem) + " con " + str(iterations) + " evaluaciones y P = " + str(population) + " G = " + str(generations))
        print("Hipervolumen 1: " + str(sum_hyper_1/len(seeds)))
        print("Hipervolumen 2: " + str(sum_hyper_2 / len(seeds)))
        print("Spacing 1: " + str(sum_spacing_1 / len(seeds)))
        print("Spacing 2: " + str(sum_spacing_2 / len(seeds)))
        print("C 1: " + str(sum_c_1 / len(seeds)))
        print("C 2: " + str(sum_c_2 / len(seeds)))
        print("#####################################################")
        print("\n")