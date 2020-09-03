import os

evaluations = [(40, 100), (80, 50), (100, 40), (40, 250), (100, 100), (200, 50)]
seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
problems = ["ZDT3", "CF64D", "CF616D"]


response_one = "2" # Cuantos archivos quiero estudiar
response_two = "0" # Quiero estudiar 1 generacion (0) o todas (1)
response_three = "2" # Numero de objetivos
response_six = "1" # Generacion a estudiar del primer archivo
response_nine = "1" # Generacion a estudiar del segundo archivo
reponse_ten = "0" # Quiero que se cree un punto de referencia para el hiper volumen


##########################################
for problem in problems:
    for evaluation in evaluations:
        for seed in seeds:
            population = evaluation[0]
            generations = evaluation[1]
            iterations = population * generations
            response_four = "results/MyAlgorithm/"+ problem +"/EVAL" + str(iterations) + "/P" + str(population) + "G" + str(generations) + "/final_pop_seed" + str(seed) + ".out" # Ruta del primer archivo
            response_five = str(population) # Poblacion del primer archivo
            response_seven = "results/NSGAII/"+ problem +"/EVAL" + str(iterations) + "/P" + str(population) + "G" + str(generations) + "/all_popm_seed" + str(seed) + ".out" # Ruta del segundo archivo
            response_eight = str(population) # Poblacion del segundo archivo
            response_nine = str(generations)  # Generacion a estudiar del segundo archivo

            ruta_input_file = "metrics/inputfiles/"+ problem +"/EVAL" + str(iterations) + "/P" + str(population) + "G" + str(
                generations) + "/seed" + str(seed) + ".in"
            f = open(ruta_input_file, "w")
            f.write(str(response_one + "\n" +
                        response_two + "\n" +
                        response_three + "\n" +
                        response_four + "\n" +
                        response_five + "\n" +
                        response_six + "\n" +
                        response_seven + "\n" +
                        response_eight + "\n" +
                        response_nine + "\n" +
                        reponse_ten))
            f.close()

            result = os.popen("metrics/metrics < ./" + ruta_input_file).read()
            r = open("results/comparation/"+ problem +"/EVAL" + str(iterations) + "/P" + str(population) + "G" + str(
                generations) + "/seed" + str(seed) + ".txt", "w+")
            r.write(result)
            r.close()

#######################################