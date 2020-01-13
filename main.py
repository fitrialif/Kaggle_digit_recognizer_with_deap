# inspired by https://medium.com/@RaghavPrabhu/kaggles-digit-recogniser-using-tensorflow-lenet-architecture-92511e68cee1


# imports
import deap_functions
import data_load
import random
import time


# 1
def main_standard():
    start_time = time.time()
    # parenthood probability
    parenthood_probability = 0.5

    # create toolbox
    # create toolbox using deap_functions
    toolbox = deap_functions.create_toolbox()

    # loading data
    input_file = 'data/train.csv'
    image_size = 28
    training_dataset, testing_dataset, validating_dataset, training_labels, testing_labels, validating_labels = \
        data_load.data_load(input_file, image_size)

    #output = open("../data/initial_populationxxxx2.txt", "w+")

    # create population
    population_size = 30
    population = toolbox.population(population_size)

    # evaluate all
    # fitnesses = list(map(toolbox.evaluate_normal, pop))
    # for ind, fit in zip(population, fitnesses):
    #     ind.fitness.values = fit
    print("evaluating initial population")
    for i, ind in enumerate(population):
        print("evaluating individual: ", i, end='')
        ind.fitness.values = toolbox.evaluate_fast(ind, training_dataset, validating_dataset,
                                                   testing_dataset, training_labels, validating_labels,
                                                   testing_labels)
        print("   fitness :", ind.fitness.values)
    print("initial population", len(population))
    for ind in population:
        print(ind.fitness.values, ind)
    print("initial population - end")
    # start loop
    time_used = time.time() - start_time
    print("+++ time used {:.5f} +++" .format(time_used))
    for generation in range(1, 200):
        print("--- generation: ", generation)

        # select elite 20%
        elite = toolbox.select_elite(population, int(population_size*0.2))
        print("elite selected: ", len(elite))
        for ind in elite:
            print(ind.fitness.values, ind)
        print("elite - end")

        # select through tournament 80% and mate them (1st with 2nd, 3rd with 4th etc) with probability
        # that random number is less than this probability
        # since mate returns child, then do mating in a loop, where number of children should be = len(pop) - len(elite)
        children = []
        while len(children) < population_size - len(elite):
            parents = toolbox.select_parents(population, 2)
            if random.random() < parenthood_probability:
                #print("mating")
                child = toolbox.mate(toolbox, parents[0], parents[1])

                # imprint child with generation number
                child[8] = generation
                children.append(child)
            #else:
                #print("not mating")

        # calculate fitness for children
        print("calculating fitness of the children with max time : ", int(generation*2+10))
        max_time = int(8 + generation*2)
        for i, ind in enumerate(children):
            print("child: ", i)
            ind.fitness.values = toolbox.evaluate(ind, max_time, training_dataset, validating_dataset,
                                                       testing_dataset, training_labels, validating_labels,
                                                       testing_labels)
            print("child: ", i, end='')
            print("   fitness: ", ind.fitness.values)
        # merge children with elite
        population = children + elite
        print("intermediate population", len(population))
        for ind in population:
            print(ind.fitness.values, ind)
        print("intermediate population - end")
        time_used = time.time() - start_time
        print("+++ time used {:.5f} +++".format(time_used))

    print("final population", len(population))
    for ind in population:
        print(ind.fitness.values, ind)
    print("final population - end")
    time_used = time.time() - start_time
    print("+++ time used {:.5f} +++" .format(time_used))


# 2
def main_additional_training_for_elite():
    start_time = time.time()
    # parenthood probability
    parenthood_probability = 0.5

    # create toolbox
    # create toolbox using deap_functions
    toolbox = deap_functions.create_toolbox()

    # loading data
    input_file = 'data/train.csv'
    image_size = 28
    training_dataset, testing_dataset, validating_dataset, training_labels, testing_labels, validating_labels = \
        data_load.data_load(input_file, image_size)

    #output = open("../data/initial_populationxxxx2.txt", "w+")

    # create population
    population_size = 30
    population = toolbox.population(population_size)

    # evaluate all
    #fitnesses = list(map(toolbox.evaluate_normal, pop))
    # for ind, fit in zip(population, fitnesses):
    #     ind.fitness.values = fit
    print("evaluating initial population")
    for i, ind in enumerate(population):
        print("evaluating individual: ", i, end='')
        ind.fitness.values = toolbox.evaluate_fast(ind, training_dataset, validating_dataset,
                                                   testing_dataset, training_labels, validating_labels,
                                                   testing_labels)
        print("   fitness :", ind.fitness.values)
    print("initial population size : ", len(population))
    for ind in population:
        print(ind.fitness.values, ind)
    print("initial population - end")
    # start loop
    time_used = time.time() - start_time
    print("+++ time used {:.5f} +++" .format(time_used))
    for generation in range(1, 200):
        # 1) select elite
        # 2) make children in the number equal population - elite
        # 3) calculate fitness of the children - train from scratch
        # 4) calculate fitness of the elite (additional training)

        print("--- generation : ", generation, " --- ")

        # 1) select elite 20%
        print("list of the elite members : ")
        elite = toolbox.select_elite(population, int(population_size * 0.2))
        print("elite selected: ", len(elite))
        for ind in elite:
            print(ind.fitness.values, ind)
        print("list of the elite members - end")

        # 2) make children
        # select through tournament 80% and mate them (1st with 2nd, 3rd with 4th etc) with probability
        # that random number is less than this probability
        # since mate returns child, then do mating in a loop, where number of children should be = len(pop) - len(elite)
        children = []
        while len(children) < population_size - len(elite):
            parents = toolbox.select_parents(population, 2)
            if random.random() < parenthood_probability:
                #print("mating")
                child = toolbox.mate(toolbox, parents[0], parents[1])

                # imprint child with generation number
                child[8] = generation
                children.append(child)
            #else:
                #print("not mating")

        # calculate fitness for children
        max_time = int(10 + generation*2)
        print("calculating fitness of the children with max time : ", max_time, " s")
        for i, ind in enumerate(children):
            #print("child: ", i)
            ind.fitness.values = toolbox.evaluate2(ind, max_time, training_dataset, validating_dataset,
                                                       testing_dataset, training_labels, validating_labels,
                                                       testing_labels)
            print("child: ", i, end='')
            print("   fitness: ", ind.fitness.values)

        # additional training for elite
        max_time = 2
        print("additional training for elite : ", max_time," s")
        for i, ind in enumerate(elite):
            #print("elite : ", i)
            ind.fitness.values = toolbox.evaluate2(ind, max_time, training_dataset, validating_dataset,
                                                       testing_dataset, training_labels, validating_labels,
                                                       testing_labels)
            print("elite: ", i, end='')
            print("   fitness: ", ind.fitness.values)

        # merge children with elite
        population = children + elite
        print("intermediate population", len(population))
        for ind in population:
            print(ind.fitness.values, ind)
        print("intermediate population - end")
        time_used = time.time() - start_time
        print("+++ time used {:.5f} +++".format(time_used))

    print("final population", len(population))
    for ind in population:
        print(ind.fitness.values, ind)
    print("final population - end")
    time_used = time.time() - start_time
    print("+++ time used {:.5f} +++" .format(time_used))



# 3
def main_additional_training_for_elite_and_mutation():

    def print_ind(individual):
        print("{:.7f}".format(individual.fitness.values[0]), end=' ')
        print("{:>6}".format(individual[0]), end=' ')
        print("{:05.3f}".format(individual[1]), end=' ')
        print("{:05.3f}".format(individual[2]), end=' ')
        for i in range(3, 9):
            print("{:>3}".format(individual[i]), end=' ')
        print("  ", end='')
        for i in range(9, 12):
            print(individual[i], end='   ')
        print("")
    def print_pop(pop):
        for individual in pop:
            print_ind(individual)

    start_time = time.time()

    parenthood_probability = 0.5
    elite_size = 0.2
    population_size = 30
    multiplicator_for_data_augmentation = 10

    # create toolbox
    # create toolbox using deap_functions
    toolbox = deap_functions.create_toolbox()

    # loading data
    input_file = 'data/train.csv'
    image_size = 28
    training_dataset, testing_dataset, validating_dataset, training_labels, testing_labels, validating_labels = \
        data_load.data_load2(input_file, image_size, multiplicator_for_data_augmentation)

    #output = open("../data/initial_populationxxxx2.txt", "w+")

    # create population

    population = toolbox.population(population_size)

    # evaluate all
    #fitnesses = list(map(toolbox.evaluate_normal, pop))
    # for ind, fit in zip(population, fitnesses):
    #     ind.fitness.values = fit
    print("evaluating initial population")
    max_time = 10
    for i, ind in enumerate(population):
        print("evaluating individual: ", i, end='')
        # ind.fitness.values = toolbox.evaluate_fast(ind, training_dataset, validating_dataset,
        #                                            testing_dataset, training_labels, validating_labels,
        #                                            testing_labels)
        ind.fitness.values = toolbox.evaluate2(ind, max_time, training_dataset, validating_dataset,
                                                   testing_dataset, training_labels, validating_labels,
                                                   testing_labels)
        print("   fitness :", ind.fitness.values)
    print("initial population size : ", len(population))
    print_pop(population)
    print("initial population - end")

    # start loop
    time_used = time.time() - start_time
    print("+++ time used {:.5f} +++" .format(time_used))
    for generation in range(1, 200):
        # 1) select elite
        # 2) make children in the number equal population - elite
        # 3) every elite member produces one mutant
        # 4) calculate fitness of the children and mutants - train from scratch
        # 5) calculate fitness of the elite (additional training)


        print("--- generation : ", generation, " --- ")

        # 1) select elite 20%
        print("list of the elite members : ")
        elite = toolbox.select_elite(population, int(population_size * elite_size))

        print("elite selected: ", len(elite))
        print_pop(elite)
        print("list of the elite members - end")

        # 2) make children
        # select through tournament 80% and mate them (1st with 2nd, 3rd with 4th etc) with probability
        # that random number is less than this probability
        # since mate returns child, then do mating in a loop, where number of children should be = len(pop) - len(elite)
        children = []
        while len(children) < population_size - len(elite*2):
            parents = toolbox.select_parents(population, 2)
            if random.random() < parenthood_probability:
                #print("mating")
                child = toolbox.mate(toolbox, parents[0], parents[1])

                # imprint child with generation number
                child[8] = generation
                children.append(child)
            #else:
                #print("not mating")

        # 3) create mutants
        mutants = []
        #print(" printing elite and mutants")
        for i, ind in enumerate(elite):
            mutant = toolbox.mutate(toolbox, ind, 1)
            mutant[8] = generation
            #print(i)
            #print_ind(ind)
            #print_ind(mutant)
            del mutant.fitness.values
            mutants.append(mutant)
        #print(" printing elite and mutants - end")

        # 4) calculate fitness for children and mutants
        offspring = children + mutants
        max_time = int(10 + generation*2)
        print("calculating fitness of the children and mutants with max time : ", max_time, " s")
        for i, ind in enumerate(offspring):
            #print("child: ", i)
            ind.fitness.values = toolbox.evaluate2(ind, max_time, training_dataset, validating_dataset,
                                                       testing_dataset, training_labels, validating_labels,
                                                       testing_labels)
            print("offspring : ", i, end='')
            print("   fitness : {:5f}".format(ind.fitness.values[0]))
            print("------------------------------------------------")

        # 5) additional training for elite
        max_time = 2
        print("additional training for elite : ", max_time," s")
        for i, ind in enumerate(elite):
            #print("elite : ", i)
            ind.fitness.values = toolbox.evaluate2(ind, max_time, training_dataset, validating_dataset,
                                                       testing_dataset, training_labels, validating_labels,
                                                       testing_labels)
            print("elite: ", i, end='')
            print("   fitness : {:5f}".format(ind.fitness.values[0]))


        # merge children with elite and mutants
        population = elite + offspring

        # list intermediate population
        print("intermediate population", len(population))
        print_pop(population)
        print("intermediate population - end")

        time_used = time.time() - start_time
        print("+++ time used {:.5f} +++".format(time_used))

    # list final population
    print("final population", len(population))
    print_pop(population)
    print("final population - end")

    time_used = time.time() - start_time
    print("+++ time used {:.5f} +++" .format(time_used))




# 4
def main_mutation_and_children_no_additional_training():

    def print_ind(individual):
        print("{:.7f}".format(individual.fitness.values[0]), end=' ')
        print("{:>6}".format(individual[0]), end=' ')
        print("{:05.3f}".format(individual[1]), end=' ')
        print("{:05.3f}".format(individual[2]), end=' ')
        for i in range(3, 9):
            print("{:>3}".format(individual[i]), end=' ')
        print("  ", end='')
        for i in range(9, 12):
            print(individual[i], end='   ')
        print("")
    def print_pop(pop):
        for individual in pop:
            print_ind(individual)

    start_time = time.time()

    parenthood_probability = 0.5
    elite_size = 0.2
    population_size = 30
    multiplicator_for_data_augmentation = 10

    # create toolbox
    # create toolbox using deap_functions
    toolbox = deap_functions.create_toolbox()

    # loading data
    input_file = 'data/train.csv'
    image_size = 28
    training_dataset, testing_dataset, validating_dataset, training_labels, testing_labels, validating_labels = \
        data_load.data_load2(input_file, image_size, multiplicator_for_data_augmentation)

    #output = open("../data/initial_populationxxxx2.txt", "w+")

    # create population

    population = toolbox.population(population_size)

    # evaluate all
    #fitnesses = list(map(toolbox.evaluate_normal, pop))
    # for ind, fit in zip(population, fitnesses):
    #     ind.fitness.values = fit
    print("evaluating initial population")
    max_time = 10
    for i, ind in enumerate(population):
        print("evaluating individual: ", i, end='')
        # ind.fitness.values = toolbox.evaluate_fast(ind, training_dataset, validating_dataset,
        #                                            testing_dataset, training_labels, validating_labels,
        #                                            testing_labels)
        ind.fitness.values = toolbox.evaluate2(ind, max_time, training_dataset, validating_dataset,
                                                   testing_dataset, training_labels, validating_labels,
                                                   testing_labels)
        print("   fitness :", ind.fitness.values)
    print("initial population size : ", len(population))
    print_pop(population)
    print("initial population - end")

    # start loop
    time_used = time.time() - start_time
    print("+++ time used {:.5f} +++" .format(time_used))
    for generation in range(1, 200):
        # 1) select elite
        # 2) make children in the number equal population - elite
        # 3) every elite member produces one mutant
        # 4) calculate fitness of the children and mutants - train from scratch


        print("--- generation : ", generation, " --- ")

        # 1) select elite 20%
        print("list of the elite members : ")
        elite = toolbox.select_elite(population, int(population_size * elite_size))

        print("elite selected: ", len(elite))
        print_pop(elite)
        print("list of the elite members - end")

        # 2) make children
        # select through tournament 80% and mate them (1st with 2nd, 3rd with 4th etc) with probability
        # that random number is less than this probability
        # since mate returns child, then do mating in a loop, where number of children should be = len(pop) - len(elite)
        children = []
        while len(children) < population_size - len(elite*2):
            parents = toolbox.select_parents(population, 2)
            if random.random() < parenthood_probability:
                #print("mating")
                child = toolbox.mate(toolbox, parents[0], parents[1])

                # imprint child with generation number
                child[8] = generation
                children.append(child)
            #else:
                #print("not mating")

        # 3) create mutants
        mutants = []
        #print(" printing elite and mutants")
        for i, ind in enumerate(elite):
            mutant = toolbox.mutate(toolbox, ind, 1)
            mutant[8] = generation
            #print(i)
            #print_ind(ind)
            #print_ind(mutant)
            del mutant.fitness.values
            mutants.append(mutant)
        #print(" printing elite and mutants - end")

        # 4) calculate fitness for children and mutants
        offspring = children + mutants
        #max_time = int(10 + generation*2)
        max_time = 68
        print("calculating fitness of the children and mutants with max time : ", max_time, " s")
        for i, ind in enumerate(offspring):
            #print("child: ", i)
            ind.fitness.values = toolbox.evaluate2(ind, max_time, training_dataset, validating_dataset,
                                                       testing_dataset, training_labels, validating_labels,
                                                       testing_labels)
            print("offspring : ", i, end='')
            print("   fitness : {:5f}".format(ind.fitness.values[0]))
            print("------------------------------------------------")

        # 4) additional training for elite
        max_time = 2
        print("additional training for elite : ", max_time," s")
        for i, ind in enumerate(elite):
            #print("elite : ", i)
            ind.fitness.values = toolbox.evaluate2(ind, max_time, training_dataset, validating_dataset,
                                                       testing_dataset, training_labels, validating_labels,
                                                       testing_labels)
            print("elite: ", i, end='')
            print("   fitness : {:5f}".format(ind.fitness.values[0]))


        # merge children with elite and mutants
        population = elite + offspring

        # list intermediate population
        print("intermediate population", len(population))
        print_pop(population)
        print("intermediate population - end")

        time_used = time.time() - start_time
        print("+++ time used {:.5f} +++".format(time_used))

    # list final population
    print("final population", len(population))
    print_pop(population)
    print("final population - end")

    time_used = time.time() - start_time
    print("+++ time used {:.5f} +++" .format(time_used))




def main_with_fast_evaluation():
    print()

def main_with_initial_population():
    print()
    # import initial population

def main_with_initial_population_and_fast_evaluation():
    print()


# call main function
#main_standard_recalculate_elite()
#main_standard()
#main_additional_training_for_elite()
main_additional_training_for_elite_and_mutation()
