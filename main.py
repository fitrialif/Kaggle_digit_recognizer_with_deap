# imports
import deap_functions
import data_load
import random
import time

# this function
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
    population_size = 20
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
    for generation in range(200):
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
                children.append(child)
            #else:
                #print("not mating")

        # calculate fitness for children
        print("calculating fitness of the children with max time : ", int(generation/2+10))
        max_time = int(10 + generation/2)
        for i, ind in enumerate(children):
            print("child: ", i, end='')
            ind.fitness.values = toolbox.evaluate(ind, max_time, training_dataset, validating_dataset,
                                                       testing_dataset, training_labels, validating_labels,
                                                       testing_labels)
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
    time_used = time.time() -start_time
    print("+++ time used {:.5f} +++" .format(time_used))

    #condition to finish calculation is nuber of generations or total time


def main_with_fast_evaluation():
    print()

def main_with_initial_population():
    print()
    # import initial population

def main_with_initial_population_and_fast_evaluation():
    print()


# call main function
main_standard()
