import unittest
import evaluate
import data_load
import deap_functions
import time
import random

class BaseUnitTest(unittest.TestCase):
    def setUp(self):
        print(self._testMethodDoc)


class TestEvaluate(BaseUnitTest):

    def test_evaluate_individual_fast(self):
        """testing evaluate.evaluate_individual_fast"""

        # create toolbox using deap_functions
        toolbox = deap_functions.create_toolbox()

        # loading data
        # input_file = '../data/sample.csv'
        input_file = '../data/train.csv'
        image_size = 28
        training_dataset, testing_dataset, validating_dataset, training_labels, testing_labels, validating_labels = \
            data_load.data_load(input_file, image_size)

        # create one individual
        individual = toolbox.individual()

        # run evaluation
        accuracy = evaluate.evaluate_individual_fast(individual, training_dataset, validating_dataset, testing_dataset,
                                                     training_labels, validating_labels, testing_labels)

        print(accuracy)
        self.assertTrue(hasattr(toolbox, 'individual'))

    def test_evaluate_individual_fast_several_times(self):
        """testing evaluate.evaluate_individual_fast"""

        # create toolbox using deap_functions
        toolbox = deap_functions.create_toolbox()

        # loading data
        input_file = '../data/train.csv'
        image_size = 28
        training_dataset, testing_dataset, validating_dataset, training_labels, testing_labels, validating_labels = \
            data_load.data_load(input_file, image_size)

        output = open("../data/initial_population2.txt", "w+")

        for i in range(3000):

            start_time = time.time()
            # create one individual
            individual = toolbox.individual()

            # run evaluation
            accuracy = evaluate.evaluate_individual_fast(individual, training_dataset, validating_dataset,
                                                         testing_dataset, training_labels, validating_labels,
                                                         testing_labels)

            print("--- run --- ", i, " --- ", accuracy)
            # print("--- run --- ", i, " --- ", accuracy[0])

            if accuracy[0] > 0.3:
                print("writing to file")
                output.write(str(accuracy[0]))

                for x in range(8):
                    output.write(",")
                    output.write(str(individual[x]))

                output.write("\r\n")
                output.flush()
            time_used = time.time() - start_time

            print("time used for this loop: {:.5f}" .format(time_used))
        output.close()

    def test_evaluate_individual_fast_several_times2(self):
        """testing evaluate.evaluate_individual_fast"""

        # create toolbox using deap_functions
        toolbox = deap_functions.create_toolbox()

        output = open("../data/initial_population.txt", "w+")

        for x in range(100000):
            # create one individual
            individual = toolbox.individual()
            output.write(str(individual[0]))

            for i in range(1, 8):

                output.write(",")
                output.write(str(individual[i]))

            output.write("\r\n")
            output.flush()

        output.close()

    def test_play_with_population2(self):
        """testing evaluate.evaluate_individual_fast"""

        # create toolbox using deap_functions
        toolbox = deap_functions.create_toolbox()

        population_size = 10
        parenthood_probability = 0.5
        pop = toolbox.population(population_size)

        print(type(pop))
        print(type(pop[0]))
        print(random.random())
        print([random.random()])

        for ind in pop:
            ind.fitness.values = [random.random()]

        for ind in pop:
            print("fitness", ind.fitness.values, ind)

        k = int(len(pop)*.2)
        elite = toolbox.select_elite(pop, k)
        print("--------")
        for ind in elite:

            print("fitness", ind.fitness.values, ind)

        children = []
        while len(children) < population_size -len(elite):
            parents = toolbox.select_parents(pop, 2)
            if random.random() < parenthood_probability:
                child = toolbox.mate(toolbox, parents[0], parents[1])
                children.append(child)

        print(" ----- children -----")
        for ind in children:
            print(ind)

        pop = elite + children

        print(" ---- new population ----")
        for ind in pop:
            print(ind.fitness.values, ind)