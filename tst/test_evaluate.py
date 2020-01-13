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

    def test_train_individual_fast(self):

        """testing evaluate.train_individual_fast"""

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
        accuracy = evaluate.train_individual_fast(individual, training_dataset, validating_dataset,
                                                            testing_dataset, training_labels, validating_labels,
                                                            testing_labels)

        print("accuracy from training: ", accuracy)
        #print("type of model :", type(protobuf))
        import sys

        #print(protobuf)
        #self.assertTrue(hasattr(toolbox, 'individual'))
        #print("size of model : ", sys.getsizeof(protobuf))
        #print("deep size of the the model : ", deep_getsizeof(protobuf, set()))

        # use model in protobuf to evaluate individual
        accuracy = evaluate.evaluate_individual(individual, testing_dataset, testing_labels)
        print("accuracy from evaluation: ", accuracy)
        print("accuracy from evaluation: ", accuracy)
        print("accuracy from evaluation: ", accuracy)


    def test_train_individual_fast_0_97(self):

        """testing evaluate.train_individual_fast"""

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
        xx = [6822, 0.06, 0.013, 5, 16, 7, 17, 5]  # 0.97
        #x = [3917, 0.089, 0.032, 7, 13, 5, 14, 7]  # 0.84
        #x = [4099, 0.105, 0.042, 9, 3, 11, 25, 2]  # 0.42
        print(xx)
        print(individual)

        for i, elem in enumerate(xx):
            #print(i)
            individual[i] = xx[i]


        print(individual)
        print(individual[10])

        # run evaluation
        accuracy = evaluate.train_individual_fast(individual, training_dataset, validating_dataset,
                                                            testing_dataset, training_labels, validating_labels,
                                                            testing_labels)

        print("accuracy from training: ", accuracy)
        #print("type of model :", type(protobuf))
        import sys

        #print(protobuf)
        #self.assertTrue(hasattr(toolbox, 'individual'))
        #print("size of model : ", sys.getsizeof(protobuf))
        #print("deep size of the the model : ", deep_getsizeof(protobuf, set()))

        # use model in protobuf to evaluate individual
        accuracy = evaluate.evaluate_individual_2(individual, testing_dataset, testing_labels)
        print("accuracy from evaluation: ", accuracy)
        #print("accuracy from evaluation: ", accuracy)
        #print("accuracy from evaluation: ", accuracy)








    def test_train_individual_fast_0_97_and_retrain(self):

        """testing evaluate.train_individual_fast and retrain it again a bit"""

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
        #xx = [6822, 0.06, 0.013, 5, 16, 7, 17, 5]  # 0.97
        #x = [3917, 0.089, 0.032, 7, 13, 5, 14, 7]  # 0.84
        xx = [4099, 0.105, 0.042, 9, 3, 11, 25, 2]  # 0.42
        print(xx)
        print(individual)

        for i, elem in enumerate(xx):
            #print(i)
            individual[i] = xx[i]


        print(individual)
        print(individual[10])

        # run training 1
        accuracy = evaluate.train_individual_fast(individual, training_dataset, validating_dataset,
                                                            testing_dataset, training_labels, validating_labels,
                                                            testing_labels)
        print("accuracy from training 1 : ", accuracy)


        # use model in protobuf to evaluate individual after first training
        accuracy = evaluate.evaluate_individual_2(individual, testing_dataset, testing_labels)
        print("accuracy from evaluation 1: ", accuracy)



        for i in range(10):
            accuracy = evaluate.train_individual2(individual, 10, training_dataset, validating_dataset,
                                                  testing_dataset, training_labels, validating_labels,
                                                  testing_labels)
            print("accuracy from training ",i , " : ", accuracy)


            # use model in protobuf to evaluate individual after first training
            accuracy = evaluate.evaluate_individual_2(individual, testing_dataset, testing_labels)
            print("accuracy from evaluation ",i , " : ",  accuracy)











    def test_train_individual_fast_several_times(self):
        """testing evaluate.train_individual_fast"""

        # create toolbox using deap_functions
        toolbox = deap_functions.create_toolbox()

        # loading data
        input_file = '../data/train.csv'
        image_size = 28
        training_dataset, testing_dataset, validating_dataset, training_labels, testing_labels, validating_labels = \
            data_load.data_load(input_file, image_size)

        output = open("../data/initial_population2.txt", "w+")

        for i in range(10):

            start_time = time.time()
            # create one individual
            individual = toolbox.individual()

            # run evaluation
            accuracy, protobuf, metagraph, x, y_, save_path = evaluate.train_individual_fast(individual, training_dataset, validating_dataset,
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





    def evaulate_trained_individual(self):

        """testing evaluate.train_individual_fast"""

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

        # run training
        accuracy, protobuf = evaluate.train_individual_fast(individual, validating_dataset, validating_labels)

        # run evaluation in order to check if accuracy of the individual is equal to accuracy
        # calculated as part of the training
        accuracy = evaluate.evaluate_individual(individual,protobuf, testing_dataset, testing_labels)


        print("accuracy from training: ", accuracy)
        print("type of model :", type(protobuf))
        import sys

        print(protobuf)
        self.assertTrue(hasattr(toolbox, 'individual'))
        print("size of model : ", sys.getsizeof(protobuf))
        print("deep size of the the model : ", deep_getsizeof(protobuf, set()))

        # use model in protobuf to evaluate individual
        accuracy = evaluate.evaluate_individual(individual, protobuf, testing_dataset, testing_labels)
        print("accuracy from evaluation: ", accuracy)




    # TODO: this function does not really evaluate individual fast but just create tons of individuals - fix it
    def test_train_individual_fast_several_times2(self):
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


from collections import Mapping, Container
from sys import getsizeof
# Find the memory footprint of a Python object
#
# This is a recursive function that drills down a Python object graph
# like a dictionary holding nested dictionaries with lists of lists
# and tuples and sets.
#
# The sys.getsizeof function does a shallow size of only. It counts each
# object inside a container as pointer only regardless of how big it
# really is.
# :param o: the object
# :param ids:
# :return:
def deep_getsizeof(o, ids):

    d = deep_getsizeof
    if id(o) in ids:
        return 0

    r = getsizeof(o)
    ids.add(id(o))

    if isinstance(o, str) or isinstance(0, str):
        return r

    if isinstance(o, Mapping):
        return r + sum(d(k, ids) + d(v, ids) for k, v in o.iteritems())

    if isinstance(o, Container):
        return r + sum(d(x, ids) for x in o)

    return r
