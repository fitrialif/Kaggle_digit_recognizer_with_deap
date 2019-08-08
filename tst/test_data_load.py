import unittest
import data_load
import deap


class BaseUnitTest(unittest.TestCase):
    def setUp(self):
        print(self._testMethodDoc)


class TestDataLoad(BaseUnitTest):
    def test_data_load(self):
        """testing data_load.data_load """
        input_file = '../data/sample.csv'
        image_size = 28
        training_dataset, testing_dataset, validating_dataset, training_labels, testing_labels, validating_labels = \
            data_load.data_load(input_file, image_size)

        # just check if shape is correct
        self.assertEqual(training_dataset.shape, (80, 28, 28, 1))
        self.assertEqual(testing_dataset.shape, (4, 28, 28, 1))
        self.assertEqual(validating_dataset.shape, (16, 28, 28, 1))

        self.assertEqual(training_labels.shape, (80, 10))
        self.assertEqual(testing_labels.shape, (4, 10))
        self.assertEqual(validating_labels.shape, (16, 10))


class TestLoadInitialPopulation(BaseUnitTest):
    def test_load_initial_population(self):
        """testing data_load.load_initial_population"""

        input_file = '../data/initial_population2.txt'
        pop = data_load.load_initial_population(input_file)

        self.assertIsInstance(pop[0][0], int)
        self.assertIsInstance(pop[1][0], int)
        self.assertIsInstance(pop[0][1], float)
        self.assertIsInstance(pop[1][1], float)
        self.assertIsInstance(pop[0][7], int)
        self.assertIsInstance(pop[1][7], int)
        self.assertIsNotNone(pop[0].fitness)
        self.assertIsNotNone(pop[1].fitness)
        self.assertIsInstance(pop[0].fitness, deap.creator.FitnessMax)
        self.assertIsInstance(pop[1].fitness, deap.creator.FitnessMax)

    def test_load_initial_population_evaluate_few(self):
        """testing data_load.load_initial_population by evaluating few of them"""


