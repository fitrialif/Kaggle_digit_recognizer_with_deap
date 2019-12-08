import unittest
import deap_functions
import numpy as np
import deap
import data_load


class BaseUnitTest(unittest.TestCase):
    def setUp(self):
        print(self._testMethodDoc)


class TestReformat(BaseUnitTest):
    # test if reformat works correctly by returning shape (2, 3, 3, 1)
    def test_reformat(self):
        """testing deap_functions.reformat if reformat works"""
        test_dataset = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9]])
        self.assertEqual(data_load.reformat(test_dataset, image_size=3).shape, (2, 3, 3, 1))
        self.assertNotEqual(data_load.reformat(test_dataset, 3).shape, (2, 3, 3, 2))


class TestPadding(BaseUnitTest):
    # test if padding of 2 zeros is added
    def test_padding(self):
        """testing deap_functions.padding if padding works"""
        # this is how the array looks like before
        source_dataset = np.array([[[[1], [2], [3]],
                                    [[4], [5], [6]],
                                    [[7], [8], [9]]],
                                   [[[1], [2], [3]],
                                    [[4], [5], [6]],
                                    [[7], [8], [9]]]])
        # this is how the array should look like after
        result_dataset = np.array([[[[0], [0], [0], [0], [0], [0], [0]],
                                    [[0], [0], [0], [0], [0], [0], [0]],
                                    [[0], [0], [1], [2], [3], [0], [0]],
                                    [[0], [0], [4], [5], [6], [0], [0]],
                                    [[0], [0], [7], [8], [9], [0], [0]],
                                    [[0], [0], [0], [0], [0], [0], [0]],
                                    [[0], [0], [0], [0], [0], [0], [0]]],
                                   [[[0], [0], [0], [0], [0], [0], [0]],
                                    [[0], [0], [0], [0], [0], [0], [0]],
                                    [[0], [0], [1], [2], [3], [0], [0]],
                                    [[0], [0], [4], [5], [6], [0], [0]],
                                    [[0], [0], [7], [8], [9], [0], [0]],
                                    [[0], [0], [0], [0], [0], [0], [0]],
                                    [[0], [0], [0], [0], [0], [0], [0]]]])

        self.assertTrue((data_load.padding(source_dataset, padding_no=2) == result_dataset).all())


class TestRandomOddInt(BaseUnitTest):
    def test_random_odd_int_if_odd(self):
        """testing deap_functions.random_odd_int if result modulo by 2 gives rest equal 0"""
        i = 0
        while i < 100:
            self.assertNotEqual(deap_functions.random_odd_int(1, 100) % 2, 0)
            i = i + 1

    def test_random_odd_int_if_int(self):
        """testing deap_functions.random_odd_int if result returned is int """
        i = 0
        while i < 10:
            self.assertIsInstance(deap_functions.random_odd_int(1, 100), int)
            i = i + 1

    def test_random_odd_int_if_within_range(self):
        """testing deap_functions.random_odd_int if result returned is within range 1 to 100"""
        i = 0
        while i < 10:
            self.assertTrue(1 <= deap_functions.random_odd_int(1, 100) <= 100)
            i = i + 1


class TestRandomRoundedFloat(BaseUnitTest):
    def test_random_rounded_float_if_rounded(self):
        """testing deap_functions.random_rounded_float if result rounded correctly"""
        while True:
            x = deap_functions.random_rounded_float(1, 100, 4)
            y = round(x, 4)
            z = round(x, 3)
            if y != z:
                break
        self.assertEqual(x, y)
        self.assertAlmostEqual(x, y)
        self.assertNotEqual(x, z)

    def test_random_rounded_float_if_float(self):
        """testing deap_functions.random_rounded_float if result is float"""
        self.assertIsInstance(deap_functions.random_rounded_float(1, 100, 4), float)


class TestBlxAlphaInt(BaseUnitTest):
    def test_blx_alpha_int_if_within_range(self):
        """testing deap_functions.blx_alpha_int if within range"""
        p1 = 18
        p2 = 78
        min_val = 10
        max_val = 100
        alpha = 0.1
        gamma = deap_functions.blx_alpha_int(p1, p2, alpha, min_val, max_val)
        self.assertTrue(min_val <= gamma <= max_val)

    def test_blx_alpha_int_if_int(self):
        """testing deap_functions.blx_alpha_int if int"""
        p1 = 18
        p2 = 78
        min_val = 10
        max_val = 100
        alpha = 0.1
        gamma = deap_functions.blx_alpha_int(p1, p2, alpha, min_val, max_val)
        self.assertIsInstance(gamma, int)

    def test_blx_alpha_int_if_not_negative(self):
        """testing deap_functions.blx_alpha_intif not negative"""
        p1 = 18
        p2 = 78
        min_val = 10
        max_val = 100
        alpha = 0.1
        gamma = deap_functions.blx_alpha_int(p1, p2, alpha, min_val, max_val)
        self.assertGreater(gamma, 0)


class TestBlxAlphaIntOdd(BaseUnitTest):
    def test_blx_alpha_int_odd_if_within_range(self):
        """testing deap_functions.blx_alpha_float if within range"""
        p1 = 18
        p2 = 78
        min_val = 10
        max_val = 100
        alpha = 0.1
        gamma = deap_functions.blx_alpha_int_odd(p1, p2, alpha, min_val, max_val)
        self.assertTrue(min_val <= gamma <= max_val)

    def test_blx_alpha_int_odd_if_int(self):
        """testing deap_functions.blx_alpha_float if int"""
        p1 = 18
        p2 = 78
        min_val = 10
        max_val = 100
        alpha = 0.1
        gamma = deap_functions.blx_alpha_int_odd(p1, p2, alpha, min_val, max_val)
        self.assertIsInstance(gamma, int)

    def test_blx_alpha_int_odd_if_not_negative(self):
        """testing deap_functions.blx_alpha_float if not negative"""
        p1 = 18
        p2 = 78
        min_val = 10
        max_val = 100
        alpha = 0.1
        gamma = deap_functions.blx_alpha_int_odd(p1, p2, alpha, min_val, max_val)
        self.assertGreater(gamma, 0)

    def test_blx_alpha_int_odd_if_odd(self):
        """testing deap_functions.blx_alpha_float if odd"""
        p1 = 20
        p2 = 96
        min_val = 7
        max_val = 120
        alpha = 0.1
        i = 0
        while i < 100:
            gamma = deap_functions.blx_alpha_int_odd(p1, p2, alpha, min_val, max_val)
            self.assertNotEqual(gamma % 2, 0)
            i = i + 1


class TestBlxAlphaFloat(BaseUnitTest):
    def test_blx_alpha_float_if_within_range(self):
        """testing deap_functions.blx_alpha_int_odd if within range"""
        p1 = 18.4
        p2 = 78.45
        min_val = 11.34
        max_val = 101.45
        alpha = 0.1
        gamma = deap_functions.blx_alpha_int(p1, p2, alpha, min_val, max_val)
        self.assertTrue(min_val <= gamma <= max_val)

    def test_blx_alpha_float_if_float(self):
        """testing deap_functions.blx_alpha_int_odd if float"""
        p1 = 18.34
        p2 = 78.54
        min_val = 10.1
        max_val = 199.4
        alpha = 0.1
        gamma = deap_functions.blx_alpha_int(p1, p2, alpha, min_val, max_val)
        self.assertIsInstance(gamma, int)

    def test_blx_alpha_float_if_not_negative(self):
        """testing deap_functions.blx_alpha_int_odd if not negative"""
        p1 = 18.35
        p2 = 78.45
        min_val = 11.34
        max_val = 101.45
        alpha = 0.1
        gamma = deap_functions.blx_alpha_int(p1, p2, alpha, min_val, max_val)
        self.assertGreater(gamma, 0)


class TestHyperParameters(BaseUnitTest):
    def test_create_toolbox_if_parameters_filled_in(self):
        """testing deap_functions.hyper_parameters if hyperparameters are filled in"""

        # create toolbox using deap_functions
        toolbox = deap_functions.create_toolbox()

        hyper_params = deap_functions.hyper_parameters(toolbox)
        # check if created toolbox contains all needed functions that fill in parameters
        # by testing if calling toolbox.hyper_parameters() returns object that has those parameters
        # of type int or float
        self.assertIsInstance(hyper_params[0], int)
        self.assertIsInstance(hyper_params[1], float)
        self.assertIsInstance(hyper_params[2], float)
        self.assertIsInstance(hyper_params[3], int)
        self.assertIsInstance(hyper_params[4], int)
        self.assertIsInstance(hyper_params[5], int)
        self.assertIsInstance(hyper_params[6], int)
        self.assertIsInstance(hyper_params[7], int)


class TestCreateToolbox(BaseUnitTest):
    def test_create_toolbox_if_hyperparams_methods_registered(self):
        """testing deap_functions.create_toolbox if methods for filling in hyperparameters are registered"""

        # create toolbox using deap_functions
        toolbox = deap_functions.create_toolbox()

        # check if created toolbox contains all needed functions that fill in parameters
        # by testing if those parameters are of type int or float
        self.assertTrue(hasattr(toolbox, 'attr_batch_size'))
        self.assertTrue(hasattr(toolbox, 'attr_learning_rate'))
        self.assertTrue(hasattr(toolbox, 'attr_standard_distribution'))
        self.assertTrue(hasattr(toolbox, 'attr_receptive_field_conv_1'))
        self.assertTrue(hasattr(toolbox, 'attr_filters_conv_1'))
        self.assertTrue(hasattr(toolbox, 'attr_receptive_field_conv_2'))
        self.assertTrue(hasattr(toolbox, 'attr_filters_conv_2'))
        self.assertTrue(hasattr(toolbox, 'attr_activation'))

    def test_create_toolbox_create_individual(self):
        """testing deap_functions.create_toolbox if creating individual works"""

        # create toolbox using deap_functions
        toolbox = deap_functions.create_toolbox()

        self.assertTrue(hasattr(toolbox, 'individual'))

        # create one individual
        ind = toolbox.individual()

        self.assertIsInstance(ind[0], int)
        self.assertIsInstance(ind[1], float)
        self.assertIsInstance(ind[7], int)
        self.assertIsNotNone(ind.fitness)
        self.assertIsInstance(ind.fitness, deap.creator.FitnessMax)

    def test_create_toolbox_create_population(self):
        """testing deap_functions.create_toolbox if creating population works"""

        # create toolbox using deap_functions
        toolbox = deap_functions.create_toolbox()

        self.assertTrue(hasattr(toolbox, 'population'))

        # create population
        pop = toolbox.population(2)

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


class TestGeneRange(BaseUnitTest):
    def test_gene_range(self):
        """testing deap_functions.GeneRange for class attributes"""
        self.assertEqual(deap_functions.GeneRange.attr_batch_size_min, 10)
        self.assertEqual(deap_functions.GeneRange.attr_batch_size_max, 10000)

    def test_mutate_gene_range(self):
        """testing deap_functions.GeneRange for mutation of the class attributes"""
        deap_functions.GeneRange.attr_batch_size_min = 11
        self.assertEqual(deap_functions.GeneRange.attr_batch_size_min, 11)

        # new instance of the GeneRange class
        gene_range = deap_functions.GeneRange()
        self.assertEqual(gene_range.attr_batch_size_min, 11)

        gene_range.attr_batch_size_max = 5000
        deap_functions.GeneRange.attr_batch_size_max = 10001
        self.assertEqual(gene_range.attr_batch_size_max, 5000)
        self.assertEqual(deap_functions.GeneRange.attr_batch_size_max, 10001)


class TestMateIndividuals(BaseUnitTest):
    def test_mate_individuals(self):
        """testing deap_functions.mate_individual if mating works """

        # create toolbox using deap_functions
        toolbox = deap_functions.create_toolbox()

        # create population
        pop = toolbox.population(2)

        # mate parents
        child = deap_functions.mate_individuals(toolbox, pop[0], pop[1])

        self.assertIsInstance(child[0], int)
        self.assertIsInstance(child[1], float)
        self.assertIsInstance(child[7], int)
        self.assertIsNotNone(child.fitness)
        self.assertIsInstance(child.fitness, deap.creator.FitnessMax)


class TestMutateIndividual(BaseUnitTest):
    def test_mutate_individual(self):
        """testing deap_functions.mutate_individual if mutation works"""
        print("not implemented yet - just do nothing")
