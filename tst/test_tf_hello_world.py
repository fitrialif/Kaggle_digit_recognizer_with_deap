import unittest
import tf_hello_world


class BaseUnitTest(unittest.TestCase):
    def setUp(self):
        print(self._testMethodDoc)


class TestEvaluate(BaseUnitTest):

    def test_evaluate_individual_fast(self):
        """testing evaluate.evaluate_individual_fast"""
        tf_hello_world.tf_hello_world2()
