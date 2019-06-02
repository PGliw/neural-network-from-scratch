import unittest
import blocks
import sample
import content
import numpy as np

"""
class TestNeuronMethods(unittest.TestCase):
    def setUp(self):
        self.neuron = blocks.Neuron(3, lambda x: blocks.sigmoid(x), [3, 2, 1], 1, 37)

    def test_activation_fun(self):
        self.assertEqual(self.neuron.activation_function(0), 0.5, "incorrect activation function")

    def test_randoms(self):
        neuron_rand = blocks.Neuron(5, lambda x: blocks.sigmoid(x))
        self.assertEqual(len(neuron_rand.weights), 5)

    def test_produce(self):
        self.assertEqual(self.neuron.produce([1, 2, 3]), 11)

    def test_predict(self):
        self.assertAlmostEqual(self.neuron.predict([1, 2, 3]), 0.99998329)


class TestLayer(unittest.TestCase):
    def setUp(self):
        self.layer = blocks.Layer(2, 3, lambda x: blocks.sigmoid(x),
                                  lambda x: blocks.sigmoid_der(x),

                                  )

"""

class TestDataUnpacking(unittest.TestCase):
    def test_ys_to_one_hot(self):
        self.assertEqual(content.ys_to_one_hot([1, 2, 3, 0, 3]).shape[0], 5)
        self.assertEqual(content.ys_to_one_hot([1, 2, 3, 0, 3]).shape[1], 4)

        self.assertEqual(
            content.ys_to_one_hot([1, 2, 3, 0, 4]).all(),
            np.all(
                [[0, 1, 0, 0],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1],
                 [1, 0, 0, 0],
                 [0, 0, 0, 1]])
        )

    def test_reshape_x_data(self):
        self.assertEqual(content.reshape_x_data(
                [[1, 2, 3, 4],
                 [5, 6, 7, 8],
                 [9, 10, 11, 12]]
            ).shape[1], 3)
        self.assertEqual(content.reshape_x_data(
            [[1, 2, 3, 4],
             [5, 6, 7, 8],
             [9, 10, 11, 12]]
        ).shape[2], 4)
        self.assertEqual(
            content.reshape_x_data(
                [[1, 2, 3, 4],
                 [5, 6, 7, 8],
                 [9, 10, 11, 12]]
            ).all(),
            np.all(
                [[[1, 2, 3, 4]],
                 [[5, 6, 7, 8]],
                 [[9, 10, 11, 12]]]
            )
        )
