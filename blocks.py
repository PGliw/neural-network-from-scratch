import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_der(x):
    return sigmoid(x) * (1 - sigmoid(x))


def softmax(x):
    exp_x = np.exp(x)
    return exp_x / exp_x.sum()


def mean_squared_error(y_pred, y_true):
    pass


class Layer:
    def __init__(self, neurons_number: int, input_size: int, activation_function, weightss=None, biases=None):
        # self.neurons_number = neurons_number
        self.activation_function = activation_function
        self.input_size = input_size
        self.neurons_number = neurons_number
        self.neurons = []
        for n in range(neurons_number):
            self.neurons.append(Neuron(input_size, activation_function))

    def predict(self, xs):
        output = []
        for neuron in self.neurons:
            output.append(neuron.predict(xs))
        return np.array(output)


class Neuron:
    def __init__(self, input_size: int, activation_function, weights=None, bias=None, random_seed=37):
        """
        :param input_size: size of input data
        :param activation_function: activation function: ex. lambda:x -> sigmoid(x)
        :param weights: np.array of weights for the neuron
        :param bias: int - bias for the neuron
        :param random_seed: int, used to initialize weights and bias
        """
        self.input_size = input_size
        self.activation_function = activation_function
        np.random.seed(random_seed)
        self.weights = np.random.rand(input_size, 1) if weights is None else weights
        self.bias = np.random.rand(1) if bias is None else bias

    def produce(self, xs):
        return np.dot(xs, self.weights)+self.bias

    def predict(self, xs):
        """
        :param xs: nd.array of size self.input_size
        :return: one value between 0 and 1
        """
        return self.activation_function(self.produce(xs))


class Model:
    def __init__(self, layers_list):
        self.layers_list = layers_list

    def fit(self, x_train, y_train, epochs_number: int):
        """
        :param x_train: array of 42 000 1296-elem-lists
        :param y_train: array of 42 000 true labels
        :param epochs_number: number of epochs
        :return:
        """
        for epoch in range(epochs_number):
            #   feed - forward
            inputs = x_train
            for layer in self.layers_list:
                inputs = layer.predict(inputs)
            outputs = inputs  # must be 10d vector
            error = outputs - y_train
            #   compare outputs vs y_train


layer = Layer(10, 10, lambda x: sigmoid(x))
prediction = layer.predict([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(prediction)
