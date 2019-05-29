import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_der(x):
    return sigmoid(x) * (1 - sigmoid(x))


def softmax(x):
    exp_x = np.exp(x)
    return exp_x / exp_x.sum()


def mean_squared_error(y_pred, y_true):
    return np.mean(np.power(y_true - y_pred, 2))


def mean_squared_error_der(y_pred, y_true):
    return 2*(y_pred-y_true)/y_true.size


class Layer:
    def __init__(self, input_size: int, neurons_number: int, activation_function, activation_function_der, weights=None, biases=None):
        """
        :param input_size: integer, size of layer's input
        :param neurons_number: number of neurons in the layer
        :param activation_function: activation function for neurons in layer
        :param weights: matrix NxM where N is number of neurons and M is number of inputs
        :param biases: 1xN array where N is number of neurons
        """
        self.input_size = input_size
        self.neurons_number = neurons_number
        self.activation_function = activation_function
        self.activation_function_der = activation_function_der
        self.neurons_number = neurons_number
        # self.weights = np.random.rand(input_size, neurons_number) if weights is None else weights
        self.weights = np.random.uniform(-0.5, 0.5, (input_size, neurons_number)) if weights is None else weights
        # self.bias = np.random.rand(1, neurons_number) if biases is None else biases
        self.bias = np.random.uniform(-0.5, 0.5, (1, neurons_number)) if biases is None else biases
        self.xs = np.zeros([input_size, 1])
        self.xs_mul_ws = np.zeros([input_size, 1])

    def predict(self, xs):
        self.xs = xs
        product = np.dot(xs, self.weights) + self.bias
        self.xs_mul_ws = product
        return self.activation_function(product)

    def propagate_back(self, error_out, learning_rate):
        error_pre_activation = self.activation_function_der(self.xs_mul_ws) * error_out
        #   error back propagation
        error_in = np.dot(error_pre_activation, self.weights.T)    # T = transpose
        error_weights = np.dot(self.xs.T, error_pre_activation)
        #   adjust parameters
        self.learn(error_weights, error_pre_activation, learning_rate)
        #   propagate the error
        return error_in

    def learn(self, error_weights, error_out, learning_rate):
        self.weights -= learning_rate * error_weights
        self.bias -= learning_rate * error_out


class Model:
    def __init__(self, layers_list, cost_function, cost_function_der):
        self.layers_list = layers_list
        self.cost_function = cost_function
        self.cost_function_der = cost_function_der

    def predict(self, x):
        """
        :param x: D x 1 vector of features
        :return: M x 1 vector of results
        """
        # predictions for one piece of data
        prediction = x
        for layer in self.layers_list:
            #   predict and pass result as an input for the next layer
            prediction = layer.predict(prediction)
        #   append the final prediction for vector x
        return prediction

    def fit(self, x_train, y_train, epochs_number: int, learning_rate):
        """
        :param learning_rate: alpha parameter for weights correction in one step (float)
        :param x_train: array of 42 000 1296-elem-lists
        :param y_train: array of 42 000 true labels
        :param epochs_number: number of epochs
        :return:
        """
        for epoch in range(epochs_number):
            data_pieces_errors = []
            for i in range(len(x_train)):
                #   predict
                prediction = self.predict(x_train[i])
                #   calc error
                error_for_data_piece = self.cost_function(prediction, y_train[i])
                data_pieces_errors.append(error_for_data_piece)
                #   propagate the error back and learn
                error = self.cost_function_der(prediction, y_train[i])
                for layer in reversed(self.layers_list):
                    error = layer.propagate_back(error, learning_rate)
            average_epoch_error = np.mean(data_pieces_errors)
            print("epoch {} error={}".format(epoch, average_epoch_error))

    def get_hyper_params(self):
        """
        :return: list of tuples (weights, bias) for each layer
        """
        hyper_params_list = []
        for layer in self.layers_list:
            hyper_params_list.append((layer.weights, layer.bias))
        return hyper_params_list
