import pickle as pkl
import numpy as np


def load_data():
    PICKLE_FILE_PATH = 'train.pkl'
    with open(PICKLE_FILE_PATH, 'rb') as f:
        return pkl.load(f)


data = load_data()


def divide_data(data_to_divide):
    """
    :param data_to_divide: np array with data
    :return: tuple (x_train, y_train, x_val, y_val)
    """
    VALIDATE_BATCH_SIZE = 18000
    xs = data[0]
    ys = data[1]
    x_train = xs[:-VALIDATE_BATCH_SIZE]
    y_train = ys[:-VALIDATE_BATCH_SIZE]
    x_val = xs[-VALIDATE_BATCH_SIZE:]
    y_val = ys[-VALIDATE_BATCH_SIZE:]
    return x_train, y_train, x_val, y_val


x_train, y_train, x_val, y_val = divide_data(data)