import pickle as pkl
import numpy as np
from tensorflow import keras, nn


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
model = keras.Sequential([
    keras.layers.Dense(128, activation=nn.relu),
    keras.layers.Dense(10, activation=nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
total_loss, test_acc = model.evaluate(x_val, y_val)
print(total_loss, test_acc)