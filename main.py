import pickle as pkl
import numpy as np
from tensorflow import keras, nn
import blocks


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


def ys_to_one_hot(ys):
    inputs = np.array(ys)
    one_hot_size = len(set(inputs))
    outputs = np.zeros((len(ys), one_hot_size))
    outputs[np.arange(len(ys)), inputs] = 1
    return outputs

def reshape_x_data(x_data):
    return np.reshape(x_data, (len(x_data), 1, len(x_data[0])))

x_train_raw, y_train_raw, x_val_raw, y_val_raw = divide_data(data)
x_train = reshape_x_data(x_train_raw)
x_val = reshape_x_data(x_val_raw)
y_train = ys_to_one_hot(y_train_raw)
y_val = ys_to_one_hot(y_val_raw)



"""
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
"""

model = blocks.Model(
    layers_list=[
        blocks.Layer(1296,
                     128,
                     lambda x: blocks.sigmoid(x),
                     lambda x:blocks.sigmoid_der(x)),
        blocks.Layer(128,
                     10,
                     lambda x: blocks.sigmoid(x),
                     lambda x: blocks.sigmoid_der(x))
    ],
    cost_function=lambda y_pred, y_true: blocks.mean_squared_error(y_pred, y_true),
    cost_function_der=lambda y_pred, y_true: blocks.mean_squared_error_der(y_pred, y_true)
)

pre_preds = []
for x in x_val[0:10]:
    pre_preds.append(np.argmax(model.predict(x)))

model.fit(x_train[:1000], y_train[:1000], epochs_number=10, learning_rate=0.1)

post_preds = []
for x in x_val[0:10]:
    post_preds.append(np.argmax(model.predict(x)))

print(pre_preds, "\n", post_preds, "\n", y_val_raw[0:10])
