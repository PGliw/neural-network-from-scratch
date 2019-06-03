import pickle as pkl
import numpy as np
import blocks

PICKLE_FILE_PATH = 'train.pkl'
SAVED_FILE_PATH = 'params.pkl'
LEARNING_BATCH_SIZE = 40000


def load_data():
    with open(PICKLE_FILE_PATH, 'rb') as f:
        return pkl.load(f)


def save_hyper_params(model_hyper_params):
    pickle_out = open(SAVED_FILE_PATH, "wb")
    pkl.dump(model_hyper_params, pickle_out)
    pickle_out.close()


def load_hyper_params():
    pickle_in = open(SAVED_FILE_PATH, "rb")
    return pkl.load(pickle_in)

# data = load_data()

def divide_data(data_to_divide):
    """
    :param data_to_divide: np array with data
    :return: tuple (x_train, y_train, x_val, y_val)
    """
    VALIDATE_BATCH_SIZE = 18000
    xs = data_to_divide[0]
    ys = data_to_divide[1]
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


def train_model(whole_data, batch_size):
    """
    :param whole_data: train and test data
    :return: list of model parameters - pairs (weights, biases) for each layer
    """
    x_train_raw, y_train_raw, x_val_raw, y_val_raw = divide_data(whole_data)
    x_train = reshape_x_data(x_train_raw)
    x_val = reshape_x_data(x_val_raw)
    y_train = ys_to_one_hot(y_train_raw)
    y_val = ys_to_one_hot(y_val_raw)

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

    model.fit(x_train[:batch_size], y_train[:batch_size], epochs_number=10, learning_rate=0.1)

    post_preds = []
    for x in x_val[0:10]:
        post_preds.append(np.argmax(model.predict(x)))

    pre_accuracy = np.sum(pre_preds == y_val_raw[0:10])
    post_accuracy = np.sum(post_preds == y_val_raw[0:10])
    print(pre_preds, pre_accuracy, "\n", post_preds, post_accuracy, "\n", y_val_raw[0:10])

    return model.get_hyper_params()


# hyper_params = train_model(data, LEARNING_BATCH_SIZE)
# save_hyper_params(hyper_params)

"""
x_train_raw, y_train_raw, x_val_raw, y_val_raw = divide_data(data)
x_val = reshape_x_data(x_val_raw)


loaded_hyper_params = load_hyper_params()
weights_l1, biases_l1 = loaded_hyper_params[0]
weights_l2, biases_l2 = loaded_hyper_params[1]

model2 = blocks.Model(
    layers_list=[
        blocks.Layer(1296,
                     128,
                     lambda x: blocks.sigmoid(x),
                     lambda x:blocks.sigmoid_der(x),
                     weights=weights_l1,
                     biases=biases_l1),
        blocks.Layer(128,
                     10,
                     lambda x: blocks.sigmoid(x),
                     lambda x: blocks.sigmoid_der(x),
                     weights=weights_l2,
                     biases=biases_l2)
    ],
    cost_function=lambda y_pred, y_true: blocks.mean_squared_error(y_pred, y_true),
    cost_function_der=lambda y_pred, y_true: blocks.mean_squared_error_der(y_pred, y_true)
)


predictions = model2.batch_predict(x_val[:2500])
print(predictions, y_val_raw[:2500])
accuracy = np.sum(predictions == y_val_raw[:2500])/2500
print(accuracy)
"""