import numpy as np
import blocks

# training data
x_train = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
y_train = np.array([[0], [1], [1], [0]])

x_dummy = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_dummy = [0, 1, 1, 0]
x_re = np.reshape(x_dummy, (len(x_dummy), 1 , len(x_dummy[0])))
y_re = np.reshape(y_dummy, (1, len(y_dummy)))
print("x_dummy")
print(x_dummy)
print("x_re")
print(x_re)
print("x_train")
print(x_train)
#print(y_dummy)
#print(y_re)
#print(y_re.flatten())

"""
def sigmoid(x):
    return lambda: blocks.sigmoid(x)


def sigmoid_der(x):
    return lambda: blocks.sigmoid_der(x)
"""

model = blocks.Model(
    layers_list=[
        blocks.Layer(2,
                     3,
                     lambda x: blocks.sigmoid(x),
                     lambda x:blocks.sigmoid_der(x)),
        blocks.Layer(3,
                     1,
                     lambda x: blocks.sigmoid(x),
                     lambda x: blocks.sigmoid_der(x))
    ],
    cost_function=lambda y_pred, y_true: blocks.mean_squared_error(y_pred, y_true),
    cost_function_der=lambda y_pred, y_true: blocks.mean_squared_error_der(y_pred, y_true)
)

model.fit(x_train, y_train, epochs_number=1000, learning_rate=0.9)
print(model.predict([1, 1]))
