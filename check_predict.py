import numpy as np

from content import divide_data
from content import load_data
from predict import predict

data = load_data()
x_train_raw, y_train_raw, x_val_raw, y_val_raw = divide_data(data)
x_val = np.array(x_train_raw)
print(type(x_val))
print(x_val.shape)
predictions = predict(x_val)
print(predictions)
# accuracy = np.sum(predictions == y_val_raw)/len(y_val_raw)
# print(accuracy)
