import numpy as np
from content import load_data
from content import load_hyper_params
from content import divide_data
from content import reshape_x_data
from predict import predict

data = load_data()
x_train_raw, y_train_raw, x_val_raw, y_val_raw = divide_data(data)

predictions = predict(x_val_raw)
print(predictions, y_val_raw)
accuracy = np.sum(predictions == y_val_raw)/len(y_val_raw)
print(accuracy)
