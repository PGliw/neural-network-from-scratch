# --------------------------------------------------------------------------
# ------------  Metody Systemowe i Decyzyjne w Informatyce  ----------------
# --------------------------------------------------------------------------
#  Zadanie 4: Zadanie zaliczeniowe
#  autorzy: A. Gonczarek, J. Kaczmar, S. Zareba, M Zieba
#  2019
# --------------------------------------------------------------------------

import pickle as pkl
import numpy as np

import blocks
from content import load_hyper_params
from content import reshape_x_data


def predict(x):
    """
    Funkcja pobiera macierz przykladow zapisanych w macierzy X o wymiarach NxD i zwraca wektor y o wymiarach Nx1,
    gdzie kazdy element jest z zakresu {0, ..., 9} i oznacza znak rozpoznany na danym przykladzie.
    :param x: macierz o wymiarach NxD
    :return: wektor o wymiarach Nx1
    """
    loaded_hyper_params = load_hyper_params()
    weights_l1, biases_l1 = loaded_hyper_params[0]
    weights_l2, biases_l2 = loaded_hyper_params[1]
    x_reshaped = reshape_x_data(x)

    model2 = blocks.Model(
        layers_list=[
            blocks.Layer(1296,
                         128,
                         lambda x: blocks.sigmoid(x),
                         lambda x: blocks.sigmoid_der(x),
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

    predictions = model2.batch_predict(x_reshaped)
    return predictions
