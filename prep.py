
import numpy as np
import pandas as pd
import math

def save_data(x_train: np.ndarray, y_train: np.ndarray, x_fname: str, y_fname: str) -> None:
    np.savetxt(x_fname, x_train, delimiter=",", fmt="%1.5f")
    np.savetxt(y_fname, y_train, delimiter=",", fmt="%d")

def get_binary_column(y_train) :
    y_train= pd.Series(y_train.astype(int))
    y_dummies = pd.get_dummies(y_train)
    return y_dummies.to_numpy()

def normalize_features(file_data,a,b):
    x_max = file_data.max(axis=0)
    x_min = file_data.min(axis=0)
    return ((file_data[:, :40] - x_min) * (1 / ((x_max - x_min) + math.pow(10, -8)))) * (
        b - a
    ) + a

filter_v = np.genfromtxt("filter_v.csv", delimiter=",", dtype=float)
index = np.genfromtxt("index.csv", delimiter=",", dtype=int)
index -= 1

data_train = np.genfromtxt("dtrn.csv", delimiter=",", dtype=float)
y_train= data_train[:, 41].astype(int)
y_train= get_binary_column(y_train)
x_train = data_train[:, index]
x_train = normalize_features(x_train, 0.01, 0.99)
x_train = np.matmul(x_train, filter_v)
save_data(x_train, y_train, "xtrn.csv", "ytrn.csv")

data_test = np.genfromtxt("dtst.csv", delimiter=",", dtype=float)

y_test = data_test[:, 41].astype(int)
y_test = get_binary_column(y_test)
x_test = data_test[:, index]
x_test = normalize_features(x_test, 0.01, 0.99)
x_test = np.matmul(x_test, filter_v)
save_data(x_test, y_test, "xtst.csv", "ytst.csv")
