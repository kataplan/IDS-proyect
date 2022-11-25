# Pre-processing...

import pandas as pd
import numpy as np
import math

# Load data
def load_data(file_name:str):
    file_data = np.genfromtxt(file_name, dtype=float, delimiter=",")
    x = file_data[:,range(len(file_data[0])-1)]
    y = file_data[:, len(file_data[0])-1]
    return ( x , y )
# Normazationg of the features


def norma_data(df: np.ndarray):
    b = 0.99
    a = 0.01
    x_min = df.min(axis=0)
    x_max = df.max(axis=0)
    return ((df[:, :len(df[0])] - x_min) * (1 / ((x_max - x_min) + math.pow(10, -8)))) * (b - a) + a
# Create binary label


def label_binary(df:np.ndarray,file_name:str):
    binary_array = []
    aux = []
    for x in df:
        if(x==1):
            aux = [1,0]
        if(x==2):
            aux = [0,1]

        binary_array.append(aux)
    np.savetxt(file_name,binary_array,delimiter=",",fmt='%d')
    return (binary_array)
    

train_x, train_y = load_data("dtrn.csv")
test_x, test_y = load_data("dtst.csv")

train_y = label_binary(train_y,"ytrn.csv")
test_y = label_binary(test_y, "ytst.csv")

index = np.genfromtxt("index.csv", dtype=int, delimiter=",")

train_x = norma_data(train_x)
test_x = norma_data(test_x)

index = index -1
train_x = train_x[:,index]
test_x = test_x[:,index]
filter_v = np.genfromtxt("filter_v.csv", dtype=float, delimiter=",",)

train_x = np.matmul(filter_v.T , train_x.T).T
test_x = np.matmul(filter_v.T , test_x.T).T


np.savetxt("xtrn.csv",train_x,delimiter=",",fmt="%1.5f")
np.savetxt("xtst.csv",train_x,delimiter=",",fmt="%1.5f")
