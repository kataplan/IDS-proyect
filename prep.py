# Pre-processing...

import pandas as pd
import numpy as np

# Load data
def load_data(file_name:str):
    file_data = np.genfromtxt(file_name, dtype=float, delimiter=",")
    x = file_data[:,range(len(file_data[0])-2)]
    y = file_data[:, len(file_data[0])-1]
    return ( x,y )
# Normazationg of the features


def norma_data(df: np.ndarray):
    #move in columns
    for i in range(len(df[0])):
        min = df[i].min()
        max = df[i].max()
        b = 0.99
        a = 0.01
        j = 0
        if (max != min):
            #move in rows
            for x in df[:,i]:                
                df[j,i] = ((x - min) / (max - min)) * (b - a) + a
                j = j + 1
    return (df)
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
    
config = np.genfromtxt("cnf_sv.csv", dtype=int, delimiter=",")
singularity_vector = config[3]
train_x, train_y = load_data("dtrn.csv")
test_x, test_y = load_data("dtst.csv")

train_y = label_binary(train_y,"ytrn.csv")
test_y = label_binary(test_y, "ytst.csv")

index = np.genfromtxt("index.csv", dtype=int, delimiter=",")
train_x = train_x[:,index]
test_x = test_x[:,index]
filter_v = np.genfromtxt("filter_v.csv", dtype=float, delimiter=",")

train_x = np.matmul(filter_v[:,:singularity_vector].T , train_x.T).T
test_x = np.matmul(filter_v[:,:singularity_vector].T , test_x.T).T

train_x = norma_data(train_x)
test_x = norma_data(test_x)

np.savetxt("xtrn.csv",train_x,delimiter=",")
np.savetxt("xtst.csv",train_x,delimiter=",")
