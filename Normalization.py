import numpy as np
import pandas as pd



def readFile(string):
    file_data = np.genfromtxt(string, dtype=str, delimiter=",")
    return file_data

def reduce_data(file_data, data_limit, class_1_bool,class_2_bool, class_3_bool):
    
    
    return file_data

def classify(file_data):
    count = 0
    class_1 = ['normal']
    class_2 = ['neptune', 'teardrop', 'smurf', 'pod', 'back', 'land', 'apache2', 'processtable', 'mailbomb', 'udpstorm']
    class_3 = ['ipsweep', 'portsweep', 'nmap', 'satan', 'saint', 'mscan']
    for i in file_data[:, 41]:
    
        if i in class_1:
            file_data[count, 41] = 1    
            
        elif i in class_2:
            file_data[count, 41] = 2
            
        elif i in class_3:
            file_data[count, 41] = 3
        else:
            file_data = np.delete(file_data, count, axis=0)
            count = count - 1

        count = count + 1
    
    return file_data

def singularity_array(file_data, column):
    array = []
    for i in file_data[:, column]:
        if i not in array:
            array.insert(len(array), i)

    return array

def transform_categoric_variables(file_data, column):
    array = singularity_array(file_data, column)
    count = 0
    for i in file_data[:, column]:
        file_data[count, column] = array.index(i)
        count = count +1
    return file_data

def matrix_to_integer(X):
    for i in range(len(X)):
        for j in range(len(X[0])):
            X[i][j] = float(X[i][j])
    return X
def findMax(file_data, column):
    max = 0
    for i in file_data[:, column]:
        if( float(i) > max ):
            max = float(i)
    return max

def findMin(file_data, column):
    min = 900000
    for i in file_data[:, column]:
        if( float(i) < min ):
            min = float(i)
    return min

def normalize_column(file_data, column):
    min = findMin(file_data, column)
    max = findMax(file_data, column)
    b = 0.99
    a = 0.01
    row=0
    for x in file_data[:,column]:
        
        file_data[row,column]= float(file_data[row,column])
        x= float(x)
        file_data[row,column] = ( ( x - min ) / ( max - min )) * ( b - a ) + a 
        row = row + 1
    return file_data

def normalize_data(file_data):
    for i in range(0 , 40):
        if(i == 8):
            return file_data
        file_data = normalize_column(file_data,i)
    return file_data



