import numpy as np

def readFile(string):
    file_data = np.genfromtxt(string, dtype=str, delimiter=",")
    return file_data

#function to reduce the data given the config limiters
def reduce_data(file_data: np.ndarray, data_limit, class_1_bool,class_2_bool, class_3_bool):
    new_file_data=[]
    row=0
    data_class_1= 0
    data_class_2= 0
    data_class_3= 0
    for i in file_data[:, 41]:
        i=int(i) #parse i to match condition
        #condition for the first row added to new_file_data
        if(data_class_1+data_class_2+data_class_3 == 0):
            if(class_1_bool and i == 1):
                new_file_data = [file_data[row]]
                data_class_1= data_class_1 + 1
                
            elif(class_2_bool and i == 2):
                new_file_data = [file_data[row]]
                data_class_2= data_class_2 + 1
              
            elif(class_3_bool and i == 3):
                new_file_data = [file_data[row]] 
                data_class_3= data_class_3 + 1
        
        #check if it available class, if is the number and if bellow the data limit
        elif(class_1_bool and i == 1 and data_class_1 < data_limit):
            new_file_data = np.append(new_file_data, [file_data[row]], axis=0)
            data_class_1= data_class_1 + 1
            
        elif(class_2_bool and i == 2 and data_class_2 < data_limit):
            new_file_data = np.append(new_file_data, [file_data[row]], axis=0)
            data_class_2= data_class_2 + 1
            
        elif(class_3_bool and i == 3 and data_class_3 < data_limit):
            new_file_data = np.append(new_file_data, [file_data[row]], axis=0)
            data_class_3= data_class_3 + 1
        row = row + 1
    
    return new_file_data[:, range(0,41)], new_file_data[:,41]

def classify(file_data: np.ndarray):
    count = 0
    #data dictionary
    class_1 = ['normal']
    class_2 = ['neptune', 'teardrop', 'smurf', 'pod', 'back', 'land', 'apache2', 'processtable', 'mailbomb', 'udpstorm']
    class_3 = ['ipsweep', 'portsweep', 'nmap', 'satan', 'saint', 'mscan']
    #cycling for each data in the column 41 to change the value from de data dictionary to number value
    for i in file_data[:, 41]:
        if i in class_1:
            file_data[count, 41] = 1    
            
        elif i in class_2:
            file_data[count, 41] = 2
            
        elif i in class_3:
            file_data[count, 41] = 3
        else:
            #if not in de data dictionary delete the complete row
            file_data = np.delete(file_data, count, axis=0)
            count = count - 1

        count = count + 1
    
    return file_data

#find the array with only one of each data in the column
def singularity_array(file_data: np.ndarray, column):
    array = []
    for i in file_data[:, column]:
        if i not in array:
            array.insert(len(array), i)

    return array

#function to change from categoric variables to numeric
def transform_categoric_variables(file_data: np.ndarray, column):
    array = singularity_array(file_data, column)
    count = 0
    for i in file_data[:, column]:
        file_data[count, column] = array.index(i)
        count = count +1
    return file_data

#change the matrix to integer
def matrix_to_integer(X):
    for i in range(len(X)):
        for j in range(len(X[0])):
            X[i][j] = float(X[i][j])
    return X

#Normalize one column
def normalize_column(file_data: np.ndarray):
    file_data = file_data.astype(np.float64)
    min = file_data.min()
    max = file_data.max()
    b = 0.99
    a = 0.01
    row=0
    if( max != min ):
        for x in file_data:
            file_data[row]= float(file_data[row])
            x= float(x)
            file_data[row] = ( ( x - min ) / ( max - min )) * ( b - a ) + a 
            row = row + 1
    return file_data

#Normalize all the data
def normalize_data(file_data: np.ndarray):
    for i in range(len(file_data[0])):
        file_data[:,i] = normalize_column(file_data[:,i])
    return file_data

# Normalize entropy
def normalize_entropy(entropy: np.float64, I: np.int64, x: np.int64):
    return (1/(np.log(I)/np.log(x)))*entropy

def calc_entropy_x(file_data: np.ndarray):
    rows, columns = file_data.shape
    I = int(np.ceil(np.sqrt(rows)))
    file_data = file_data.astype(np.float64)
    entropy_list = []
    

    for i in range(columns):
        file_data_column = file_data[:,i]
        min_x = file_data_column.min()
        max_x = file_data_column.max()
        R = max_x - min_x
        values, count = np.unique(file_data_column, return_counts=True)
        filtered_values = dict(zip(values, count))
        entropy = 0
        for j in range(0,I):  #moving in partition
            lower_bound = min_x + R/I *j
            upper_bound = min_x + R/I *(j+1)
            item_list = []
            for item in filtered_values.items():
                if(item[0] >= lower_bound and item[0] <= upper_bound):
                    item_list.append(item)

            d_i = 0
            for item in item_list:
                if item:
                    d_i= d_i + item[1]
            
            probability = d_i / rows
            if(probability != 0):
                entropy = entropy + probability * np.log2(probability)        
        if(entropy != 0):
            entropy_list.append(-entropy)
        else:
            entropy_list.append(entropy)
    return entropy_list

def calc_entropy_y(file_data: np.ndarray):
    N = file_data.size
    count = np.unique(file_data, return_counts=True)
    entropy = 0

    filtered_values = dict(zip(count[0], count[1]))
    
    for i in np.unique(file_data):

        freq = filtered_values[i]
        prob = freq/N

        entropy = entropy + (prob * np.log2(prob))

    return -entropy
    
def calc_joint_entropy(array_x, array_y ):
    entropy = 0.0
    x_value_list = np.unique(array_x)
    y_value_list = np.unique(array_y)
    N = array_x.size
    for i_x in x_value_list:
        for j_y in y_value_list:
            p_xy = len(np.where(np.in1d(np.where( array_x==i_x )[0],np.where( array_y==j_y )[0])==True)[0])/N
            if p_xy > 0.0:
                entropy += p_xy * np.log2(p_xy)
    return -entropy