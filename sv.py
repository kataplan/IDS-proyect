import numpy as np

def load_data(file_name:str):
    file_data = np.genfromtxt(file_name, dtype=float, delimiter=",")
    x = file_data[:,range(len(file_data[0])-2)]
    y = file_data[:, len(file_data[0])-1]
    return ( x,y )

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

config = np.genfromtxt("cnf_sv.csv", dtype=int, delimiter=",")

train_size = config[0]
test_size  = config[1]
k = config[2]
singularity_vector = config[3]
class_1_bool = bool(config[4])
class_2_bool = bool(config[5])
class_3_bool = bool(config[6])
train_x, train_y = load_data("dtrn.csv")

list_entropy_x = calc_entropy_x(train_x)
entropy_y = calc_entropy_y(train_y)
array = []
index_array = []
filter_array = []
for i in range(len(train_x[0])):
    entropy_x = list_entropy_x[i]
    entropy_xy = calc_joint_entropy(train_x[:,i], train_y)
    G = entropy_x+ entropy_y - entropy_xy
    correlation = 2* G/(entropy_y + entropy_xy)
    array.append([correlation,i])
array.sort(key=lambda a: a[0] ,reverse=True)


for i in range(k):
    index_array.append(int(array[i][1]))
    filter_array.append(array[i][0])

np.savetxt("index.csv",index_array,delimiter=",", fmt='%d')
np.savetxt("filter.csv",filter_array, delimiter=",")

train_x = train_x[:,index_array]
train_x = train_x.astype(np.float64)
train_y = train_y.astype(np.int64)
P, D, matrix_v = np.linalg.svd(train_x, full_matrices=False)
train_x = np.matmul(matrix_v[:,:singularity_vector].T , train_x.T)
print(train_x)
np.savetxt("filter_v.csv",matrix_v,delimiter=",")
