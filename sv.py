from functions_sv import *

config = readFile("cnf_sv.csv")

train_size = int(config[0])
test_size  = int(config[1])
k = int(config[2])
singularity_vector = int(config[3])
class_1_bool = bool(int(config[4]))
class_2_bool = bool(int(config[5]))
class_3_bool = bool(int(config[6]))

train = readFile("KDDtrain.txt")
train = classify(train)
train = transform_categoric_variables(train,1)
train = transform_categoric_variables(train,2)
train = transform_categoric_variables(train,3)
train_x, train_y= reduce_data(train,train_size,class_1_bool,class_2_bool,class_3_bool)

list_entropy_x = calc_entropy_x(train_x)
entropy_y = calc_entropy_y(train_y)
array = []
train_x = normalize_data(train_x)
for i in range(len(train_x[0])):
    entropy_x = list_entropy_x[i]
    entropy_xy = calc_joint_entropy(train_x[:,i], train_y)
    G = entropy_x+ entropy_y - entropy_xy
    correlation = 2* G/(entropy_y + entropy_xy)
    array.append([correlation,i])

array.sort(key=lambda a: a[0] ,reverse=True)
a = []
b = []

for i in range(k):
    a.append(int(array[i][1]))
    b.append(array[i][0])
b = normalize_column(np.array(b))
np.savetxt("index.csv",a,delimiter=",", fmt='%d')
np.savetxt("filter.csv",b, delimiter=",")
train_x = train_x[:,a]
train_x = train_x.astype(np.float64)
train_y = train_y.astype(np.int64)
P, D, matrix_v = np.linalg.svd(train_x, full_matrices=False)

train_x = np.matmul(matrix_v[:,:singularity_vector].T , train_x.T)

np.savetxt("filter_v.csv",matrix_v,delimiter=",")
np.savetxt("dtrn.csv",train_x,delimiter=",")
np.savetxt("etrn.csv",train_y,delimiter=",",fmt='%d')

test = readFile("KDDtest.txt")
test = classify(test)
test = transform_categoric_variables(test,1)
test = transform_categoric_variables(test,2)
test = transform_categoric_variables(test,3)
test_x, test_y = reduce_data(test,test_size,class_1_bool,class_2_bool,class_3_bool)

test_x = normalize_data(test_x)
test_x = test_x[:,a]
test_x = test_x.astype(np.float64)
test_y = test_y.astype(np.int64)
test_x = np.matmul(matrix_v[:,:singularity_vector].T , test_x.T)

np.savetxt("dtst.csv",test_x,delimiter=",")
np.savetxt("etst.csv",test_y,delimiter=",",fmt='%d')
