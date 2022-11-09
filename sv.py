from functions_sv import *

config = readFile("cnf_sv.csv")

train_size = int(config[0])
test_size  = int(config[1])
relevance_value = int(config[2])
class_1_bool = bool(int(config[4]))
class_2_bool = bool(int(config[5]))
class_3_bool = bool(int(config[6]))

train = readFile("KDDtrain.txt")
#test = readFile("KDDtest.txt")


train = classify(train)
#test = classify(test)

train = reduce_data(train,train_size,class_1_bool,class_2_bool,class_3_bool)
##test = reduce_data(test,test_size,class_1_bool,class_2_bool,class_3_bool)

train = transform_categoric_variables(train,1)
train = transform_categoric_variables(train,2)
train = transform_categoric_variables(train,3)

#test = transform_categoric_variables(test,1)
#test = transform_categoric_variables(test,2)
#test = transform_categoric_variables(test,3)

#train= normalize_data(train)
#test = normalize_data(test)

calc_entropy_x(train)
calc_entropy_y(train[:, 0])