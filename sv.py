from Normalization import *

config = readFile("cnf_sv.csv")

train_size = int(config[0])
test_size  = int(config[1])
relevance_value = int(config[2])
class_1_bool = bool(config[3])
class_2_bool = bool(config[4])
class_3_bool = bool(config[5])


train = readFile("KDDtrain.txt")
test = readFile("KDDtest.txt")


train = classify(train)
test = classify(test)

train = reduce_data(train,train_size,class_1_bool,class_2_bool,class_3_bool)
test = reduce_data(test,train_size,class_1_bool,class_2_bool,class_3_bool)

train = transform_categoric_variables(train,1)
train = transform_categoric_variables(train,2)
train = transform_categoric_variables(train,3)

test = transform_categoric_variables(test,1)
test = transform_categoric_variables(test,2)
test = transform_categoric_variables(test,3)

test = normalize_data(test)
print(test)