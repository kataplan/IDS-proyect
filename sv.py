from Normalization import *

config = readFile("cnf_sv.csv")

train_size = config[0]
test_size  = config[1]
relevance_value = config[2]
class_1_bool = config[3]
class_2_bool = config[4]
class_3_bool = config[5]


train = readFile("KDDtrain.txt")
test = readFile("KDDtest.txt")


train = classify(train)
test = classify(test)

train = transform_categoric_variables(train,1)
train = transform_categoric_variables(train,2)
train = transform_categoric_variables(train,3)

test = transform_categoric_variables(test,1)
test = transform_categoric_variables(test,2)
test = transform_categoric_variables(test,3)

test = normalize_data(test)
print(test)