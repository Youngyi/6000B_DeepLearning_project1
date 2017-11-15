"""
authopr: YangYi
No:20459219
HKUST-MSBD-6000B
Depic: Accuracy 
"""
import  sklearn
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.svm import SVC
import numpy as np
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier   
from sklearn.ensemble import GradientBoostingClassifier
import tensorflow as tf
#data prepocess
features = []
target = []
featuresT = []

with open("traindata.csv") as fea:
    data = fea.readlines()
    for each in data:
        line = each.replace('\n', '').split(',')
        feature = []
        for i in line[0:]:
            feature.append(float(i))
        features.append(feature)
#print(features[3219]) #test

with open("trainlabel.csv") as la:
    data2 = la.readlines()
    for each in data2:
        line = each.replace('\n', '').split(',')
        tar = []
        for i in line[0:]:
            tar.append(float(i))
        target.append(tar)
#print(target[1]) #test

with open("testdata.csv") as tes:
    data3 = tes.readlines()
    for each in data3:
        line = each.replace('\n', '').split(',')
        featureT = []
        for i in line[0:]:
            featureT.append(float(i))
        featuresT.append(featureT)
#print(featuresT[1379]) #test
#print("After data preprocess features")


is_versicolor=(target==1)

binary_target = np.zeros(len(target))
binary_target[is_versicolor] = 1


"""
model = LogisticRegression()
model.fit(features,target)
Accuracy = np.mean(model.predict(features) == target)
print(Accuracy)
Accuracy = np.mean(model.predict(features) == target)
"""

model = LogisticRegression()
model.fit(features,target)
AccuracyL = np.mean(model.predict(featuresT) == target)
print("***********************************")
print("LogisticRegression Accuracy:%f" % (AccuracyL))
model = SVC()
model.fit(features,target)
AccuracyS = np.mean(model.predict(featuresT) == target)
print("***********************************")
print("SupportVectotMachine Accuracy:%f" % (AccuracyS))
model = GradientBoostingClassifier(n_estimators=100)
model.fit(features,target)
AccuracyG = np.mean(model.predict(featuresT) == target)
print("***********************************")
print("GradientBoostingClassifier Accuracy:%f" % (AccuracyG))
model = RandomForestClassifier(n_estimators=10)
model.fit(features,target)
AccuracyR = np.mean(model.predict(featuresT) == target)
print("***********************************")
print("RandomForestClassifier Accuracy:%f" % (AccuracyR))
model = tree.DecisionTreeClassifier()
model.fit(features,target)
AccuracyR = np.mean(model.predict(featuresT) == target)
print("***********************************")
print("DecisionTree Accuracy:%f" % (AccuracyR))

print("******************* Neural Network BP ****************************")
# neiral network layer
def add_layer(input, in_size, out_size, activation_function = None):

    # original weights
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    # biases
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    # w*x+b
    W_mul_x_plus_b = tf.matmul(input, Weights) + biases
    # activating function
    if activation_function is None:
        output = W_mul_x_plus_b
    else:
        output = activation_function(W_mul_x_plus_b)

    return output

# 3 layers net (1,10,1)
# input
x_data = features #np.linspace(-1, 1, 300)[:, np.newaxis]
# out
y_data = target #np.square(x_data) + 1 + noise

# input data
xs = tf.placeholder(tf.float32, [None, 57])
# output data
ys = tf.placeholder(tf.float32, [None, 1])

# two hiden layer
hidden_layer = add_layer(xs, 57, 45, activation_function = tf.nn.relu)
hidden_layer1 = add_layer(hidden_layer, 45, 45, activation_function = tf.sigmoid)
# output layer
prediction = add_layer(hidden_layer1, 45, 1, activation_function = tf.sigmoid)

# nerual network parameters
# loss function
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices = [1]))
# training process
train_step = tf.train.AdamOptimizer(0.1).minimize(loss)
# initialize
init = tf.initialize_all_variables()
# Session
sess = tf.Session()
# initializing
sess.run(init)
# begins
for i in range(2000):
    sess.run(train_step, feed_dict = {xs: x_data, ys: y_data})
    if i % 100 == 0:
        loss_value = sess.run(loss, feed_dict = {xs: x_data, ys: y_data})
        print(loss_value)
temp = sess.run(prediction, feed_dict={xs: x_data})
aver = np.average(temp, axis = 0)
print(aver)
result = [0 if i < aver+0.05 else 1 for i in temp]
result = np.array(result)
cv = 0
for i, j in zip(result, y_data):
    r = float(i) - float(j[0])
    if int(r) == 0:
        cv = cv + 1
print(float(cv)/float(len(result)))

aver = np.average(temp, axis = 0)
result = [0 if i < aver else 1 for i in temp]
result = np.array(result)
with open('project1_204592192.csv', "w+") as file:
    for prediction_result in result:
        print(prediction_result)
        file.write(str(prediction_result))
        file.write('\n')
file.close()


sess.close()
