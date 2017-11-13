"""
authopr: YangYi
No:20459219
HKUST-MSBD-6000B
Depic: Accuracy 
"""
import  sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.svm import SVC
import numpy as np
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier   
from sklearn.ensemble import GradientBoostingClassifier
#data prepocess
features = []
target = []
featuresT = []

with open("traindata.csv") as fea:
    data = fea.readlines()
    for each in data:
        line = each[:-1].split(',')
        feature = []
        for i in line[0:]:
            feature.append(float(i))
        features.append(feature)
#print(features[3219]) #test

with open("trainlabel.csv") as la:
    data2 = la.readlines()
    for each in data2:
        line = each[:-1].split(',')
        tar = []
        for i in line[0:]:
            tar.append(float(i))
        target.append(tar)
#print(target[1]) #test

with open("testdata.csv") as tes:
    data3 = tes.readlines()
    for each in data3:
        line = each[:-1].split(',')
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

"""
model = RandomForestClassifier(n_estimators=10)
model.fit(features,target)
AccuracyR = np.mean(model.predict(featuresT) == target)
print("***********************************")
print("RandomForestClassifier Accuracy:%f" % (AccuracyR))
"""
