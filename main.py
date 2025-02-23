import functions
import sklearn.tree as tree
import sklearn.naive_bayes as bayes
import sklearn.neighbors as neighbors
import sklearn.svm as svm
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
import sklearn.preprocessing as preprocessing
import sklearn.neural_network as neural_network
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
import numpy as np

# create arrays to use from the data
x1, y1 = functions.createArrays("src/files/set1.txt") 
x2, y2 = functions.createArrays("src/files/set2.txt")

print(compute_class_weight(class_weight="balanced", classes=np.array([0, 1]), y=y1))
print(compute_class_weight(class_weight="balanced", classes=np.array([0, 1]), y=y2))


set1DecisionTree = tree.DecisionTreeClassifier()
# functions.applyCrossValidation(set1DecisionTree, x1, y1)

set2DecisionTree = tree.DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, min_samples_split=3, max_leaf_nodes=9, splitter="best")
# functions.applyCrossValidation(set2DecisionTree, x2, y2)

set1NaiveBayes = bayes.GaussianNB()
# functions.applyCrossValidation(set1NaiveBayes, x1, y1)

set2NaiveBayes = bayes.GaussianNB()
# functions.applyCrossValidation(set2NaiveBayes, x2, y2)

set1NearestNeighbor = make_pipeline(preprocessing.StandardScaler(), neighbors.KNeighborsClassifier())
# functions.applyCrossValidation(set1NearestNeighbor, x1, y1)

set2NearestNeighbor = make_pipeline(preprocessing.StandardScaler(), neighbors.KNeighborsClassifier()) 
# functions.applyCrossValidation(set2NearestNeighbor, x2, y2)

set1supportVectorMachines = make_pipeline(preprocessing.StandardScaler(), svm.SVC()) 
# functions.applyCrossValidation(set1supportVectorMachines, x1, y1)

set2supportVectorMachines = make_pipeline(preprocessing.StandardScaler(), svm.SVC()) 
# functions.applyCrossValidation(set2supportVectorMachines, x2, y2)

set1MLP = make_pipeline(preprocessing.StandardScaler(), neural_network.MLPClassifier(max_iter=500, early_stopping=True))
functions.applyCrossValidation(set1MLP, x1, y1)

set2MLP = make_pipeline(preprocessing.StandardScaler(), neural_network.MLPClassifier(max_iter=500, early_stopping=True))
functions.applyCrossValidation(set2MLP, x2, y2)

