from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import sklearn.neighbors as neighbors
import sklearn.svm as svm
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
import sklearn.preprocessing as preprocessing
import sklearn.neural_network as neural_network
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
import numpy as np
import functions
import sklearn.tree as tree
import sklearn.naive_bayes as bayes

"""
# create arrays to use from the data
x1, y1 = functions.createArrays("src/files/set1.txt") 
x2, y2 = functions.createArrays("src/files/set2.txt")

print(compute_class_weight(class_weight="balanced", classes=np.array([0, 1]), y=y1))
print(compute_class_weight(class_weight="balanced", classes=np.array([0, 1]), y=y2))
"""

def hyperparameter_tune_tree(X:np.ndarray, y:np.ndarray,params_dt:dict):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=47)
    hyp_tree  = GridSearchCV(tree.DecisionTreeClassifier(), params_dt)
    hyp_tree.fit(X_train, y_train)
    return hyp_tree.best_params_

def hyperparameter_tune_naive_bayes(X:np.ndarray, y:np.ndarray,params_nb:dict):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=47)
    hype_bayes = GridSearchCV(bayes.GaussianNB(), params_nb)
    hype_bayes.fit(X_train, y_train)
    return hype_bayes.best_params_

def hyperparameter_tune_knn(X:np.ndarray, y:np.ndarray,params_knn:dict):
    X_train, X_test,y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=47)
    pipeline_knn = make_pipeline(preprocessing.StandardScaler(), neighbors.KNeighborsClassifier())
    hyp_knn = GridSearchCV(pipeline_knn, params_knn)
    hyp_knn.fit(X_train, y_train)
    return hyp_knn.best_params_

def hyperparameter_tune_svm(X:np.ndarray, y:np.ndarray,params_svm:dict):
    X_train, X_test,y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=47)
    pipeline_svm = make_pipeline(preprocessing.StandardScaler(), svm.SVC()) 
    hyp_svm = GridSearchCV(pipeline_svm, params_svm)
    hyp_svm.fit(X_train, y_train)
    return hyp_svm.best_params_

def hyperparameter_tune_nn(X:np.ndarray, y:np.ndarray, params_nn:dict):
    X_train, X_test,y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=47)
    pipeline_nn = make_pipeline(preprocessing.StandardScaler(), neural_network.MLPClassifier(max_iter=500, early_stopping=True))
    hyp_nn = GridSearchCV(pipeline_nn, params_nn)
    hyp_nn.fit(X_train, y_train)
    return hyp_nn.best_params_