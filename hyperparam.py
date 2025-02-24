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
import pandas as pd
from sklearn.model_selection import cross_validate, KFold

def createArrays(src):
    f = open(src, "r")
    xValues = []
    yValues = []
    for line in f:
        arr = line.split("\t")
        x = []
        for i in range(0, len(arr)-1):
            value = arr[i]
            if value == "Absent":
                value = 0
            elif value == "Present":
                value = 1
            else:
                value = float(arr[i])
            x.append(value)
        y = int(arr[len(arr)-1])
        xValues.append(x)
        yValues.append(y)
    f.close()
    return xValues, yValues

"""
# create arrays to use from the data
x1, y1 = functions.createArrays("src/files/set1.txt") 
x2, y2 = functions.createArrays("src/files/set2.txt")

print(compute_class_weight(class_weight="balanced", classes=np.array([0, 1]), y=y1))
print(compute_class_weight(class_weight="balanced", classes=np.array([0, 1]), y=y2))
"""
params_dt1 = {
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],

    'min_samples_leaf': [1, 5, 10],
    'max_leaf_nodes': [None, 5, 10, 15],

    'splitter': ['best', 'random']

}
params_bayes1 = {

    'var_smoothing': [1e-10, 1e-9, 1e-8]

}
params_knn1 = {
    'kneighborsclassifier__n_neighbors': [3, 5, 7, 10],
    'kneighborsclassifier__weights': ['uniform', 'distance']}
params_svm1 = {

    'svc__C': [0.01, 0.1, 1, 10],

    'svc__kernel': ['linear', 'rbf']
}
params_nn1 = {
    'mlpclassifier__hidden_layer_sizes': [(200,100), (100,90), (10, 60)],

    'mlpclassifier__alpha': [0.0001, 0.05],

    'mlpclassifier__learning_rate': ['constant', 'adaptive'],
    'mlpclassifier__solver': ['sgd', 'adam'],
}


def hyperparameter_tune_tree(X:np.ndarray, y:np.ndarray,params_dt:dict)->dict:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=47)
    hyp_tree  = GridSearchCV(tree.DecisionTreeClassifier(), params_dt)
    hyp_tree.fit(X_train, y_train)
    #results = {'best_estimate':hyp_tree.best_estimator_, 'best_parmas': hyp_tree.best_params_, }
    return hyp_tree.best_estimator_, hyp_tree.best_params_

def hyperparameter_tune_naive_bayes(X:np.ndarray, y:np.ndarray,params_nb:dict)->dict:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=47)
    hype_bayes = GridSearchCV(bayes.GaussianNB(), params_nb)
    hype_bayes.fit(X_train, y_train)
    return hype_bayes.best_estimator_,hype_bayes.best_params_

def hyperparameter_tune_knn(X:np.ndarray, y:np.ndarray,params_knn:dict)->dict:
    X_train, X_test,y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=47)
    pipeline_knn = make_pipeline(preprocessing.StandardScaler(), neighbors.KNeighborsClassifier())
    hyp_knn = GridSearchCV(pipeline_knn, params_knn)
    hyp_knn.fit(X_train, y_train)
    return hyp_knn.best_estimator_, hyp_knn.best_params_

def hyperparameter_tune_svm(X:np.ndarray, y:np.ndarray,params_svm:dict)->dict:
    X_train, X_test,y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=47)
    pipeline_svm = make_pipeline(preprocessing.StandardScaler(), svm.SVC()) 
    hyp_svm = GridSearchCV(pipeline_svm, params_svm)
    hyp_svm.fit(X_train, y_train)
    return hyp_svm.best_estimator_,hyp_svm.best_params_

def hyperparameter_tune_nn(X:np.ndarray, y:np.ndarray, params_nn:dict)->dict:
    X_train, X_test,y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=47)
    pipeline_nn = make_pipeline(preprocessing.StandardScaler(), neural_network.MLPClassifier(max_iter=500, early_stopping=True))
    hyp_nn = GridSearchCV(pipeline_nn, params_nn)
    hyp_nn.fit(X_train, y_train)
    return hyp_nn.best_estimator_,hyp_nn.best_params_

def run_everything(X:np.ndarray, y:np.ndarray, params_dt: dict, params_nb: dict, params_knn: dict, params_svm: dict, params_nn: dict):
    pass

def applyCrossValidation(model, X, y):
    kf = KFold(n_splits=10, shuffle=False)
    scores = cross_validate(model, X, y, cv = kf, scoring=['accuracy', 'precision', 'recall', 'f1'])
    results = {

        'mean accuracy': scores['test_accuracy'].mean(),
        'mean precision': scores['test_precision'].mean(),
        'mean recall': scores['test_recall'].mean(),
        'mean f1 score': scores['test_f1'].mean()
    }
    return results
