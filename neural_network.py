
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import pandas as pd

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

X, y = createArrays("project1_dataset1.txt")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=47)
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

mlp = MLPClassifier(max_iter=100, random_state=47)

param_grid = {
    'hidden_layer_sizes': [(200,100), (100,90), (10, 60)],

    'alpha': [0.0001, 0.05],

    'learning_rate': ['constant', 'adaptive'],
    'solver': ['sgd', 'adam'],
}

exhaust_app = GridSearchCV(mlp, param_grid, n_jobs=-1, scoring='accuracy')
exhaust_app.fit(X_train, y_train)

optimal_nn = exhaust_app.best_estimator_
y_pred = optimal_nn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
