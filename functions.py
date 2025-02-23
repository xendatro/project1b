from sklearn.model_selection import cross_validate, KFold

# just for colors in console for testing
from colorama import Fore, Back, Style

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

def applyCrossValidation(model, X, y):
    kf = KFold(n_splits=10, shuffle=False)
    scores = cross_validate(model, X, y, cv = kf, scoring=['accuracy', 'precision', 'recall', 'f1'])
    print(Fore.RED + "Mean Accuracy:", scores['test_accuracy'].mean())
    print(Fore.BLUE + "Mean Precision:", scores['test_precision'].mean())
    print(Fore.YELLOW + "Mean Recall:", scores['test_recall'].mean())
    print(Fore.WHITE + "Mean F1:", scores['test_f1'].mean())
