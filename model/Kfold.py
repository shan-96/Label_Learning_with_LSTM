import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from sklearn.model_selection import StratifiedKFold

from model.Plot import Plot


##TODO: build this model
class Kfold:
    skf = None
    f1 = []
    acc = []
    recall = []
    prec = []

    def __init__(self, n_splits, random_state):
        self.skf = StratifiedKFold(n_splits=n_splits, random_state=random_state)

    def build(self):
        for train_index, test_index in self.skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf_rf.fit(X_train, y_train)
            prob_pred = list(map(return_max, clf_rf.predict_proba(X_test)))
            self.f1.append(f1_score(y_test, prob_pred))
            self.acc.append(accuracy_score(y_test, prob_pred))
            self.recall.append(recall_score(y_test, prob_pred))
            self.prec.append(precision_score(y_test, prob_pred))

        dict = {'data': [self.f1, self.acc, self.prec, self.recall],
                'marker': ['o', '*', 'o', '*'],
                'color': ['black', 'blue', 'yellow', 'red'],
                'linestyle': ['dashed', 'dashed', 'dashed', 'dashed'],
                'label': ['F1-Score', 'Accuracy', 'Precision', 'Recall']}

        data_points = pd.DataFrame(dict)

        print("Average F1: " + np.mean(self.f1))
        print("Average Precision: " + np.mean(self.prec))
        print("Average Recall: " + np.mean(self.recall))
        print("Average Accuracy: " + np.mean(self.acc))

        plot = Plot(data_points)
        plot.draw(20, 10, "--Various Accuracy Measures--")
