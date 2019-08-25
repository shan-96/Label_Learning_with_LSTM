import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

from model.kfold import Kfold

skf1 = Kfold(10, None)

skf = StratifiedKFold(n_splits=10, random_state=None)

f1 = []
acc = []
recall = []
prec = []
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clf_rf.fit(X_train, y_train)
    prob_pred = list(map(return_max, clf_rf.predict_proba(X_test)))
    f1.append(f1_score(y_test, prob_pred))
    acc.append(accuracy_score(y_test, prob_pred))
    recall.append(recall_score(y_test, prob_pred))
    prec.append(precision_score(y_test, prob_pred))
    print("F1 Score, Acc Score, Recall Score,Precision Score:", f1_score(y_test, prob_pred),
          accuracy_score(y_test, prob_pred),
          recall_score(y_test, prob_pred), precision_score(y_test, prob_pred))

print("Average F1,Accuracy,recall,prec:", np.mean(f1), np.mean(acc), np.mean(recall), np.mean(prec))
plt.figure(figsize=(20, 10))
plt.plot(f1, color='black', marker='o', linestyle='dashed', label='F1')
plt.plot(acc, color='blue', marker='*', linestyle='dashed', label='Acc')
plt.plot(recall, color='yellow', marker='o', linestyle='dashed', label='recall')
plt.plot(prec, color='red', marker='*', linestyle='dashed', label='precision')
plt.title("Various Accuracy Measures")
plt.legend()
