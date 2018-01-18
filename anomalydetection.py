# coding=utf-8

# write code...
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager


from sklearn.datasets import make_classification
from sklearn.cross_validation import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import roc_auc_score

n_samples = 200
outliers_fraction = 0.01
n_inliers = int((1. - outliers_fraction) * n_samples)
n_outliers = int(outliers_fraction*n_samples)



X_work = np.round(np.random.normal(500, 200, (n_inliers, 117)))
y_work  = np.ones(n_inliers)
X_non_work =  np.round(np.random.normal(5, 1, (n_outliers, 117)))
y_non_work = np.ones(n_outliers) * -1

X = np.concatenate((X_work, X_non_work), axis=0)
y = np.concatenate((y_work, y_non_work), axis=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

print(X_train)


clf = svm.OneClassSVM(nu=outliers_fraction, kernel="poly")
clf.fit(X)


y_pred_train = clf.predict(X_train)
y_pred_non_work = clf.predict(X_non_work)

print(y_pred_train)
print(y_train)
print(y_pred_train - y_train)

print("non_work")
print(X_non_work)
print(y_pred_non_work)
