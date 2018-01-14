# coding=utf-8

# write code...

import numpy as np

from sklearn.datasets import make_classification
from sklearn.cross_validation import train_test_split

#from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
#from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager


from sklearn.metrics import roc_auc_score

# サンプルデータの生成
# 1000 samples、5(infomative) + 2(redundant) + 13(independent) =  20 feature のデータを生成
#dat = make_classification(n_samples=1000, n_features=106, n_informative=80, n_redundant=20, n_classes=2, weights= [0.99,0.01])
#X = np.round((dat[0]+30)*2)
#y = dat[1]

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

#clf = RandomForestClassifier(n_estimators=500, random_state=123)
#clf.fit(X_train, y_train)
#print("RandomForestClassifier AUC =", roc_auc_score(y_test, clf.predict_proba(X_test)[:,1]))

n_samples = 200
outliers_fraction = 0.01
n_inliers = int((1. - outliers_fraction) * n_samples)
n_outliers = int(outliers_fraction*n_samples)



X_work = np.round(np.random.normal(500, 200, (n_inliers, 106)))
y_work  = np.ones(n_inliers)
X_non_work =  np.round(np.random.normal(5, 1, (n_outliers, 106)))
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
