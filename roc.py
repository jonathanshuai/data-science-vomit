# Skeleton file for basic exploratory analysis

import numpy as np
import pandas as pd
import scipy.optimize
import scipy.stats

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix

import matplotlib as mpl
from matplotlib import pyplot as plt
import hedgeplot as hplt

import statsmodels.api as sm

data_file = "./data/Email-Offer.csv"

df = pd.read_csv(data_file) 

# Visualize
fig, ax = plt.subplots()

x_data = df['Gender']
y_data = df['Age']
color_data = df['TookAction']
ax.scatter(x_data, y_data, c=color_data)

plt.show()

df = pd.get_dummies(df)
feature_columns = ['Age', 'Gender_Male']
X = df[feature_columns]
y = df['TookAction']


X_train, X_test, y_train, y_test = train_test_split(X, y)


from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
parameters = {'penalty': ['l1', 'l2'], 'C': [1e-2, 1e-1, 1, 10, 1e2, 1e3, 1e4]}

clf = GridSearchCV(model, parameters)
clf.fit(X_train, y_train)

train_score = clf.score(X_train, y_train)
test_score = clf.score(X_test, y_test)

print("train score: {}".format(train_score))
print("test score: {}".format(test_score))

print("---best estimator coef---")
print(feature_columns)
print(clf.best_estimator_.coef_)

ground_truth = y_test
probs = clf.predict_proba(X_test)

# Really brute force way to get roc thresholds 
def get_roc_thresholds(ground_truth, probs):
  fp_rate = []
  tp_rate = []
  for i in np.arange(0.0, 1.05, 0.01):
    preds = [1 if p[0] < i else 0 for p in probs]
    conf_matr = confusion_matrix(ground_truth, preds)
    tp, tn, fp, fn = conf_matr[1][1], conf_matr[0][0], conf_matr[0][1], conf_matr[1][0]
    fp_rate.append(fp / (tn + fp)) 
    tp_rate.append(tp / (tp + fn)) 
  return fp_rate, tp_rate

x, y = get_roc_thresholds(ground_truth, probs)

fig, ax = plt.subplots()
ax.plot(x, y)
ax.plot([0, 1], [0, 1], color='r', linestyle='--')
plt.show()




