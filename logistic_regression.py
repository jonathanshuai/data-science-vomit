# Skeleton file for basic exploratory analysis

import numpy as np
import pandas as pd
import scipy.optimize
import scipy.stats

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

import matplotlib as mpl
from matplotlib import pyplot as plt
import hedgeplot as hplt

import statsmodels.api as sm

data_file = "./data/Email-Offer.csv"

df = pd.read_csv(data_file) 

# Visualize

df = pd.get_dummies(df)
X = df[['Age', 'Gender_Male']]
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




