# Code notes/exercises from a lecture on feature selection
# in linear regression models.

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
from statsmodels.stats.outliers_influence import summary_table
from statsmodels.sandbox.regression.predstd import wls_prediction_std

data_file = "./data/50-Startups.csv"

df = pd.read_csv(data_file)


df =  pd.get_dummies(df)
relevant_columns = ['R&D Spend', 'Administration', 'Marketing Spend', 'State_California']

X = df[relevant_columns].values
y = df['Profit'].values

X_train, X_test, y_train, y_test = train_test_split(X, y)

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.feature_selection import f_regression

parameters = {'alpha': [1e-1, 1, 10, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7]}
lasso_model = GridSearchCV(Lasso(), parameters)
lasso_model.fit(X_train, y_train)

train_pred = lasso_model.predict(X_train)

test_pred = lasso_model.predict(X_test)

print("lasso training mse: {}".format(mean_squared_error(train_pred, y_train)))
print("lasso testing mse: {}".format(mean_squared_error(test_pred, y_test)))

# See which features lasso regression selected
print(lasso_model.coef_)



# Note: Should I add an intercept term to X_train before this??

# Simple feature selection algorithms for linear regression

def backwards_elimination(X, y, columns=[], exit_sl=0.05):
  if columns == []:
    return []
  _, p_values = f_regression(X[:, columns], y)
  i = p_values.argmax()
  if p_values[i] > exit_sl:
    del columns[i]
    return backwards_elimination(X, y, columns, exit_sl)
  else:
    return columns 


def forward_selection(X, y, columns=[], entry_sl=0.05):
  n_columns = X.shape[1]
  best_p_value, best_index = 1, -1
  for i in range(n_columns):
    if i not in columns:
      new_columns = columns + [i]
      _, p_values = f_regression(X[:,new_columns], y)
      if p_values[-1] < best_p_value:
        best_p_value, best_index = p_values[-1], i
  if best_p_value <= entry_sl:
    new_columns = columns + [best_index]
    return forward_selection(X, y, new_columns, entry_sl)
  else:
    return columns

def bidirectional_elimination(X, y, columns=[], entry_sl=0.05, exit_sl=0.05):
  converged = False
  while not converged:
    new_columns = forward_selection(X, y, columns)
    new_columns = backwards_elimination(X, y, new_columns)
    if columns == new_columns:
      converged = True
    columns = new_columns

  return new_columns

columns = bidirectional_elimination(X_train, y_train)



clf = LinearRegression()
clf.fit(X_train, y_train)
clf.predict(X_train)
clf.predict(X_test)

X_train = X_train[columns]
X_test = X_test[columns]



