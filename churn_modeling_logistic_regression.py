import numpy as np
import pandas as pd
import scipy.optimize
import scipy.stats

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

import matplotlib as mpl
from matplotlib import pyplot as plt
import hedgeplot as hplt

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

train_file = "./data/Churn-Modelling.csv"
test_file = "./data/Churn-Modelling-Test-Data.csv"

train_df = pd.read_csv(train_file) 
test_df = pd.read_csv(test_file) 

# Drop unnecessary columns
train_df = train_df.drop('Surname', axis=1)
test_df = test_df.drop('Surname', axis=1)

# Create dummy variables
train_df = pd.get_dummies(train_df)
test_df = pd.get_dummies(test_df)

# Perform transformations
train_df['Balance_dummy'] = train_df['Balance'].apply(lambda x: 0 if x != 0 else 1)
train_df['Balance_ln'] = train_df['Balance'].apply(lambda x: np.log(x) if x != 0 else 0)
train_df['WealthAccumulation'] = train_df[['Balance', 'Age']].apply(lambda x: x[0]/x[1], axis=1)

test_df['Balance_dummy'] = test_df['Balance'].apply(lambda x: 0 if x != 0 else 1)
test_df['Balance_ln'] = test_df['Balance'].apply(lambda x: np.log(x) if x != 0 else 0)
test_df['WealthAccumulation'] = test_df[['Balance', 'Age']].apply(lambda x: x[0]/x[1], axis=1)


# Get the label and predictor columns
label_column = 'Exited'
predictor_columns = [
  'CreditScore', 'Age', 'Tenure', 'NumOfProducts', 
  'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
  'Gender_Female', 'Geography_Germany', 'Geography_Spain',
  'Balance_dummy', 'Balance_ln', 'WealthAccumulation']

# Get the relevant data
X_train, y_train = train_df[predictor_columns], train_df[label_column]
X_test, y_test = test_df[predictor_columns], test_df[label_column]

# Simple to print model results
def get_model_results(clf, X, y):
  preds = clf.predict(X)
  acc = accuracy_score(y, preds)
  recall = recall_score(y, preds)
  precision = precision_score(y, preds)
  conf_matr = confusion_matrix(y, preds)

  print("Accuracy: {}".format(acc))
  print("Recall: {}".format(recall))
  print("Precision: {}".format(precision))
  print("True Negative: {}".format(conf_matr[0][0]))
  print("False Positive: {}".format(conf_matr[0][1]))
  print("False Negative: {}".format(conf_matr[1][0]))
  print("True Positive: {}".format(conf_matr[1][1]))

  return preds, acc, recall, precision, conf_matr


# Modeling the data

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
parameters = {
  'penalty': ['l1', 'l2'], 
  'C': [1e-2, 0.1, 1, 10, 1e2, 1e3]}

clf = GridSearchCV(model, parameters)
clf.fit(X_train, y_train)

print("===== Results on training set =====")
(train_preds, train_acc, train_recall,
train_precision, train_conf_matr) = get_model_results(clf, X_train, y_train)
print(predictor_columns)
print(clf.best_estimator_.coef_)
print("====================================")


# print("===== Results on testing set =====")
# (test_preds, test_acc, test_recall,
# test_precision, test_conf_matr) = get_model_results(clf, X_test, y_test)
# print("====================================")

