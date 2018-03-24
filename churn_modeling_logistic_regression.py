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

# Dummy variable if balance is 0
train_df['Balance_dummy'] = train_df['Balance'].apply(lambda x: 0 if x != 0 else 1)
test_df['Balance_dummy'] = test_df['Balance'].apply(lambda x: 0 if x != 0 else 1)

# ln(balance)
train_df['Balance_ln'] = train_df['Balance'].apply(lambda x: np.log(x) if x != 0 else 0)
test_df['Balance_ln'] = test_df['Balance'].apply(lambda x: np.log(x) if x != 0 else 0)

# balance / age (think of wealth accumulation speed)
train_df['WealthAccumulation'] = train_df[['Balance', 'Age']].apply(lambda x: x[0] / x[1], axis=1)
test_df['WealthAccumulation'] = test_df[['Balance', 'Age']].apply(lambda x: x[0] / x[1], axis=1)

# ln(WealthAccumulation)
train_df['Log_WA'] = train_df['WealthAccumulation'].apply(lambda x: np.log(x) if x != 0 else 0)
test_df['Log_WA'] = test_df['WealthAccumulation'].apply(lambda x: np.log(x) if x != 0 else 0)


# Get the label and predictor columns
label_column = 'Exited'
predictor_columns = [
  'CreditScore', 
  'Age',
  'Tenure',
  'NumOfProducts',
  'HasCrCard',
  'IsActiveMember',
  'EstimatedSalary',
  'Gender_Female',
  'Geography_Germany',
  'Geography_Spain',
  'Balance_dummy',
  'Balance_ln',
  # 'Balance',
  'WealthAccumulation'
  # 'Log_WA'
  ]

# Get the relevant data
X_train, y_train = train_df[predictor_columns], train_df[label_column]
X_test, y_test = test_df[predictor_columns], test_df[label_column]

# Get variance inflation factors (measure of multi?collinearity)
def get_vif(X):
  vif_df = pd.DataFrame()
  vif_df['feature'] = X.columns
  vif_df['vif_value'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
  return vif_df

vif_df = get_vif(X_train)
print("=== variance inflation factors ===")
print(vif_df)

# Look at correlation 
correlation_columns = ['Balance_dummy', 'Balance_ln', 'WealthAccumulation', 'Age']
temp_df = X_train[correlation_columns]
correlation_df = pd.DataFrame(np.corrcoef(temp_df.values, rowvar=False))
correlation_df.index = correlation_columns
correlation_df.columns = correlation_columns


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

# Plot a CAP (cumulative accuracy profile) curve
# Note: the CAP can be interpreted as what percent of samples (on the x axis) 
# should we look at to get some accuracy (on the y axis), similar to the stats
# reported in Predictive Analytics. 
# e.g. Customers in the top 50% of probability scores accounted for 80% of the exits
# or something along those lines.
# This kind of analysis relies on the predictions to be sorted by probability

n_samples = len(y_train)
total_positives = np.sum(y_train)
train_probs = clf.predict_proba(X_train)
sorted_train_probs = list(zip(train_probs[:,0], y_train)) # Note: train_probs[:,0] is the probability of NEGATIVE
sorted_train_probs.sort() # This sorts by ASC so samples predicted more likely to be positive are first

# Get the points to plot accuracy curve
seen_positives = 0

step = 1 / (n_samples)
accuracy_curve_x = np.arange(step, 1 + step, step)
accuracy_curve_y = []
for i, (_, label) in enumerate(sorted_train_probs):
  seen_positives += label
  accuracy_curve_y.append(seen_positives / total_positives)


fig, ax = plt.subplots()
ax.plot([0, 1], [0, 1], color='r', linestyle='--')
ax.plot(accuracy_curve_x, accuracy_curve_y, color='r')

plt.show()
