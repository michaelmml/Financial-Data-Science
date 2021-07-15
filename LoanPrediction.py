import pandas as pd
import numpy as np
from numpy import mean
import csv
import random
import matplotlib.pyplot as plt
import scipy

from random import seed

from sklearn import svm
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# Imputing missing values and scaling values
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

# Machine Learning Models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsRegressor

import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import cv

# Hyperparameter tuning
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score

# Seaborn for visualization
import seaborn as sns

sns.set(font_scale=2)
data = pd.read_csv('SBAsubset.csv')


"""def missing_values_table(df):
    # Total missing values
    mis_val = df.isnull().sum()

    # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)

    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: 'Missing Values', 1: '% of Total Values'})

    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)

    # Print some summary information
    print("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
                                                              "There are " + str(mis_val_table_ren_columns.shape[0]) +
          " columns that have missing values.")

    # Return the dataframe with missing information
    return mis_val_table_ren_columns


# Get the columns with > 50% missing
missing_df = missing_values_table(data);
missing_columns = list(missing_df[missing_df['% of Total Values'] > 50].index)
"""

data = data.drop(['LoanNr_ChkDgt', 'Name', 'City', 'Zip', 'BankState', 'ApprovalDate', 'DisbursementDate',
                  'DisbursementGross'], axis=1)

# Initial predictor focused on whether default or not - classification problem
data = data.drop(['ChgOffDate', 'BalanceGross', 'ChgOffPrinGr', ], axis=1)

# Create classification by default or not
default_mapping = {"P I F": 0, "CHGOFF": 1}
data['MIS_Status'] = data['MIS_Status'].map(default_mapping)


def clean_currency(x):
    if isinstance(x, str):
        return x.replace('$', '').replace(',', '')
    return x


# Clean up dollar amount to work out cover level

data['GrAppv'] = data['GrAppv'].apply(clean_currency).astype('float')
data['SBA_Appv'] = data['SBA_Appv'].apply(clean_currency).astype('float')
data['ApprovalRatio'] = data['SBA_Appv'] / data['GrAppv']
data = data.drop(['GrAppv', 'SBA_Appv'], axis=1)
data = data.drop(['State', 'Bank'], axis=1)

data = data.dropna(subset=['MIS_Status'])
data['MIS_Status'] = data['MIS_Status'].astype(int)
acceptance = ["Y", "N", "0", "1"]
data = data[data['LowDoc'].isin(acceptance)]
data = data[data['RevLineCr'].isin(acceptance)]

"""binary_mapping = {"Y": 1, "N": 0}
data['RevLineCr'] = data['RevLineCr'].map(binary_mapping)
data['LowDoc'] = data['LowDoc'].map(binary_mapping)
"""

# Prefer One-hot Encoding because there is no logic in the numerical encoding of Revolving Lines or some of the other
# columns
# obj_df["OHC_Code"] = np.where(obj_df["engine_type"].str.contains("ohc"), 1, 0)

data = pd.get_dummies(data, columns=['LowDoc', 'RevLineCr'], prefix=['LowDoc', 'RevLine'])
data['LowDoc_Yes'] = data['LowDoc_Y']
data['RevLine_Yes'] = data['RevLine_1'] + data['RevLine_Y']

data = data.drop(['LowDoc_0', 'LowDoc_Y', 'LowDoc_N', 'RevLine_0', 'RevLine_N', 'RevLine_1', 'RevLine_Y'], axis=1)

data.loc[data['FranchiseCode'] != 1, 'FranchiseCode'] = 0

# Filling in missing data on UrbanRural based on industry type

data['Sector'] = data['NAICS'].astype(str).str[:2]
list = data['Sector'].unique().tolist()

def mode(x):
    values, counts = np.unique(x, return_counts = True)
    m = counts.argmax()
    return values[m]


guess_urbanrural = data.groupby(['Sector'])['UrbanRural'].apply(mode)

for i in list:
    data.loc[(data['UrbanRural'] == 2) & (data['Sector'] == i), 'UrbanRural'] = guess_urbanrural.loc[i]

"""data = pd.get_dummies(data, columns=['UrbanRural'], prefix=['UrbanRural'])"""

# Job growth as % of no of employee

data['JobGrowth'] = data['CreateJob'] / data['NoEmp']
data = data.drop(['CreateJob', 'RetainedJob'], axis=1)

# Absolute number of employees into bins

"""data['EmpBand'] = pd.cut(data['NoEmp'], 5)
data[['EmpBand', 'MIS_Status']].groupby(['EmpBand'], as_index=False).mean().sort_values(by='EmpBand', ascending=True)"""

"""data.loc[data['NoEmp'] <= 16, 'NoEmp'] = 0
data.loc[(data['NoEmp'] > 16) & (data['NoEmp'] <= 32), 'NoEmp'] = 1
data.loc[(data['NoEmp'] > 32) & (data['NoEmp'] <= 48), 'NoEmp'] = 2
data.loc[(data['NoEmp'] > 48) & (data['NoEmp'] <= 64), 'NoEmp'] = 3
data.loc[data['NoEmp'] > 64, 'NoEmp'] = 4"""

# Normalising loan terms...

data = data.drop(['NAICS', 'ApprovalFY', 'Sector'], axis=1)
data.info()

"""def ames_eda(df):
    eda_df = {}
    eda_df['null_sum'] = df.isnull().sum()
    eda_df['null_pct'] = df.isnull().mean()
    eda_df['dtypes'] = df.dtypes
    eda_df['count'] = df.count()
    eda_df['mean'] = df.mean()
    eda_df['median'] = df.median()
    eda_df['min'] = df.min()
    eda_df['max'] = df.max()

    return pd.DataFrame(eda_df)

"""

targets = data['MIS_Status']
features = data.drop(['MIS_Status'], axis=1)

X, X_test, y, y_test = train_test_split(features, targets, test_size=0.3)

print(X.shape)
print(X_test.shape)
print(y.shape)
print(y_test.shape)

# Create the scaler object with a range of 0-1, remove influence of units
scaler = MinMaxScaler(feature_range=(0, 1))

# Fit on the training data
scaler.fit(X)

# Transform both the training and testing data
X = scaler.transform(X)
X_test = scaler.transform(X_test)

# Convert y to one-dimensional array (vector)
y = np.array(y).reshape((-1,))
y_test = np.array(y_test).reshape((-1,))

svc = SVC()
svc.fit(X, y)
# Y_pred = knn.predict(X_test)
acc_svc = round(svc.score(X, y) * 100, 2)
print('SV Classification: %0.4f' % acc_svc)

knn = KNeighborsClassifier(metric='euclidean', n_neighbors=3)
knn.fit(X, y)
# Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X, y) * 100, 2)
print('K-Nearest Neighbors Classification: %0.4f' % acc_knn)
repeatedkfold = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)
cv_results = cross_val_score(knn, X, y, cv=repeatedkfold, scoring='accuracy')
print(mean(cv_results))

gnb = GaussianNB()
gnb.fit(X, y)
# Y_pred = knn.predict(X_test)
acc_gnb = round(gnb.score(X, y) * 100, 2)
print('Naive Bias Classification: %0.4f' % acc_svc)

random_forest = RandomForestClassifier(max_features='log2', n_estimators=1000)
random_forest.fit(X, y)
# Y_pred = random_forest.predict(X_test)
acc_random_forest = round(random_forest.score(X, y) * 100, 2)
print('Random Forest Classifier: %0.4f' % acc_random_forest)
repeatedkfold = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)
cv_results = cross_val_score(random_forest, X, y, cv=repeatedkfold, scoring='accuracy')
print(mean(cv_results))

gradient_boosted = GradientBoostingClassifier(n_estimators=100)
gradient_boosted.fit(X, y)
# Y_pred = gradient_boosted.predict(X_test)
acc_gradient_boosted = round(gradient_boosted.score(X, y) * 100, 2)
print('Gradient Boosted Classification: %0.4f' % acc_gradient_boosted)
repeatedkfold = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)
cv_results = cross_val_score(gradient_boosted, X, y, cv=repeatedkfold, scoring='accuracy')
print(mean(cv_results))

"""# Importing the Keras libraries and packages

from keras.models import Sequential
from keras.layers import Dense

#Initializing Neural Network
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))
# Adding the second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling Neural Network
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting our model
classifier.fit(X, y, batch_size = 10, nb_epoch = 100)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)"""