# Financial Data Science
Leveraging statistical and data science techniques in analysing financial information and application of machine learning. Importance of an efficient data cleaning approach due to the quantity of data. Financial news / data extraction from Twitter and news; exploratory data analysis with sentiment analysis using DistilBERT.

![FinSentiment](https://user-images.githubusercontent.com/84533632/154868200-ebd14d16-d407-413f-8e50-35350d8f2eee.PNG)

## Specific Example (Outdated)
In this case, ~130,000 SBA loan applications by SMEs. Data often also has undefined entries / blanks and various approaches are required to resolve this - whether in dropping the data or filling it by a set of conditions. Another problem is the number of categories, e.g. by state, sector, bank with a large range of unique entries that make numerical / one-hot encoding impractical.

## Table of Contents
* [Exploratory Data Analysis](#exploratory-data-analysis)
* [SBA Loan EDA](#sba-loan-eda)

## Exploratory Data Analysis
**Managing Financial Data**

Methods to clean strings such as dollar amounts or unusual expressions.

```
def clean_currency(x):
    if isinstance(x, str):
        return x.replace('$', '').replace(',', '')
    return x

# Clean up dollar amount to work out cover level
data[col] = data[col].apply(clean_currency).astype('float')
```

Missing information - populate based on statistics of another column. In this case, the mode for every unique entry of col is determined.

```
def mode(x):
    values, counts = np.unique(x, return_counts = True)
    m = counts.argmax()
    return values[m]

list = data[col].unique().tolist()
guess_col_target = data.groupby([col])[col_target].apply(mode)

for i in list:
    data.loc[(data[col_target] == 2) & (data[col] == i), 'col_target'] = guess_col_target.loc[i]
```

Categories - removing null data through filter and isin method. One-hot encode for categorical features, as opposed to numerical encoding.

```
acceptance = ["Y", "N", "0", "1"]
data = data[data[col].isin(acceptance)]
data = pd.get_dummies(data, columns=[col], prefix=['col'])
```


## SBA Loan EDA

Resulting clean-up gives two numerical subsets - term and number of employees. Rescaled with minmax. The remaining are categorical data (either binary or else one-hot encode is applied) and percentages. To deal with certain other categories with a large number of unique entries, Groupby / Pivot and correlation analysis can be performed to determine the "importance" of those columns.

```
# X   -Column         -Non-Null -Count   -Dtype
 0   -Term           -136062 -non-null  -int64  
 1   -NoEmp          -136062 -non-null  -int64  
 2   -NewExist       -136062 -non-null  -int64  
 3   -FranchiseCode  -136062 -non-null  -int64  
 4   -UrbanRural     -136062 -non-null  -int64   
 5   -ApprovalRatio  -136062 -non-null  -float64   
 6   -LowDoc_Yes     -136062 -non-null  -uint8  
 7   -RevLine_Yes    -136062 -non-null  -uint8  
 8   -JobGrowth      -136062 -non-null  -float64
```

Final touches with train_test_split to create to separate sets of data for training and testing, following by scaler function mainly for the numerical data, i.e. loan term and number of employees.

```
X, X_test, y, y_test = train_test_split(features, targets, test_size=0.3)

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
```

An example of gradient boosting classifier applied on the data and scoring based on the score method as well as cross validation.

```
gradient_boosted = GradientBoostingClassifier(n_estimators=100)
gradient_boosted.fit(X, y)
# Y_pred = gradient_boosted.predict(X_test)
acc_gradient_boosted = round(gradient_boosted.score(X, y) * 100, 2)
print('Gradient Boosted Classification: %0.4f' % acc_gradient_boosted)
repeatedkfold = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)
cv_results = cross_val_score(gradient_boosted, X, y, cv=repeatedkfold, scoring='accuracy')
print(mean(cv_results))
```
