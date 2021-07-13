# Financial Data Science
Leveraging statistical and data science techniques in analysing financial information and application of machine learning.

## Table of Contents
* [Exploratory Data Analysis](#exploratory-data-analysis)

## Exploratory Data Analysis
**Cleaning Data**

Methods to clean strings such as dollar amounts or unusual expressions.

```
    data[col] = data[col].str.replace(',', '', regex=True)
    data[col] = data[col].str.replace(r'-(?!\d)', '', regex=True)  # important understand regular expressions
    data[col] = data[col].str.replace('n.a.', '', regex=True)
    data[col] = data[col].replace(r'^\s+$', np.nan, regex=True)
    data[col] = data[col].astype(float)
    
    # Extract method to take the string before an expression
    data['Title'] = data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
```    

Data cleaning through drop and various substitution and filling methods. Mainly dealing with null data and whether to drop or populate them based on statistics.

```
# Dropping columns
data = data.drop(data.columns[[13, 14, 15, 16, 17, 18, 19]], axis=1)
data = data.dropna() # all rows that contain null values
data = data.dropna(axis=1) # all columns that contain null values
data[col].fillna(data[col].mean()) # fill all NA in col with the mean of col

# Filling numerical data by statistics based on other columns
# Groupby returns the statistics of the values in 'Age' grouped by the other two columns
guess_ages = data.groupby(['S', 'P'])['Age'].apply(np.mean)

# Use of selector on already-cleaned S and P column data to 
  for i in range(0, 2):
      for j in range(0, 3):
          dataset.loc[(dataset['Age'].isnull()) & (dataset['S'] == i) & (dataset['P'] == j + 1), 'Age'] =
          guess_ages[i, j + 1]

# The loc selector works by Boolean.
```

Changing row data through filter, sort and various mappings. For numerical data, filter is a common tool to omit extremes and creating data-bands to normalise data (e.g. on age data).

```
# Mapping to numerical categories
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
data['Title'] = data['Title'].map(title_mapping)

# Sort
data = data[data[col] > 1]

# Mapping numerical data to data-bands
data['AgeBand'] = pd.cut(data['Age'], 5)
data.loc[data['Age'] <= 16, 'Age'] = 0
data.loc[(data['Age'] > 16) & (data['Age'] <= 32), 'Age'] = 1
data.loc[(data['Age'] > 32) & (data['Age'] <= 48), 'Age'] = 2
data.loc[(data['Age'] > 48) & (data['Age'] <= 64), 'Age'] = 3
data.loc[data['Age'] > 64, 'Age'] = 4

# More complex usage of apply method
data[new_col] = data.groupby([grouping])[col].apply(lambda x: x.fillna(0).cumsum())

# Alongside filtering
data[new_col] = data[data[grouping_2] == "None"].groupby([grouping])[col].apply(lambda x: x.fillna(0).cumsum())

# Index based on column data mathematics (e.g. find date index of the half-life)
dataindex = allprodrate[[grouping, half_life_amount]].apply(lambda x: abs(data.loc[(data[grouping] ==
                        x[grouping]), total_amount] - x[half_life_amount]).idxmin(), axis=1)
