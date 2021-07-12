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

Data cleaning through drop.

```
# Dropping columns
data = data.drop(data.columns[[13, 14, 15, 16, 17, 18, 19]], axis=1)
data = data.dropna() # all rows that contain null values
data = data.dropna(axis=1) # all columns that contain null values
data[col].fillna(data[col].mean()) # fill all NA in col with the mean of col
```
