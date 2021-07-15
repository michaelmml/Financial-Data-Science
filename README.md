# Financial Data Science
Leveraging statistical and data science techniques in analysing financial information and application of machine learning.

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

Resulting clean-up gives two numerical subsets - term and number of employees. Rescaled with minmax. The remaining are categorical data (either binary or else one-hot encode is applied) and percentages.

#### X   -Column         -Non-Null -Count   -Dtype
 0   -Term           -136062 -non-null  -int64  
 1   -NoEmp          -136062 -non-null  -int64  
 2   -NewExist       -136062 -non-null  -int64  
 3   -FranchiseCode  -136062 -non-null  -int64  
 4   -UrbanRural     -136062 -non-null  -int64   
 5   -ApprovalRatio  -136062 -non-null  -float64
 6   -LowDoc_Yes     -136062 -non-null  -uint8  
 7   -RevLine_Yes    -136062 -non-null  -uint8  
 8   -JobGrowth      -136062 -non-null  -float64
