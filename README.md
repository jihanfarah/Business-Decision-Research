# Business Decision Research

## Overview
DQLab sports center is a shop that sells various sports needs such as jackets, clothes, bags, and shoes. This shop started its business in 2013, so it has had regular customers for a long time and is still trying to get new customers until now.

In early 2019, the store manager recruited a Junior Data Analyst to help solve the problem at his store, namely the decline in customers returning to his store. The Junior Data Analyst was also responsible for processing the store's transaction data. The store manager defines a customer as not a customer (churn) when she/he has not transacted to the store again up to the last six months since the last available data update.

The store manager also provided transaction data from 2013 to 2019 in the form of CSV (comma-separated value) with data_retail.csv with 100,000 rows of data.

The data preview:

![image](https://user-images.githubusercontent.com/103634806/180349176-bb92c1b5-a40d-4758-a035-fd50d78c30b6.png)

The field in data included:
1. No
2. Row_Num
3. Customer_ID
4. Product
5. First_Transaction
6. Last_Transaction
7. Average_Transaction_Amount
8. Count_Transaction

## Solution Steps

1. Data preparation test
    - Importing data: Importing data_retail.csv into the python environment.
    - Data cleansing: Performs cleaning and modification of data so that it is ready to be used for further analysis.
2. Data visualization test: Gain insight from the visualization results that have been created.
3. Basic stats method test: Gain insight from the model and evaluate the model that has been created and tested

## Importing Data and Inspection

```
import pandas as pd

df = pd.read_csv('https://storage.googleapis.com/dqlab-dataset/data_retail.csv', sep=';')

print('First five data:')
print(df.head())

print('\nDataset info:')
print(df.info())
```
### Result

![image](https://user-images.githubusercontent.com/103634806/180350257-daa5c990-00ba-48ca-a192-6dd59196d7b3.png)

## Data Cleansing

The two columns that indicate the occurrence of the transaction are not of type datetime, so we need to change them to the datetime data type. We also delete the `no` and `Row_Num` column because those columns do not have any effect for the result.

```
import pandas as pd
df = pd.read_csv('https://storage.googleapis.com/dqlab-dataset/data_retail.csv', sep=';')

# First_Transaction Column
df['First_Transaction'] = pd.to_datetime(df['First_Transaction']/1000, unit='s', origin='1970-01-01')
# Last_Transaction Column
df['Last_Transaction']=pd.to_datetime(df['Last_Transaction']/1000, unit='s', origin='1970-01-01')

# Delete the unused columns
del df['no']
del df['Row_Num']

print('First five data:')
print(df.head())

print('\nDataset Info:')
print(df.info())
```
### Result

![image](https://user-images.githubusercontent.com/103634806/180352393-672479ba-9dcf-47c5-9f65-ed991a2ef79c.png)


## Churn Customers

To determine churn customers according to the given definition, we need to look for the most recent transaction that has been made. Based on the information, the customers churn if they do not have any transactions at the store after six months since the last available data update. First, we need to check when was the latest transaction.


```
# Checking the last transaction in dataset
print(max(df['Last_Transaction']))
```

### Result

![image](https://user-images.githubusercontent.com/103634806/180353796-1e86ee7b-f44f-48ae-bfc3-606df20a87c5.png)

Because the latest transaction is on Feb 1st, 2019, we need to subtract six months to find the limit of the transaction's date. We get that the limit ish on Aug 1st, 2018. Then, we need to classify which customers are churn statuses and which are not using boolean and add it into a new data column.

```
# Classifying the churn customers using boolean
df.loc[df['Last_Transaction'] <= '2018-08-01', 'is_churn'] = True
df.loc[df['Last_Transaction'] > '2018-08-01', 'is_churn'] = False

print('First five data:')
print(df.head())
```

### Result

![image](https://user-images.githubusercontent.com/103634806/180354406-64963c87-83f5-4bef-bf2a-3c7e89fc965a.png)

## Customer Acquisition by Year

```
import matplotlib.pyplot as plt

# Year first transaction column
df['Year_First_Transaction'] = df['First_Transaction'].dt.year
# Year first transaction column
df['Year_Last_Transaction'] = df['Last_Transaction'].dt.year

df_year = df.groupby(['Year_First_Transaction'])['Customer_ID'].count()
df_year.plot(x='Year_First_Transaction', y='Customer_ID', color='springgreen', kind='bar', title='Graph of Customer Acquisition')
plt.xlabel('Year First Transaction')
plt.ylabel('Num of Customer')
plt.tight_layout()
plt.show()
```

### Result

![image](https://user-images.githubusercontent.com/103634806/180628941-9c9bcb0c-b1e7-4a26-aab6-5d367c9408d1.png)

We can see the number of customers significantly increases from 2016 to 2018 around 80.67%, while the number of customers decreases from 2017 to 2018 by around 4.71%.

## Transaction by Year

```
import matplotlib.pyplot as plt

plt.clf()
df_year = df.groupby(['Year_First_Transaction'])['Count_Transaction'].sum()
df_year.plot(x='Year_First_Transaction', y='Count_Transaction', color='springgreen', kind='bar', title='Graph of Transaction Customer')
plt.xlabel('Year First Transaction')
plt.ylabel('Num of Transaction')
plt.tight_layout()
plt.show()

print(df_year)
```

### Result

![image](https://user-images.githubusercontent.com/103634806/180629043-0c06ead5-d5b3-4bf6-8787-083b8669012a.png)

We can see the highest number of transaction happens in 2015 and 2017, and the number of transaction significantly increases from 2014 to 2015. Then, the number of transaction slightly decrease from  2017 to 2018.

## Average Transaction Amount by Year

```
import matplotlib.pyplot as plt
import seaborn as sns

plt.clf()
ax = sns.pointplot(data = df.groupby(['Product', 'Year_First_Transaction']).mean().reset_index(), 
              x='Year_First_Transaction', 
              y='Average_Transaction_Amount', 
              hue='Product')
plt.xlabel('Year First Transaction')
plt.ylabel('Average Transaction Amount')
plt.tight_layout()
plt.show()
```
### Result

![image](https://user-images.githubusercontent.com/103634806/180629074-3fb12216-f9a3-4365-bd64-f1a8df65fe79.png)

## Proportion of Churned Customers for Each Product

```
import matplotlib.pyplot as plt

plt.clf()
# Pivot data with pivot_table
df_piv = df.pivot_table(index='is_churn', 
                        columns='Product',
                        values='Customer_ID', 
                        aggfunc='count', 
                        fill_value=0)
# Get Proportion Churn by Product
plot_product = df_piv.count().sort_values(ascending=False).head(5).index
# Pie chart plot
df_piv = df_piv.reindex(columns=plot_product)
df_piv.plot.pie(subplots=True,
                figsize=(10, 7),
                layout=(-1, 2),
                autopct='%1.0f%%',
                title='Proportion Churn by Product\n')
plt.tight_layout()
plt.show()
```

### Result

![image](https://user-images.githubusercontent.com/103634806/180629251-d804047c-e53f-4188-94fe-c53bf919c3f3.png)


## Count Transaction Categorize Distribution

```
import matplotlib.pyplot as plt

plt.clf()
# Categorize transaction
def func(row):
    if row['Count_Transaction'] == 1:
        val = '1'
    elif (row['Count_Transaction'] > 1 and row['Count_Transaction'] <= 3):
        val ='2 - 3'
    elif (row['Count_Transaction'] >3 and row['Count_Transaction'] <=6):
        val ='4 - 6'
    elif (row['Count_Transaction'] > 6 and row['Count_Transaction'] <= 10):
        val ='7 - 10'
    else:
        val ='> 10'
    return val
# Add new column
df['Count_Transaction_Group'] = df.apply(func, axis=1)

df_year = df.groupby(['Count_Transaction_Group'])['Customer_ID'].count()
df_year.plot(x='Count_Transaction_Group', y='Customer_ID', color='springgreen', kind='bar', title='Customer Distribution by Count Transaction Group')
plt.xlabel('Count Transaction Group')
plt.ylabel('Num of Customer')
plt.tight_layout()
plt.show()
```

### Result

![image](https://user-images.githubusercontent.com/103634806/180629302-0198373c-b33b-477b-9f9b-0ac9118e35e7.png)

## Transaction Amount Average Categorize Distribution

```
import matplotlib.pyplot as plt

plt.clf()
# Transaction Amount Average Distribution
def f(row):
    if (row['Average_Transaction_Amount'] >= 100000 and row['Average_Transaction_Amount'] <=250000):
        val ='1. 100.000 - 250.000'
    elif (row['Average_Transaction_Amount']> 250000 and row['Average_Transaction_Amount'] <= 500000):
        val ='2. >250.000 - 500.000'
    elif (row['Average_Transaction_Amount'] >500000 and row['Average_Transaction_Amount'] <= 750000):
        val = '3. >500.000 - 750.000'
    elif (row['Average_Transaction_Amount']>750000 and row['Average_Transaction_Amount'] <= 1000000):
        val = '4. >750.000 - 1.000.000'
    elif (row['Average_Transaction_Amount']>1000000 and row['Average_Transaction_Amount'] <= 2500000):
        val = '5. >1.000.000 - 2.500.000'
    elif (row['Average_Transaction_Amount']> 2500000 and row['Average_Transaction_Amount'] <= 5000000):
        val = '6. >2.500.000 - 5.000.000'
    elif (row['Average_Transaction_Amount']>5000000 and row['Average_Transaction_Amount'] <=10000000):
        val = '7. >5.000.000 - 10.000.000'
    else:
        val = '8. >10.000.000'
    return val
# Add new column
df['Average_Transaction_Amount_Group'] = df.apply(f, axis=1)

df_year = df.groupby(['Average_Transaction_Amount_Group'])['Customer_ID'].count()
df_year.plot(x='Average_Transaction_Amount_Group', y='Customer_ID',color='springgreen', kind='bar', title='Customer Distribution by Average Transaction Amount Group')
plt.xlabel('Average Transaction Amount Group')
plt.ylabel('Num of Customer')
plt.tight_layout()
plt.show()
```

### Result

![image](https://user-images.githubusercontent.com/103634806/180630080-43d29229-256c-4762-8dda-681835a32a25.png)

## Train, predict, and evaluate
We will use `Average_Transaction_Amount, Count_Transaction,` and `Year_Diff` as feature columns. Then, we use `is_churn` as target column. Before we train the data, we need to change `is_churn` as 'int' format and we use Logistic Regression. Last, we evaluate model with confusion matrix.

```
df['Year_Diff']=df['Year_Last_Transaction']-df['Year_First_Transaction']
feature_columns = ['Average_Transaction_Amount', 'Count_Transaction', 'Year_Diff']

X = df[feature_columns] 
y = df['is_churn'] 
y=y.astype('int')

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# Model logreg initiation
logreg = LogisticRegression()

# fit the model with data
logreg.fit(X_train, y_train)

# Predict model
y_pred = logreg.predict(X_test)

# Evaluate model with confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:\n', cnf_matrix)

# import required modules
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.clf()
# name  of classes
class_names = [0, 1] 
fig, ax = plt.subplots()

tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap='YlGnBu', fmt='g')
ax.xaxis.set_label_position('top')
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.show()
```

### Result
![image](https://user-images.githubusercontent.com/103634806/180636970-4b4f54e3-75c3-4bce-8578-fca8a971ce54.png)

![image](https://user-images.githubusercontent.com/103634806/180636972-68644a70-b807-4677-93ac-96876cbfa997.png)

## Accuracy, Precision, and Recall


```
from sklearn.metrics import accuracy_score, precision_score, recall_score

#Find Accuracy, Precision, dan Recall
print('Accuracy :', accuracy_score(y_test,y_pred))
print('Precision:', precision_score(y_test, y_pred, average='micro'))
print('Recall   :', recall_score(y_test, y_pred, average='micro'))
```

### Result 
![image](https://user-images.githubusercontent.com/103634806/180636991-f8377a22-d7e3-4aad-aed9-ae852d7b1421.png)

