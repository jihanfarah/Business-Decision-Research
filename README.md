# Business Desicion Research

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

To determine churn customers according to the given definition, we need to look for the most recent transaction that has been made. Based on the information, the customers churn if they do not have any transactions at the store after six months since their last transactions. First, we need to check when was the latest transaction.


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


