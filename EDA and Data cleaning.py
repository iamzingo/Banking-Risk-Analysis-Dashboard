# This is the EDA part where we analyse the data.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv('/content/Banking (1).csv') #Copy the path of the file in colab and paste it
df.head(5)

df.shape

df.info()

df.describe()

df['Estimated Income'].min()

bins = [0, 100000, 300000, float('inf')]
labels = ['Low', 'Med', 'High']
df['Income Band'] = pd.cut(df['Estimated Income'],bins=bins, labels=labels, right=False)

df['Income Band'].value_counts().plot(kind='bar')

categorical_cols = df[["BRId","GenderId","IAId","Amount of Credit Cards","Nationality","Occupation","Fee Structure","Loyalty Classification", "Properties Owned","Risk Weighting","Income Band"]].columns

for col in categorical_cols:
  print(f"Value Counts for'{col}':")
  display(df[col].value_counts().plot(kind='bar'))

for i, predictor in enumerate(df[["BRId","GenderId","IAId","Amount of Credit Cards","Nationality","Occupation","Fee Structure","Loyalty Classification", "Properties Owned","Risk Weighting","Income Band"]].columns):
  plt.figure(i)
  sns.countplot(data=df, x=predictor, hue='GenderId')

for i, predictor in enumerate(df[["BRId","GenderId","IAId","Amount of Credit Cards","Nationality","Occupation","Fee Structure","Loyalty Classification", "Properties Owned","Risk Weighting","Income Band"]].columns):
  plt.figure(i)
  sns.countplot(data=df, x=predictor, hue='Nationaality')

#Hisplot

for col in categorical_cols:
  if col == "Occupation":
    continue
  plt.figure(figsize=(0,4))
  sns.histplot(df[col])
  plt.title('Histogram of Occupation Count')
  plt.xlabel(col)
  plt.ylabel("Count")
  plt.show()

#Numerical Analysis

numerical_cols = ['Estimated Income','Superannuation Savings','Credit Card Balance','Bank Loans','Bank Deposits','Checking Accounts','Saving Accounts','Foreign Currency Account','Business Lending']

#Univariate analysis

plt.figure(figsize=(8,4))
for i,col in enumerate(numerical_cols):
  plt.subplot(4,3,i+1)
  sns.histplot(df[col],kde=True)
  plt.title(col)
plt.show()

#Heat maps

numerical_cols = ['Estimated Income','Superannuation Savings','Credit Card Balance','Bank Loans','Bank Deposits','Checking Accounts','Saving Accounts','Foreign Currency Account','Business Lending']

correlation_matrix = df[numerical_cols].corr()

plt.figure(figsize=(12,12))
sns.heatmap(correlation_matrix, annot=True, cmap='crest',fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

