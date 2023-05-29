

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('Customer-Churn-Records (1).csv')

print(data.head())

print("Number of rows", data.shape[0])
print("Number of columns :", data.shape[1])
print(data.columns)
print(data.dtypes)
print(data.isnull().sum())
print(data.describe())

plt.figure(figsize=(6,6))
churn_counts = data['Exited'].value_counts()
plt.pie(churn_counts, labels=['Retained','Churned'],autopct='%1.1f%%',startangle=90)
plt.axis('equal')
plt.title('Churn Distribution')
plt.show()


plt.figure(figsize=(8, 6))
sns.countplot(x='Gender', hue='Exited', data=data)
plt.title('Churn by Gender')
plt.show()

plt.figure(figsize=(8, 6))
sns.countplot(x='Geography', hue='Exited', data=data)
plt.title('Churn by Geography')
plt.show()


data['AgeGroup'] = pd.cut(data['Age'], bins=[0, 30, 40, 50, 60, np.inf], labels=['<30', '30-40'
, '40-50', '50-60', '60+'])


plt.figure(figsize=(8,6))
sns.countplot(x='AgeGroup', hue='Exited', data=data)
plt.title('Churn by Age Gp')
plt.show()

churn_rate_gender = data.groupby('Gender')['Exited'].mean()
print(churn_rate_gender)

churn_rate_geography = data.groupby('Geography')['Exited'].mean()
print(churn_rate_geography)

churn_rate_age = data.groupby('AgeGroup')['Exited'].mean()
print(churn_rate_age)


plt.figure(figsize=(8, 6))
sns.countplot(x='NumOfProducts', hue='Exited', data=data)
plt.title('Churn by No. of products')
plt.show()


plt.figure(figsize=(8, 6))
sns.countplot(x='HasCrCard', hue='Exited', data=data)
plt.title('Churn by credit card')
plt.show()


plt.figure(figsize=(8, 6))
sns.countplot(x='IsActiveMember', hue='Exited', data=data)
plt.title('Churn by activity status')
plt.show()


plt.figure(figsize=(10, 6))
sns.boxplot(x='Exited', y='Balance', data=data)
plt.title('Churn by Balance')
plt.show()



































