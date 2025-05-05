import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('E:/ALL ACADEMIC/MIDS/Titanic-Dataset.csv')
df.shape
df.dtypes
df.columns
#Sum of null values
df.isnull().sum()
#Visualize missing values
plt.figure(figsize=(12,6))
sns.heatmap(df.isnull())
plt.title('Missing values Heatmap')
plt.show()
#Clean the data
#We drop columns with excessive missing values
df.drop(columns=['Cabin','PassengerId','Name','Ticket'],axis=1,inplace=True)
df.sample(5)
df.describe()
df['Family'] = df['SibSp'] + df['Parch'] + 1
df.head()
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.isnull().sum()
df['Sex'] = df['Sex'].map({'male':0, 'female':1})
df
#Normalize or Scale Numeric Features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[['Age','Fare']] = scaler.fit_transform(df[['Age','Fare']])
#Splitting the data for ML
from sklearn.model_selection import train_test_split
X = df.drop('Survived', axis=1)
y = df['Survived']
#Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)
