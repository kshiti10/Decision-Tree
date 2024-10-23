# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 08:20:34 2024

@author: Kshitija
"""

'''
problem statement:
 1.	A cloth manufacturing company is interested to know about 
the different attributes contributing to high sales. 
Build a decision tree & random forest model with Sales 
as target variable (first convert it into categorical variable).   
    
1.	Business Problem
1.1.	What is the business objective?


1.1.	Are there any constraints?


'''
#DT by Vdo
#1.Importing Libraraies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor

#2.Load the Dataset
df=pd.read_csv("C:/datasets/Company_Data.csv")
new_df=df.drop(['Sales'],axis='columns')
target=df['Sales']
new_df.head(5)
new_df.tail(5)
new_df.isnull().sum()
new_df.info()
new_df.describe()

#3.Visualization
#EDA
#scatterplot
plt.scatter(x=new_df['Price'],y=new_df['Income'])
plt.xlabel("Price")
plt.ylabel("Income")
plt.title("Price vs Income")

#heatmap for coralation
sns.heatmap(new_df.corr(),annot=True)

#Boxplot for Outliers
sns.boxplot(data=new_df)

#Model Building
from sklearn.model_selection import train_test_split
#tain the data
X_train,X_test,y_train,y_test = train_test_split(new_df['Population'],new_df['Education'],test_size=0.2)
len(X_train)

#choose the model
regressor=DecisionTreeRegressor()

#train the model
regressor.fit(X_train.values.reshape(-1, 1), y_train.values.reshape(-1, 1))

#testing the model
y_pred=regressor.predict(y_train.values.reshape(-1,1))
y_pred

#create a new dataframe and compare the values
comp=pd.DataFrame({"Actual Value":y_train,"Predicted Value":y_pred})
comp.head(5)
comp.tail(5)
sns.heatmap(comp.corr(),annot=True)

################################################################

#DT by Sir
from sklearn.preprocessing import LabelEncoder
a_CompPrice=LabelEncoder()
a_Income=LabelEncoder()
a_Price=LabelEncoder()
a_Urban=LabelEncoder()
a_Age=LabelEncoder()
a_Advertising=LabelEncoder()

new_df['a_CompPrice']=a_CompPrice.fit_transform(new_df['CompPrice'])
new_df['a_Income']=a_Income.fit_transform(new_df['Income'])
new_df['a_Price']=a_Price.fit_transform(new_df['Price'])
new_df['a_Urban']=a_Urban.fit_transform(new_df['Urban'])
new_df['a_Age']=a_Age.fit_transform(new_df['Age'])
new_df['a_Advertising']=a_Advertising.fit_transform(new_df['Advertising'])
inputs_n=new_df.drop(['Income','Price','Urban','Age','CompPrice'],axis='columns')
target

from sklearn import tree
model=tree.DecisionTreeClassifier()
model.fit(inputs_n,target)
model.predict([[2,1,0]])
model.predict([[2,1,1]])
###########################################################