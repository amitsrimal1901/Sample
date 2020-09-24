# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 12:00:23 2020

@author: amit_srimal
"""
##Predict if the mushroom is eatable or poisonoud form the dataset------

from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
# Import test_train_split function
from sklearn.model_selection import train_test_split
# Import metrics
from sklearn import metrics

## import the data file here
dataset=pd.read_csv("C:/Users/amit_srimal/Desktop/Study/Python/Files/mushrooms.csv")
dataset= dataset.sample(frac=1)
dataset.columns # from class,cp-shape ....habitat

# now assigning column name to data set header
for label in dataset.columns:
    dataset[label]=LabelEncoder().fit(dataset[label]).transform(dataset[label])
print(dataset.info())
##class: e/p: eatable(e) vs poisonous(p)

# Now split the dependent and independent variables
X= dataset.drop(['class'],axis=1) # (8124,22)
Y= dataset['class'] # (8124,)

# Now creating the training and testing dataset
X_train, X_test, Y_train,Y_test=train_test_split(X,Y, test_size=0.3)
model= DecisionTreeClassifier(criterion='entropy',max_depth=1)
AdaBoost=AdaBoostClassifier(base_estimator=model, n_estimators=400, learning_rate=1)
boostmodel= AdaBoost.fit(X_train,Y_train)

# now tring to predict
y_pred=boostmodel.predict(X_test)
predictions= metrics.accuracy_score(Y_test, y_pred)
print('The sccuracy of model is: ',predictions*100,'%')