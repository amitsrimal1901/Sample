# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 23:58:28 2020

@author: amit_srimal
"""

## Predict Diabetic using Naive Bayes theorem (BASIC approach using Gaussian Distribution)
import pandas as pd

##STEP1: Laoding and Analysing data set
#loading the csv data set file
diabetes_data=pd.read_csv('C:/Users/amit_srimal/Desktop/Study/Python/Files/diabetes.csv')
# Check top few records to get a feel of the data structure
diabetes_data.head()
# Check last few records to get a feel of the data structure
diabetes_data.tail()
# To show the detailed summary 
diabetes_data.info()
#Lets analysze the distribution of the dependent column
diabetes_data.describe().T
#analyse data type of data set
diabetes_data.dtypes

## STEP2: Splitting Data
# separate the dependentand independent columns
X=diabetes_data.drop("Outcome", axis=1)
y= diabetes_data[["Outcome"]]
#Sklearn package's data splitting function which is based on random function
from sklearn.model_selection import train_test_split
# Split X and y into training and test set in 70:30 ratio
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1)
# we will generate a Naive Bayes model on the training set and perform prediction on the test datasets.
# Import Naive Bayes machine learning library
from sklearn.naive_bayes import GaussianNB
model=GaussianNB()
model.fit(X_train,y_train)
pred = model.predict(X_test)

##STEP3: Evaluating Model here; check how often the classifier correctly identified person with diabetes or not.
#Import the metrics
from sklearn import metrics
model_score = model.score(X_test, y_test) 
print('Model score :',model_score) #  0.7835497835497836
print('Confusion Matrix:\n', metrics.confusion_matrix(y_test,pred))
print('Accuracy :',metrics.accuracy_score(y_test,pred)) #0.7835497835497836
print('Precision :',metrics.precision_score(y_test,pred)) #0.7464788732394366
print('Recall :',metrics.recall_score(y_test,pred))# 0.6235294117647059
print('F-score :',metrics.f1_score(y_test,pred)) #0.6794871794871796
print(metrics.classification_report(y_test,pred))

# Note: We may use some optimization but the  Accuracy of the Naive Bayes Model may not very much improved.
# Reason being in Naive Bayes we assume that attributes are independent to each other but in real context they are not completely independent to each other.Some relation exists between them. 

## More detailed approach: 
#https://github.com/LaxmiChaudhary/Building-Naive-Bayes-Classifier-on-Pima-Diabetic-Dataset/blob/master/Building%20Naive%20Bayes%20Model%20on%20Pima%20Diabetes%20Dataset.ipynb
























