import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('https://raw.githubusercontent.com/amitsrimal1901/Machine_Learning_A-Z/master/Part%203%20-%20Classification/Section%2014%20-%20Logistic%20Regression/Social_Network_Ads.csv')
print (dataset)
X = dataset.iloc[:, [2, 3]].values # Age, Salary
y = dataset.iloc[:, 4].values # Buy yes/no
print(X)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0) # 300 train, 100 test

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)
"""
Random state ensures that the splits that you generate are reproducible. Scikit-learn uses random permutations to generate the splits. 
The random state that you provide is used as a seed to the random number generator. 
This ensures that the random numbers are generated in the same order.
"""
# Predicting the Test set results
y_pred = classifier.predict(X_test)

# accuracy of model
from sklearn.metrics import accuracy_score
score =accuracy_score(y_test,y_pred) # 89% score

# Predicting buy Yes/ No for a given Age, Salary
age=int(input("Enter Age: "))
sal= int(input("Enter Salary: "))
test_data_by_user= np.array([age,sal]).reshape(1,-1)
print(type(test_data_by_user))
Prob_of_Purchase = classifier.predict(test_data_by_user)
print(Prob_of_Purchase)