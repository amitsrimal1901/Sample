# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 10:17:49 2020

@author: amit_srimal
"""
#KNN (K-Nearest Neighbor) is a simple supervised classification algorithm we can use to assign a class to new data point.
# It can be used for regression as well, KNN does not make any assumptions on the data distribution, hence it is non-parametric. 
#It keeps all the training data to make future predictions by computing the similarity between an input sample and each training instance.

##KNN can be summarized as below:
    #1. Computes the distance between the new data point with every training example.
    #2. For computing the distance measures such as Euclidean distance, Hamming distance or Manhattan distance will be used.
    #3. Model picks K entries in the database which are closest to the new data point.
    #4. Then it does the majority vote i.e the most common class/label among those K entries will be the class of the new data point.
    
# Problem Statement:
# Building a model to classify the species of Iris flower based on the sepal length, speal width, petal length and petal width.

## Step1:Import the required data and check the features. 
#Import the load_iris function from datsets module
from sklearn.datasets import load_iris   

#Create bunch object containing iris dataset and its attributes.
iris = load_iris()
type(iris)

#Print the iris data
iris.data

## Each observation represents one flower and 4 columns represents 4 measurements.
## We can see the features(measures) under ‘data’ attribute, where as labels under ‘features_names’.
## labels/responses are encoded as 0,1 and 2.

#Names of 4 features (column names)
print(iris.feature_names) #['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'] NUMERIC & SAME UNIT

#Integers representing the species: 0 = setosa, 1=versicolor, 2=virginica
print(iris.target) # [0 0 0 0 0 0 1 2 2 2 2 2 0 1........................]

# 3 classes of target
print(iris.target_names) # ['setosa' 'versicolor' 'virginica']

print(type(iris.data)) # <class 'numpy.ndarray'>
print(type(iris.target)) # <class 'numpy.ndarray'>

# we have a total of 150 observations and 4 features
print(iris.data.shape) #(150,4)

# Feature matrix in a object named X
X = iris.data
# response vector in a object named y
y = iris.target

print(X.shape) # (150,4)
print(y.shape) # (150,)

##Step 3: Train the Model
# splitting the data into training and test sets (80:20)
##from sklearn.cross_validation import train_test_split # train_test_split is now in model_selection
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=4)
# Optional parameter ‘test-size’ determines the split percentage. 
# ‘random_state’ parameter makes the data split the same way every time you run.

#shape of train and test objects
print(X_train.shape) #(120,4)
print(X_test.shape) #(30,)

# shape of new y objects
print(y_train.shape) # (120,)
print(y_test.shape) #(30,)

## Step 3: Create model by importing the KNeighborsClassifier class from sklearn
from sklearn.neighbors import KNeighborsClassifier
#import metrics model to check the accuracy 
from sklearn import metrics
#Try running from k=1 through 25 and record testing accuracy
k_range = range(1,26)
scores = {}
scores_list = []
for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train,y_train)
        y_pred=knn.predict(X_test)
        scores[k] = metrics.accuracy_score(y_test,y_pred)
        scores_list.append(metrics.accuracy_score(y_test,y_pred))
# ‘fit’ method is used to train the model on training data (X_train,y_train)
# ‘predict’ method to do the testing on testing data (X_test).
# In general the Training accuracy rises as the model complexity increases, for KNN the model complexity is determined by the value of K. 
# Larger K value leads to smoother decision boundary (less complex model). 
# Smaller K leads to more complex model (may lead to overfitting)        
        
#Testing accuracy for each value of K
scores

#Plotting the graph here
#%matplotlib inline
import matplotlib.pyplot as plt
#plot the relationship between K and the testing accuracy
plt.plot(k_range,scores_list)
plt.xlabel('Value of K for KNN')
plt.ylabel('Testing Accuracy')

## FRom the graph; K values with 3 to 19 has the same accuracy which is 96.66, 
#so we can use any one value from that, i choose K as 5 and train the model with full training data.

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X,y)

##making dictionary of type for print purpose only
#0 = setosa, 1=versicolor, 2=virginica
classes = {0:'setosa',1:'versicolor',2:'virginica'}

#Making prediction on SOME UNSEEN DATA 
#predict for the below two random observations
x_new = [[3,4,5,2],
         [5,4,2,2]]
y_predict = knn.predict(x_new)
print(classes[y_predict[0]]) # prediction is versicolor
print(classes[y_predict[1]]) # prediction is setosa






























 