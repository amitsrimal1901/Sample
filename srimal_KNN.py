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


#*************************************************************************************************************
#GITHUB: https://github.com/amitsrimal1901/Machine_Learning_A-Z/blob/master/Part%203%20-%20Classification/Section%2015%20-%20K-Nearest%20Neighbors%20(K-NN)/knn.py
"""
K-NN is a non-parametric and lazy learning algorithm. 
Non-parametric means there is no assumption for underlying data distribution i.e. the model structure determined from the dataset.
Lazy algorithm because it does not need any training data points for model generation. All training data is used in the testing phase which makes training faster and testing phase slower and costlier.
K-Nearest Neighbor (K-NN) is a simple algorithm that stores all the available cases and classifies the new data or case based on a similarity measure.

In K-NN classification, the output is a class membership. 
An object is classified by a plurality vote of its neighbors, with the object being assigned to the class most common among its k nearest neighbors (k is a positive integer, typically small). 
If k = 1, then the object is simply assigned to the class of that single nearest neighbor.

To determine which of the K instances in the training dataset are most similar to a new input, a distance measure is used.
For real-valued input variables, the most popular distance measure is the EUCLIDEAN DISTANCE.

##EUCLIDEAN DISTANCE:
The Euclidean distance is the most common distance metric used in low dimensional data sets. It is also known as the L2 norm. 
The Euclidean distance is the usual manner in which distance is measured in the real world.
where p and q are n-dimensional vectors and denoted by p = (p1, p2,…, pn) and q = (q1, q2,…, qn) represent the n attribute values of two records.
While Euclidean distance is useful in low dimensions, it doesn’t work well in high dimensions and for categorical variables. 
The drawback of Euclidean distance is that it ignores the similarity between attributes. Each attribute is treated as totally different from all of the attributes.

## Other popular distance measures :
    1. Hamming Distance: Calculate the distance between binary vectors.
    2. Manhattan Distance: Calculate the distance between real vectors using the sum of their absolute difference. Also called City Block Distance.
    3. Minkowski Distance: Generalization of Euclidean and Manhattan distance.
    
## STEPS OF ALGO
    1. Divide the data into training and test data.
    2. Select a value K.
    3. Determine which distance function is to be used.
    4. Choose a sample from the test data that needs to be classified and compute the distance to its n training samples.
    5. Sort the distances obtained and take the k-nearest data samples.
    6. Assign the test class to the class based on the majority vote of its k neighbors.    

## Performance of the K-NN algorithm is influenced by three main factors :
    1. The distance function or distance metric used to determine the nearest neighbors.
    2. The decision rule used to derive a classification from the K-nearest neighbors.
    3. The number of neighbors used to classify the new example.

## Advantages of K-NN :
    1. The K-NN algorithm is very easy to implement.
    2. Nearly optimal in the large sample limit.
    3. Uses local information, which can yield highly adaptive behavior.
    4. Lends itself very easily to parallel implementation.
    
## Disadvantages of K-NN :
    1. Large storage requirements.
    2. Computationally intensive recall.
    3. Highly susceptible to the curse of dimensionality.    

## 
APPLICATION AREA OF KNN:
    1. Finance — financial institutes will predict the credit rating of customers.
    2. Healthcare — gene expression.
    3. Political Science — classifying potential voters in two classes will vote or won’t vote.
    4. Handwriting detection.
    5. Image Recognition.
    6. Video Recognition.
    7. Pattern Recognition

"""
#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('https://raw.githubusercontent.com/amitsrimal1901/Machine_Learning_A-Z/master/Part%203%20-%20Classification/Section%2015%20-%20K-Nearest%20Neighbors%20(K-NN)/Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('K-NN (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('K-NN (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

#***************************************************************************************************************************
# https://towardsdatascience.com/knn-algorithm-what-when-why-how-41405c16c36f
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

data = pd.read_csv('https://raw.githubusercontent.com/adityakumar529/Coursera_Capstone/master/diabetes.csv')

# There are some factors where the values cannot be zero. Glucose values, for example, cannot be 0 for a human.
# Similarly, BloodPressure, SkinThickness, Insulin, and BMI cannot be zero for a human.

non_zero = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
for column in non_zero:
    data[column] = data[column].replace(0,np.NaN)
    mean = int(data[column].mean(skipna = True))
    data[column] = data[column].replace(np.NaN, mean)
    print(data[column])

# sub plotting
#We have defined non_zero with the column where the values cannot be zero.
#In each column we will first check if we have 0 values. Then we replace it with NaN.
#Later we are creating a meaning of the column and replacing the earlier with mean.

import seaborn as sns
p=sns.pairplot(data, hue = 'Outcome')

# geting fecthing and sample name

























 