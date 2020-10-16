########  ** MODEL ** ########
#________________________________________________________________________________
## Decision tree on Analytical Vidhya
# Source: https://www.analyticsvidhya.com/blog/2016/04/complete-tutorial-tree-based-modeling-scratch-in-python/

# one of the best and mostly used supervised learning methods.
# empower predictive models with high accuracy, stability and ease of interpretation.
# they map non-linear relationships quite well.
# Decision tree is a type of supervised learning algorithm (having a pre-defined target variable).
# works for both categorical and continuous input and output variables.
# decision tree identifies the most significant variable and it’s value that gives best homogeneous sets of population.

## Types of Decision Trees: based on type of target variable we have
# type 1: categorical tree: which has categorical target variable. ege. Student will play cricket or not” i.e. YES or NO.
# type 2: continuos tree: which has continuous target variable

## Important Terminology related to Decision Trees:
#1. Root Node: It represents entire population or sample and this further gets divided into two or more homogeneous sets.
#2. Splitting: It is a process of dividing a node into two or more sub-nodes.
#3. Decision Node: When a sub-node splits into further sub-nodes, then it is called decision node.
#4. Leaf/ Terminal Node: Nodes do not split is called Leaf or Terminal node.
#5. Pruning: When we remove sub-nodes of a decision node, this process is called pruning. You can say opposite process of splitting.
#6. Branch / Sub-Tree: A sub section of entire tree is called branch or sub-tree.
#7. Parent and Child Node: A node, which is divided into sub-nodes is called parent node of sub-nodes where as sub-nodes are the child of parent node.

## Advantages:
#1. Easy to Understand: does not require any statistical knowledge to read and interpret them.
#2. Useful in Data exploration: we can create new variables / features that has better power to predict target variable.
#3. Less data cleaning required: not influenced by outliers and missing values to a fair degree
#4. Data type is not a constraint: It can handle both numerical and categorical variables.
#5. Non Parametric Method: have no assumptions about the space distribution and the classifier structure

## Disadvantages:
#1. Over fitting: most practical difficulty for decision tree models. This problem gets solved by setting constraints on model parameters and pruning.
#2. Not fit for continuous variables: While working with continuous numerical variables, decision tree looses information when it categorizes variables in different categories.


###### Overfitting & underfitting concept:
#1. Overfitting: 
# Overfitting is a modeling error which occurs when a function is too closely fit to a limited set of data points.
# happens when a model learns the detail and noise in the training data to the extent that it negatively impacts the performance of the model on new data. 
#2. underfitting:
# Underfitting refers to a model that can neither model the training data nor generalize to new data
# The remedy is to move on and try alternate machine learning algorithms.
    #Overfitting: Good performance on the training data, poor generliazation to other data.
    #Underfitting: Poor performance on the training data and poor generalization to other data

#IMP. Ideally, you want to select a model at the sweet spot between underfitting and overfitting.

## How does a tree decide where to split.
# decision of making strategic splits heavily affects a tree’s accuracy.
# The decision criteria is different for classification and regression trees.
#  use multiple algorithms to decide to split a node in two or more sub-nodes.
# creation of sub-nodes increases the homogeneity of resultant sub-nodes.
# Decision tree splits the nodes on all available variables and then selects the split which results in most homogeneous sub-nodes.

## The algorithm selection is also based on type of target variables.
#1. Gini Index
#2. Chi-Square
#3. Information gain
#4. Reduction in variance

## key parameters of tree modeling and how can we avoid over-fitting in decision trees
# If there is no limit set of a decision tree, it will give you 100% accuracy on training set.
# reason being it will end up making 1 leaf for each observation.
## HENCE it can be tackled by two ways as below:
    #1. Setting constraints on tree size
    #2. Tree pruning

## Setting constarint depends on:
    #1. Minimum samples for a node split
    #2. Minimum samples for a terminal node (leaf)
    #3. Maximum depth of tree (vertical depth)
    #4. maximum number of terminal nodes
    #5. maximum features to consider for split

## how to implement PRUNING in decision tree
    #1.We first make the decision tree to a large depth.
    #2.Then we start at the bottom and start removing leaves which are giving us negative returns when compared from the top.
    #3.Suppose a split is giving us a gain of say -10 (loss of 10) and then the next split on that gives us a gain of 20. A simple decision tree will stop at step 1 but in pruning, we will see that the overall gain is +10 and keep both leaves

## Are tree based models better than linear models?
# Actually, you can use any(Logictic or Linear model) algorithm. It is dependent on the type of problem you are solving
    #1.If the relationship between dependent & independent variable is well approximated by a linear model, linear regression will outperform tree based model.
    #2.If there is a high non-linearity & complex relationship between dependent & independent variables, a tree model will outperform a classical regression method.
    #3.If you need to build a model which is easy to explain to people, a decision tree model will always do better than a linear model. Decision tree models are even simpler to interpret than linear regression!

## What are ensemble methods in tree based modeling ?
# literary meaning of word ‘ensemble’ is group.
# Ensemble methods involve group of predictive models to achieve a better accuracy and model stability
# Ensemble methods are known to impart SUPREME BOOST to tree based models

# Like every other model, a tree based model also suffers from the plague of bias and variance.
#IMP: You build a small tree and you will get a model with low variance and high bias.
    # Bias means, ‘how much on an average are the predicted values different from the actual value.
    # Variance means, ‘how different will the predictions of the model be at the same point if different samples are taken from the same population’.

# Normally, as you increase the complexity of your model, you will see a reduction in prediction error due to lower bias in the model.
# As you continue to make your model more complex, you end up over-fitting your model and your model will start suffering from high variance.    
# champion model should maintain a balance between these two types of errors. This is known as the trade-off management of bias-variance errors.
# Ensemble learning is one way to execute this trade off analysis    
## Note: commonly used ensemble methods include: Bagging, Boosting and Stacking

## BAGGING
## What is Bagging? How does it work?    
# Bagging is a technique used to reduce the variance of our predictions by combining the result of multiple classifiers modeled on different sub-samples of the same data set.

## The steps followed in bagging are:
    #1.Create Multiple DataSets
    #2.Build Multiple Classifiers
    #3.Combine Classifiers    
# Higher number of models are always better or may give similar performance than lower numbers.
# It can be theoretically shown that the variance of the combined predictions are reduced to 1/n (n: number of classifiers) of the original variance, under some assumptions.    

## RANDOM FOREST
# There are various implementations of bagging models. Random forest is one of them.
# IMP: when you can’t think of any algorithm (irrespective of situation), use random forest!
# Its a versatile machine learning method capable of performing both regression and classification tasks.
# It also undertakes dimensional reduction methods, treats missing values, outlier values and other essential steps of data exploration.
# It is a type of ensemble learning method, where a group of weak models combine to form a powerful model.

## How does it works:
# In Random Forest, we grow multiple trees as opposed to a single tree in CART(Classification and Regression Tree) model    
#  To classify a new object based on attributes, each tree gives a classification and we say the tree “votes” for that class.
# The forest chooses the classification having the most votes (over all the trees in the forest) and in case of regression, it takes the average of outputs by different trees.
    
# More details on Random forest: https://www.analyticsvidhya.com/blog/2014/06/introduction-random-forest-simplified/

## Advantages:
# can solve both type of problems i.e. classification and regression and does a decent estimation at both fronts.
# the power of handle large data set with higher dimensionality. It can handle thousands of input variables and identify most significant variables so it is considered as one of the dimensionality reduction methods.
# effective method for estimating missing data and maintains accuracy when a large proportion of the data are missing.
# methods for balancing errors in data sets where classes are imbalanced.
# extended to unlabeled data, leading to unsupervised clustering, data views and outlier detection.
# using the out-of-bag error estimate removes the need for a set aside test set.
       # Info: Random Forest involves sampling of the input data with replacement called as bootstrap SAMPLING.
       # Here one third of the data is not used for training and can be used to testing. These are called the out of bag samples.     

## Disadvatages:
# does a good job at classification but not as good as for regression problem as it does not give precise continuous nature predictions.
# feel like a black box approach for statistical modelers – you have very little control on what the model does.
       
## BOOSTING
# ‘Boosting’ refers to a family of algorithms which converts weak learner to strong learners.
# eg say to idebtify a SPAM, we may few rules say 1,2,3
# Individually, these rules are not powerful enough to classify an email into ‘spam’ or ‘not spam’. Therefore, these rules are called as weak learner.

# To convert weak learner to strong learner, we’ll combine the prediction of each weak learner using methods like:
    #1. Using average/ weighted average
    #2. Considering prediction has higher vote
       
## How does it work?
# Step1: ‘How boosting identify weak rules?‘
    # To find weak rule, we apply base learning (ML) algorithms with a different distribution
    # Each time base learning algorithm is applied, it generates a new weak prediction rule.Its an iterative process.
    #  After many iterations, the boosting algorithm combines these weak rules into a single strong prediction rule
# Step2: ‘How do we choose different distribution for each round?’
    # The base learner takes all the distributions and assign equal weight or attention to each observation.
    # If there is any prediction error caused by first base learning algorithm, then we pay higher attention to observations having prediction error. Then, we apply the next base learning algorithm.
    # Iterate above step till the limit of base learning algorithm is reached or higher accuracy is achieved.

###  Two most commonly used algorithms i.e. Gradient Boosting (GBM) and XGboost (extreme gradient boosting)
# Which is more powerful: GBM or Xgboost?
# Xgboost performs better than GBM with benefits/ advantages as below:
    #1.Regularization: hence xgboost is also known as ‘regularized boosting‘ technique.
    #2.Parallel Processing
    #3.High Flexibility
    #4.Handling Missing Values
    #5.Tree Pruning
    #6.Built-in Cross-Validation
    #7.Continue on Existing Model

#-------------------------------------------------------------------------------
### XGBOOST model: https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/

#-------------------------------------------------------------------------------
## Edureka: https://www.youtube.com/watch?v=qDcl-FRnwSU
#Classification: process of dividing datasets into different categories by adding labels.    

# Type of Classification
#1. Decision tree
#2. Random forest    
#3. Naive Bayes
#4. KNN    

## Decisio ntree on fruit data---------------------------------------------------
# Source: https://towardsdatascience.com/solving-a-simple-classification-problem-with-python-fruits-lovers-edition-d20ab6b071d2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

fruits = pd.read_table('D:/PythonReferences/fruit.txt')
fruits.head()
print(fruits.shape) #  59 by 7. 59 fruits having 7 attributes each
# unique fruit type in data set
fruits['fruit_name'].unique() # array ['apple', 'mandarin', 'orange', 'lemon']
# checking size(quantity) of each fruit type
fruits.groupby('fruit_name').size() # 19,l16,m5 & o19 split of qnty.
# Note: fruit basket is well balanced except mandarin
import seaborn as sns
sns.countplot(fruits['fruit_name'],label="Count")
plt.show()

## Visualization
# Box plot for each numeric variable will give us a clearer idea of the distribution of the input variables
fruits.drop('fruit_label', axis=1).plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False, figsize=(9,9),title='Box Plot for each input variable')
plt.savefig('fruits_box')
plt.show()

# It looks like perhaps color score has a near Gaussian distribution.
import pylab as pl
fruits.drop('fruit_label' ,axis=1).hist(bins=30, figsize=(9,9))
pl.suptitle("Histogram for each numeric input variable")
plt.savefig('fruits_hist')
plt.show()
# Some pairs of attributes are correlated (mass and width). 
# This suggests a high correlation and a predictable relationship.
from pandas.tools.plotting import scatter_matrix
from matplotlib import cm

feature_names = ['mass', 'width', 'height', 'color_score']
X = fruits[feature_names]
y = fruits['fruit_label']

cmap = cm.get_cmap('gnuplot')
scatter = pd.scatter_matrix(X, c = y, marker = 'o', s=40, hist_kwds={'bins':15}, figsize=(9,9), cmap = cmap)
plt.suptitle('Scatter-matrix for each input variable')
plt.savefig('fruits_scatter_matrix')
plt.show()
# statistical summary
fruits.describe()
# We can see that the numerical values do not have the same scale. 
# We will need to apply scaling to the test set that we computed for the training set.

## Create Training and Test Sets and Apply Scaling
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0) # didnt specififed the split size

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

## Build Models based on training & testing data set & Testig the prediction accuracy with:
# Model 1: Logistic Regression
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(X_train, y_train)

print('Accuracy of Logistic regression classifier on training set: {:.2f}'.format(logreg.score(X_train, y_train)))
print('Accuracy of Logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
#Accuracy of Logistic regression classifier on training set: 0.70
#Accuracy of Logistic regression classifier on test set: 0.40

# Model 2: KNN 
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
print('Accuracy of K-NN classifier on training set: {:.2f}'.format(knn.score(X_train, y_train)))
print('Accuracy of K-NN classifier on test set: {:.2f}'.format(knn.score(X_test, y_test)))
#Accuracy of K-NN classifier on training set: 0.95
#Accuracy of K-NN classifier on test set: 1.00

# Model 3:Linear Discriminant Analysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
print('Accuracy of LDA classifier on training set: {:.2f}'.format(lda.score(X_train, y_train)))
print('Accuracy of LDA classifier on test set: {:.2f}'.format(lda.score(X_test, y_test)))
#Accuracy of LDA classifier on training set: 0.86
#Accuracy of LDA classifier on test set: 0.67

# Model 4: Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(X_train, y_train)
print('Accuracy of GNB classifier on training set: {:.2f}'.format(gnb.score(X_train, y_train)))
print('Accuracy of GNB classifier on test set: {:.2f}'.format(gnb.score(X_test, y_test)))
#Accuracy of GNB classifier on training set: 0.86
#Accuracy of GNB classifier on test set: 0.67

# model 5: Support vector machine
from sklearn.svm import SVC

svm = SVC()
svm.fit(X_train, y_train)
print('Accuracy of SVM classifier on training set: {:.2f}'.format(svm.score(X_train, y_train)))
print('Accuracy of SVM classifier on test set: {:.2f}'.format(svm.score(X_test, y_test)))
#Accuracy of SVM classifier on training set: 0.61
#Accuracy of SVM classifier on test set: 0.33

## Conclusion:
#The KNN algorithm was the most accurate model that we tried. 
#The confusion matrix provides an indication of no error made on the test set. However, the test set was very small.
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
pred = knn.predict(X_test)
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))

# Plot the Decision Boundary of the k-NN Classifier
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches

X = fruits[['mass', 'width', 'height', 'color_score']]
y = fruits['fruit_label']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

def plot_fruit_knn(X, y, n_neighbors, weights):
    X_mat = X[['height', 'width']].as_matrix()
    y_mat = y.as_matrix()

# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF','#AFAFAF'])
cmap_bold  = ListedColormap(['#FF0000', '#00FF00', '#0000FF','#AFAFAF'])

clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
clf.fit(X_mat, y_mat)

# Plot the decision boundary by assigning a color in the color map to each mesh point.
    
    mesh_step_size = .01  # step size in the mesh
    plot_symbol_size = 50
    
    x_min, x_max = X_mat[:, 0].min() - 1, X_mat[:, 0].max() + 1
    y_min, y_max = X_mat[:, 1].min() - 1, X_mat[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, mesh_step_size),
                         np.arange(y_min, y_max, mesh_step_size))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot training points
    plt.scatter(X_mat[:, 0], X_mat[:, 1], s=plot_symbol_size, c=y, cmap=cmap_bold, edgecolor = 'black')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    patch0 = mpatches.Patch(color='#FF0000', label='apple')
    patch1 = mpatches.Patch(color='#00FF00', label='mandarin')
    patch2 = mpatches.Patch(color='#0000FF', label='orange')
    patch3 = mpatches.Patch(color='#AFAFAF', label='lemon')
    plt.legend(handles=[patch0, patch1, patch2, patch3])

        
    plt.xlabel('height (cm)')
    plt.ylabel('width (cm)')
    plt.title("4-Class classification (k = %i, weights = '%s')"
              % (n_neighbors, weights))
    
    plt.show()

plot_fruit_knn(X_train, y_train, 5, 'uniform')
plot_fruit_knn(X_train, y_train, 1, 'uniform')
plot_fruit_knn(X_train, y_train, 10, 'uniform')
plot_fruit_knn(X_test, y_test, 5, 'uniform')

k_range = range(1, 20)
scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    scores.append(knn.score(X_test, y_test))
plt.figure()
plt.xlabel('k')
plt.ylabel('accuracy')
plt.scatter(k_range, scores)
plt.xticks([0,5,10,15,20])
# For this particular dateset, we obtain the highest accuracy when k=5.


## Weather prediction decision tree ## -----------------------------------------
# https://www.geeksforgeeks.org/decision-tree-implementation-python/

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

#importing Dataset   
# Function importing Dataset
def importdata():
    balance_data =  pd.read_csv('D:/PythonReferences/balance-scale.txt',sep= ',', header = None)   
     
    # Printing the dataswet shape
    print ("Dataset Lenght: ", len(balance_data))
    print ("Dataset Shape: ", balance_data.shape)
     
    # Printing the dataset obseravtions
    print ("Dataset: ",balance_data.head())
    return balance_data
 
# Function to split the dataset
def splitdataset(balance_data):
 
    # Seperating the target variable
    X = balance_data.values[:, 1:5]
    Y = balance_data.values[:, 0]
 
    # Spliting the dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split( 
    X, Y, test_size = 0.3, random_state = 100)
     
    return X, Y, X_train, X_test, y_train, y_test
# 437+188 split from 625. into train & test data set

# Terms used in code
# Gini index and information gain both of these methods are used to select from the n attributes of the dataset which attribute would be placed at the root node or the internal node
#1. Gini index
# Gini Index is a metric to measure how often a randomly chosen element would be incorrectly identified.
#It means an attribute with lower gini index should be preferred.
# Sklearn supports “gini” criteria for Gini Index and by default, it takes “gini” value.

#2. Entropy
# Entropy is the measure of uncertainty of a random variable, it characterizes the impurity of an arbitrary collection of examples.
# The higher the entropy the more the information content.

#3. Information gain 
#The entropy typically changes when we use a node in a decision tree to partition the training instances into smaller subsets.
#  Information gain is a measure of this change in entropy.

#4. Accuracy score
#Accuracy score is used to calculate the accuracy of the trained classifier.

# Function to perform training with giniIndex.
def train_using_gini(X_train, X_test, y_train):
 
    # Creating the classifier object
    clf_gini = DecisionTreeClassifier(criterion = "gini",
            random_state = 100,max_depth=3, min_samples_leaf=5)
 
    # Performing training
    clf_gini.fit(X_train, y_train)
    return clf_gini

# Function to perform training with entropy.
def tarin_using_entropy(X_train, X_test, y_train):
 
    # Decision tree with entropy
    clf_entropy = DecisionTreeClassifier(
            criterion = "entropy", random_state = 100,
            max_depth = 3, min_samples_leaf = 5)
 
    # Performing training
    clf_entropy.fit(X_train, y_train)
    return clf_entropy
 
 
# Function to make predictions
def prediction(X_test, clf_object):
 
    # Predicton on test with giniIndex
    y_pred = clf_object.predict(X_test)
    print("Predicted values:")
    print(y_pred)
    return y_pred
     
# Function to calculate accuracy
def cal_accuracy(y_test, y_pred):
     
    print("Confusion Matrix: ",
        confusion_matrix(y_test, y_pred))
     
    print ("Accuracy : ",
    accuracy_score(y_test,y_pred)*100)
     
    print("Report : ",
    classification_report(y_test, y_pred))
 
# Driver code
def main():
     
    # Building Phase
    data = importdata()
    X, Y, X_train, X_test, y_train, y_test = splitdataset(data)
    clf_gini = train_using_gini(X_train, X_test, y_train)
    clf_entropy = tarin_using_entropy(X_train, X_test, y_train)
     
    # Operational Phase
    print("Results Using Gini Index:")
     
    # Prediction using gini
    y_pred_gini = prediction(X_test, clf_gini)
    cal_accuracy(y_test, y_pred_gini)
     
    print("Results Using Entropy:")
    # Prediction using entropy
    y_pred_entropy = prediction(X_test, clf_entropy)
    cal_accuracy(y_test, y_pred_entropy)
     
# Calling main function
if __name__=="__main__":
    main()

#*********************************************************************************************************************
# Decison Tree Regression
#https://github.com/amitsrimal1901/Machine_Learning_A-Z/blob/master/Part%202%20-%20Regression/Section%208%20-%20Decision%20Tree%20Regression/regression_template.py

"""
The decision trees is used to fit a sine curve with addition noisy observation. 
As a result, it learns local linear regressions approximating the sine curve.
We can see that if the maximum depth of the tree (controlled by the max_depth parameter) is set too high, 
the decision trees learn too fine details of the training data and learn from the noise, i.e. they overfit.
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('C:/Users/amit_srimal/Documents/Study/Python/Files/Position_Salaries.csv')
#dataset = pd.read_csv('https://raw.githubusercontent.com/amitsrimal1901/Machine_Learning_A-Z/master/Part%201%20-%20Data%20Preprocessing/Data.csv')
X = dataset.iloc[:, 1:2].values  # numpy array
y = dataset.iloc[:, 2].values #  # numpy array

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor() #<class 'sklearn.tree._classes.DecisionTreeRegressor'>
regressor.fit(X, y) # DecisionTreeRegressor()

#plotting
from sklearn import tree
fig = plt.figure(figsize=(20,10))
_ = tree.plot_tree(regressor, feature_names=np.asarray(dataset.iloc[:, 1:2].columns), filled=True)
"""
In above plot, we need to have feature_namea
So we got columns name first using .iloc
Then we converted the index from abpve step to array
"""
# Predicting a new result
y_pred = regressor.predict([[6.5]]) # array([150000.])

# Getting accuracy
#from sklearn import metrics
#print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# Visualising the Decision Tree Regression results (higher resolution)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#**************************************************************************************************************************
"""
decision trees can be divided, with respect to the target values, into:
1. Classification trees used to classify samples, assign to a limited set of values - classes. In scikit-learn it is DecisionTreeClassifier.
2. Regression trees used to assign samples into numerical values within the range. In scikit-learn it is DecisionTreeRegressor.

4 ways to visualize Decision Tree in Python:
    1. print text representation of the tree with sklearn.tree.export_text method
    2. plot with sklearn.tree.plot_tree method (matplotlib needed)
    3. plot with sklearn.tree.export_graphviz method (graphviz needed)
    4. plot with dtreeviz package (dtreeviz and graphviz needed)"""

##  Decision Tree on CLASSIFICATION Task-------------------------
# DATA SET to be used for DECISION TREE
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
# Prepare the data data
iris = datasets.load_iris()
X = iris.data # array
y = iris.target # array
# Fit the classifier with default hyper-parameters
clf = DecisionTreeClassifier(random_state=1234)
model = clf.fit(X, y)
# METHOD 1:
# Exporting Decision Tree to the text representation
text_representation = tree.export_text(clf)
print(text_representation)

# METHOD 2:
# Print Text Representation
fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(clf,
                   feature_names=iris.feature_names,
                   class_names=iris.target_names,
                   filled=True)

# METHOD 3:
# Using graphviz
#METHOD 4:
#Decision Tree with dtreeviz Package
#The dtreeviz package is available in github. It can be installed with pip install dtreeviz.
# It requires graphviz to be installed (but you dont need to manually convert between DOT files and images). To plot the tree just run:

##  Decision Tree on REGRESSION  Task-------------------------
from sklearn import datasets
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
# Prepare the data data
boston = datasets.load_boston() # <class 'sklearn.utils.Bunch'>
X = boston.data # <class 'numpy.ndarray'>
y = boston.target # <class 'numpy.ndarray'>
# To keep the size of the tree small, I set max_depth = 3.
# Fit the regressor, set max_depth = 3
regr = DecisionTreeRegressor(max_depth=3, random_state=1234)
regr.fit(X, y)

# METHOD1:
text_representation = tree.export_text(regr)
print(text_representation)

# METHOD2:
fig = plt.figure(figsize=(20,10))
_ = tree.plot_tree(regr, feature_names=boston.feature_names, filled=True)
# ********************************************************************************************************************







