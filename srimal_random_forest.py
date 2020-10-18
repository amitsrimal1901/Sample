"""
ENSEMBLE methods are techniques that create multiple models and then combine them to produce improved results.
Ensemble methods usually produces more accurate solutions than a single model would.
This has been the case in a number of machine learning competitions, where the winning solutions used ensemble methods.

The term “model” to describe the output of the algorithm that trained with data.
This model is then used for making predictions. This algorithm can be any machine learning algorithm such as logistic regression, decision tree, etc.
These models, when used as inputs of ensemble methods, are called ”BASE MODELS”.
Some of the most used ensemble techniques are 1. voting, 2. stacking, 3. bagging and 4. boosting.

METHOD 1: Voting and Averaging Based Ensemble Methods
These are the easiest ensemble methods in terms of understanding and implementation.
Voting is used for classification and averaging is used for regression.

In both methods, the first step is to create multiple classification/regression models using some training dataset.
Each base model can be created using different splits of the same training dataset and same algorithm, or using the same dataset with different algorithms, or any other method.
1.1 MAJORITY VOTING
Every model makes a prediction (votes) for each test instance and the final output prediction is the one that receives more than half of the votes.
If none of the predictions get more than half of the votes, we may say that the ensemble method could not make a stable prediction for this instance.
Although this is a widely used technique, you may try the most voted prediction (even if that is less than half of the votes) as the final prediction.
This method is also being called “plurality voting”.
1.2 WEIGHTED VOTING
Unlike majority voting, where each model has the same rights, we can increase the importance of one or more models.
In weighted voting you count the prediction of the better models multiple times. Finding a reasonable set of weights is up to you.
1.3 SIMPLE AVERAGING
In simple averaging method, for every instance of test dataset, the average predictions are calculated.
This method often reduces overfit and creates a smoother regression model.
1.4 WEIGHTED AVERAGING
Weighted averaging is a slightly modified version of simple averaging, where the prediction of each model is multiplied by the weight and then their average is calculated.

----------RANDOM FOREST IS AN ENSEMBLE OF DECISION TREE ie DECISON TREE IS BASE MODEL----------------------

FUTURE Reads:
https://www.toptal.com/machine-learning/ensemble-methods-machine-learning#:~:text=Ensemble%20methods%20are%20techniques%20that,winning%20solutions%20used%20ensemble%20methods.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset= pd.read_csv('https://raw.githubusercontent.com/amitsrimal1901/Machine_Learning_A-Z/master/Part%202%20-%20Regression/Section%209%20-%20Random%20Forest%20Regression/Position_Salaries.csv')
dataset.shape # (10, 3)
dataset.describe(),  dataset.info()
dataset.isnull().any()

# Splitting feature and targets variables
# Select all rows and column 1 from dataset to x and all rows and column 2 as y
X = dataset.iloc[:, 1:2].values # <class 'numpy.ndarray'>, shape (10,1)
y = dataset.iloc[:, 2].values #   # <class 'numpy.ndarray'>, shape (1,)

"""
# Splitting in train, test data set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.3, random_state=0)
# type is <class 'numpy.ndarray'>
# X_train size is 7, x_test size is 3

# Now lets apply scaler featur here
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test) ## no fitting required as its a one time activity for a given feature, label
# Similarly we transform y train
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

## now fitting the random forest model
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=100, random_state=1)
"""
n_estimators : This is the number of trees you want to build before taking the maximum voting or averages of predictions. 
Higher number of trees give you better performance but makes your code slower.
"""
regressor.fit(X, y)

# Now predicting a new level
regressor.predict([[6.5]]) # array([156100.])
# ALTERNATIVELY To get a more cleaner o/p use
regressor.predict(np.array([6.5]).reshape(1, 1))

""""#NEEDS TO BE CORRECTED
# Follow: https://towardsdatascience.com/random-forest-in-python-24d0893d51c0
## Plotting the random forest
# Remove the labels from the features, axis 1 refers to the columns
features = dataset.drop(['Position', 'Salary'], axis = 1) # just retruns LEVEL as feature column list
# Saving feature names for later use
feature_list = list(features.columns) # ['Level'] , <class 'list'>

# Extract single tree
estimator = regressor.estimators_[5]
# values of Estimator is as: DecisionTreeRegressor(max_features='auto', random_state=550290313)

from sklearn.tree import export_graphviz
import pydot
# Export as dot file
export_graphviz(estimator, out_file='tree.dot',
                feature_names = np.array(feature_list),
                rounded = True, proportion = False,
                precision = 1, filled = True)
(graph, ) = pydot.graph_from_dot_file('tree.dot')
graph.write_png('C:/tree.png')
"""

#Plotting the data here in full
X_grid = np.arange(min(X),max(X),0.01) # array
X_grid = X_grid.reshape((len(X_grid)),1) # <class 'numpy.ndarray'> of size (900, 1)
plt.scatter(X,y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Salary Predictor')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()



#************************************************************************************************

# Random forest from https://towardsdatascience.com/random-forest-in-python-24d0893d51c0
# PROBLEM STATEMENET: predicting the max temperature for tomorrow in our city using one year of past weather data
import pandas as pd
import numpy as np
dataset = pd.read_csv('C:/Users/amit_srimal/Documents/Study/Python/Files/Weather.csv') # [348 rows x 12 columns]
# removing 3 columns not mentiioned in the post
dataset = dataset.loc[:,['year', 'month', 'day', 'week', 'temp_2', 'temp_1', 'average', 'actual', 'friend']]
# temp_2: max temperature 2 days prior
# temp_1: max temperature 1 day prior

#Handling missing data
dataset.info()  # 348 observations vs 365 days of 2016
dataset.describe()
# Conclusion: NOt much data missing and hence shoulnt give major deviation to prediction.

# Data Preparation
# Since data is categorical like weeks Ond/ Tue, Wed etc hence we use ENCODING
# One-hot encode the data using pandas get_dummies
dataset = pd.get_dummies(dataset) # eek_Sun  week_Thurs  week_Tues  week_Wed added in as DUMMY
dataset.head(5) # has got 15 ROWS
#lets avoid dummy trap by dropping ONE colimn say last columnhere
dataset = dataset.iloc[:,0:14] # now has 17 ROWS
dataset.columns # 6 days of week and one dummy day removed
#  'week_Fri', 'week_Mon', 'week_Sat', 'week_Sun', 'week_Thurs', 'week_Tues'

# Features and Targets and Convert Data to Arrays
# Labels are the values we want to predict
y = np.array(dataset['actual'])
# Remove the labels from the features,axis 1 refers to the columns
features= dataset.drop('actual', axis = 1)
# Saving feature names for later use
feature_list = list(features.columns)
# Convert to numpy array
X = np.array(features) # Alternatively called

##Training and Testing Sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.25, random_state = 42) # 261 training, 87 testing

## Establish Baseline
"""
Before we can make and evaluate predictions, we need to establish a baseline, a sensible measure that we hope to beat with our model. 
If our model cannot improve upon the baseline, then it will be a failure and we should try a different model or admit that machine learning is not right for our problem. 
The baseline prediction for our case can be the historical max temperature averages.
In other words, our baseline is the error we would get if we simply predicted the average max temperature for all days.
"""
# The baseline predictions are the historical averages
baseline_preds = X_test[:, feature_list.index('average')]
# Baseline errors, and display average baseline error
baseline_errors = abs(baseline_preds - y_test)
print('Average baseline error: ', round(np.mean(baseline_errors), 2)) # Average baseline error:  5.06 degrees.
# which means If we can’t beat an average error of 5 degrees, then we need to rethink our approach.

# Train Model
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
rf.fit(X_train, y_train)

# Make Predictions on the Test Set
predictions = rf.predict(X_test)
# Calculate the absolute errors
errors = abs(predictions - y_test) ## array
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.') # Mean Absolute Error: 3.83 degrees.
"""Our average estimate is off by 3.83 degrees. That is more than a 1 degree average improvement over the baseline. 
Although this might not seem significant, it is nearly 25% better than the baseline, which, 
depending on the field and the problem, could represent millions of dollars to a company"""

#  MOdel performamce
# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / y_test)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.') # Accuracy: 94.03 %.

## Visualizing a Single Decision Tree
"""
# Import tools needed for visualization
from sklearn.tree import export_graphviz
import pydot
# Pull out one tree from the forest
tree = rf.estimators_[5] # <class 'sklearn.tree._classes.DecisionTreeRegressor'>

# Export the image to a dot file
export_graphviz(tree, out_file = 'C:/Users/amit_srimal/Documents/Study/Python/Files/tree.dot', feature_names = feature_list, rounded = True, precision = 1)
print('a')
# Use dot file to create a graph
(graph, ) = pydot.graph_from_dot_file('C:/Users/amit_srimal/Documents/Study/Python/Files/tree.dot')
print('b')
# Write graph to a png file
graph.write_png('tree.png')
print('c')
# USE ONLINE TOOL TP CONVERT DOT FILE TO PNG, if issue persists
# use "http://viz-js.com/" to view the image file from dot file source

CONCLUSION from PNG 
# prediction for any new data point based solely on ABOVE TREE,.
1. Let’s take an example of making a prediction for Wednesday, December 27, 2017. (OUR DATA HAS 2016 records only)
2. The (actual) variables are: temp_2 = 39, temp_1 = 35, average = 44, and friend = 30. 
3. We start at the root node and the first answer is True because temp_1 ≤ 59.5. 
4. We move to the left and encounter the second question, which is also True as average ≤ 46.8. 
5. Move down to the left and on to the third and final question which is True as well because temp_1 ≤ 44.5. 
6. Therefore, we conclude that our estimate for the maximum temperature is 41.0 degrees as indicated by the value in the leaf node.

7. An interesting observation is that in the root node, there are only 162 samples despite there being 261 training data points.
8. This is because each tree in the forest is trained on a random subset of the data points with replacement (called bagging, short for bootstrap aggregating).
9. Random sampling of data points, combined with random sampling of a subset of the features at each node of the tree, is why the model is called a ‘random’ forest.

10. notice that in our tree, there are only 2 variables we actually used to make a prediction! 
11. According to this particular decision tree, the rest of the features are not important for making a prediction.
12. Month of the year, day of the month, and our friend’s prediction are utterly useless for predicting the maximum temperature tomorrow! 
13. The only important information according to our simple tree is the temperature 1 day prior and the historical average

"""
# Limit depth of tree to 3 levels
from sklearn.tree import export_graphviz
import pydot
rf_small = RandomForestRegressor(n_estimators=10, max_depth = 3)
rf_small.fit(X_train, y_train)
# Extract the small tree
tree_small = rf_small.estimators_[5]
# Save the tree as a png image
export_graphviz(tree_small, out_file = 'C:/Users/amit_srimal/Documents/Study/Python/Files/small_tree.dot', feature_names = feature_list, rounded = True, precision = 1)
print('a')
(graph, ) = pydot.graph_from_dot_file('C:/Users/amit_srimal/Documents/Study/Python/Files/small_tree.dot')
print('b')
graph.write_png('C:/Users/amit_srimal/Documents/Study/Python/Files/small_tree.png');
print('c')
# USE ONLINE TOOL TP CONVERT DOT FILE TO PNG, if issue persists
# use "http://viz-js.com/" to view the image file from dor file source

## VARIABLE IMPORTANCE
#NOTE: Depth of algos not required at this stage
#In order to quantify the usefulness of all the variables in the entire random forest, we can look at the relative importances of the variables.
# The importances returned in Skicit-learn represent how much including a particular variable improves the prediction.
# Get numerical feature importances
importances = list(rf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(dataset, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];
"""
Variable: temp_1               Importance: 0.7
Variable: average              Importance: 0.19
Variable: day                  Importance: 0.03
Variable: temp_2               Importance: 0.02
Variable: friend               Importance: 0.02
Variable: month                Importance: 0.01
Variable: year                 Importance: 0.0
Variable: week_Fri             Importance: 0.0
Variable: week_Mon             Importance: 0.0
Variable: week_Sat             Importance: 0.0
Variable: week_Sun             Importance: 0.0
Variable: week_Thurs           Importance: 0.0
Variable: week_Tues            Importance: 0.0
Variable: week_Wed             Importance: 0.0
"""
#At the top of the list is temp_1, the max temperature of the day before.
# This tells us the best predictor of the max temperature for a day is the max temperature of the day before, a rather intuitive finding.
# The second most important factor is the historical average max temperature,
# In future implementations of the model, we can remove those variables that have no importance and the performance will not suffer.
# ADDITIONALLY, if we are using a different model, say a support vector machine, we could use the random forest feature importances as a kind of feature selection method.

"""
So next we can MAKE A RANDOM FOREST with only the TWO MOST IMPORTANT VARIABLES
    1. the max temperature 1 day prior and 
    2. the historical average and see how the performance compares.
And we get performance as     
    1. Mean Absolute Error: 3.9 degrees.
    2. Accuracy: 93.8 %.
This implies we actually do not need all the data we collected to make accurate predictions! 
If we were to continue using this model, we could only collect the two variables and achieve nearly the same performance

PATH AHEAD:
Frm this point, if we want to improve our model, we could try different hyperparameters (settings) try a different algorithm, 
or the best approach of all, gather more data! The performance of any model is directly proportional to the amount of valid data it can learn from, 
and we were using a very limited amount of information for training.

###########  Random Forests vs Decision Trees
                1. Random forests is a set of multiple decision trees.
                2. Deep decision trees may suffer from overfitting, but random forests prevents overfitting by creating trees on random subsets.
                3. Decision trees are computationally faster.
                4. Random forests is difficult to interpret, while a decision tree is easily interpretable and can be converted to rules.
"""



