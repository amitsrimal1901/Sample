## https://realpython.com/logistic-regression-python/
# Classification is among the most important areas of machine learning, and logistic regression is one of its basic methods.
# Classification is a very important area of supervised machine learning.
# Classification is an area of supervised machine learning that tries to predict which class or category some entity belongs to, based on its features.
"""
Supervised machine learning algorithms analyze a number of observations and try to mathematically express the dependence between the inputs and outputs.
These mathematical representations of dependencies are the MODELS.

The nature of the dependent variables differentiates regression and classification problems.
Regression problems have continuous and usually unbounded outputs. An example is when you’re estimating the salary as a function of experience and education level.
While, classification problems have discrete and finite outputs called classes or categories.
For example, predicting if an employee is going to be promoted or not (true or false) is a classification problem.

There are two main types of classification problems:
    1. Binary or binomial classification: exactly two classes to choose between (usually 0 and 1, true and false, or positive and negative)
    2. Multiclass or multinomial classification: three or more classes of the outputs to choose from.

If there’s only one input variable, then it’s usually denoted with 𝑥.
For more than one input, you’ll commonly see the vector notation 𝐱 = (𝑥₁, …, 𝑥ᵣ), where 𝑟 is the number of the predictors (or independent features).
The output variable is often denoted with 𝑦 and takes the values 0 or 1.

## WHAT IS LOGISTIC CLASSIFICATION
Logistic regression is a fundamental classification technique.
It belongs to the group of linear classifiers and is somewhat similar to polynomial and linear regression.
We use a linear function 𝑓(𝐱) = 𝑏₀ + 𝑏₁𝑥₁ + ⋯ + 𝑏ᵣ𝑥ᵣ, also called the logit and calculate log to derive logistic function.
Logistic regression is fast and relatively uncomplicated, and it’s convenient for you to interpret the results.
Although it’s essentially a method for binary classification, it can also be applied to multiclass problems.

Binary classification has four possible types of results:
    1. True positives: correctly predicted positives (ones)
    2. True negatives: correctly predicted negatives (zeros)
    3. False positives: incorrectly predicted positives (ones)
    4. False negatives: incorrectly predicted negatives (zeros)

The most straightforward indicator of classification ACCURACY is the ratio of the number of correct predictions to the total number of predictions (or observations).
Other indicators of binary classifiers include the following:
    1. The positive predictive value is the ratio of the number of true positives to the sum of the numbers of true and false positives.
    2. The negative predictive value is the ratio of the number of true negatives to the sum of the numbers of true and false negatives.
    3. The SENSITIVITY (also known as RECALL or true positive rate) is the ratio of the number of true positives to the number of actual positives.
    4. The SPECIFICITY (or true negative rate) is the ratio of the number of true negatives to the number of actual negatives.
NOTE: The most suitable indicator depends on the problem of interest.

SINGLE VARIATE logistic regression is the most straightforward case of logistic regression.
There is only one independent variable (or feature), which is 𝐱 = 𝑥.

MULTI VARIATE logistic regression has more than one input variable.

REGUALRIZATION:
Overfitting is one of the most serious kinds of problems related to machine learning.
The model then learns not only the relationships among data but also the noise in the dataset.
Overfitting usually occurs with complex models. Regularization normally tries to reduce or penalize the complexity of the model.

Regularization techniques applied with logistic regression mostly tend to penalize large coefficients 𝑏₀, 𝑏₁, …, 𝑏ᵣ:
    1. L1 regularization penalizes the LLF with the scaled sum of the absolute values of the weights: |𝑏₀|+|𝑏₁|+⋯+|𝑏ᵣ|.
    2. L2 regularization penalizes the LLF with the scaled sum of the squares of the weights: 𝑏₀²+𝑏₁²+⋯+𝑏ᵣ².
    3. Elastic-net regularization is a linear combination of L1 and L2 regularization.
Regularization can significantly improve model performance on unseen data.
"""

#---------------------Logistic Regression in Python With scikit-learn: Example 1 ----------------------#
## Step 1: Import Packages, Functions, and Classes
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

## Step 2: Get Data
# The input and output should be NumPy arrays
x = np.arange(10).reshape(-1, 1)
print(type(x))  #<class 'numpy.ndarray'>
# The array x is required to be two-dimensional. It should have one column for each input, and the number of rows should be equal to the number of observations.
# To make x two-dimensional, you apply .reshape() with the arguments -1 to get as many rows as needed and 1 to get one column.
# x has two dimensions:
    # One column for a single input
    # Ten rows, each corresponding to one observation
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1]) #y is one-dimensional with ten items

## Step 3: Create a Model and Train It
model = LogisticRegression(solver='liblinear', random_state=0)
# solver is a string ('liblinear' by default) that decides what solver to use for fitting the model. Other options are 'newton-cg', 'lbfgs', 'sag', and 'saga'.
# random_state is an integer, an instance of numpy.RandomState, or None (default) that defines what pseudo-random number generator to use.

# Once the model is created, you need to fit (or train) it.
# Model fitting is the process of determining the coefficients 𝑏₀, 𝑏₁, …, 𝑏ᵣ that correspond to the best value of the cost function.
model.fit(x, y)

## Alternatively, we can combine all above steps as --> model = LogisticRegression(solver='liblinear', random_state=0).fit(x, y)

## At this point, we have the classification model defined we can quickly get the attributes of our model
model.classes_ # array([0, 1]) # This is the example of binary classification, and y can be 0 or
# get slope b1  and intercept 𝑏₀ of linear finction
model.intercept_ # array([-1.04608067])
model.coef_ #array([[0.51491375]])
## NOTE: 𝑏₀ is given inside a one-dimensional array, while 𝑏₁ is inside a two-dimensional array.

## Step 4: Evaluate the Model
# Once a model is defined, you can check its performance with .predict_proba(), which returns the matrix of probabilities that the predicted output is equal to zero or one.
model.predict_proba(x)
    # in the o/p each row corresponds to a single observation.
    # The first column is the probability of the predicted output being zero, that is 1 - 𝑝(𝑥).
    # The second column is the probability that the output is one, or 𝑝(𝑥).

# get the actual predictions, based on the probability matrix and the values of 𝑝(𝑥)
model.predict(x) # This function returns the predicted output values as a one-dimensional array.

#  the accuracy of your model
model.score(x, y) # .score() takes the input and output as arguments and returns the ratio of the number of correct predictions to the number of observations.

# get more information on the accuracy of the model with a confusion matrix
confusion_matrix(y, model.predict(x))
"""
True negatives in the upper-left position
False negatives in the lower-left position
False positives in the upper-right position
True positives in the lower-right position
"""

## Step 5:  to visualize the confusion matrix.
# We will do that with .imshow() from Matplotlib
cm = confusion_matrix(y, model.predict(x))

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cm)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
plt.show()
## creates a heatmap that represents the confusion matrix

# more comprehensive report on the classification with classification_report()
# function also takes the actual and predicted outputs as arguments.
# It returns a report on the classification as a dictionary if you provide output_dict=True or a string otherwise.
print(classification_report(y, model.predict(x)))

## STEP 6: Improve the Model
# We can improve your model by setting different parameters.
# For example, let’s work with the regularization strength C equal to 10.0, instead of the default value of 1.0:
model = LogisticRegression(solver='liblinear', C=10.0, random_state=0)
model.fit(x, y)
## Now we have another model with different parameters. It’s also going to have a different probability matrix and a different set of coefficients and predictions
model.intercept_
model.coef_
model.predict_proba(x)
model.predict(x)
model.score(x, y)
confusion_matrix(y, model.predict(x))
print(classification_report(y, model.predict(x)))
# The score (or accuracy) of 1 and the zeros in the lower-left and upper-right fields of the confusion matrix indicate that the actual and predicted outputs are the same.

#---------------------Logistic Regression in Python With scikit-learn: Example 2 ----------------------#
# another classification problem. It’s similar to the previous one, except that the output differs in the second value
# Step 1: Import packages, functions, and classes
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Step 2: Get data
x = np.arange(10).reshape(-1, 1)
y = np.array([0, 1, 0, 0, 1, 1, 1, 1, 1, 1])

# Step 3: Create a model and train it
model = LogisticRegression(solver='liblinear', C=10.0, random_state=0)
model.fit(x, y)

# Step 4: Evaluate the model
p_pred = model.predict_proba(x)
y_pred = model.predict(x)
score_ = model.score(x, y)
conf_m = confusion_matrix(y, y_pred)
report = classification_report(y, y_pred)

print('x:', x, sep='\n')
print('y:', y, sep='\n', end='\n\n')
print('intercept:', model.intercept_)
print('coef:', model.coef_, end='\n\n')
print('p_pred:', p_pred, sep='\n', end='\n\n')
print('score_:', score_, end='\n\n')
print('conf_m:', conf_m, sep='\n', end='\n\n')
print('report:', report, sep='\n')
# In this case, the score (or accuracy) is 0.8.
#There are two observations classified incorrectly. One of them is a false negative, while the other is a false positive.
"""
one important characteristic of this example is not linearly separable.
That means you can’t find a value of 𝑥 and draw a straight line to separate the observations with 𝑦=0 and those with 𝑦=1. 
There is no such line.
Keep in mind that logistic regression is essentially a linear classifier, so you theoretically can’t make a logistic regression model with an accuracy of 1 in this case.
"""











# #*************************************************************************************************************************
## Logistic regression on Edureka## 
# produces result in binary format which is used to predict the outcome of a categorical dependent variabble.
# so the outcome is discrete such as 0/1, T/F, Yes/No, High/Low etc.
# Generally represented by Sigmoid (S curve).
# intermediate values are entertained by THRESHOLD of S curve. value above Threshold is 1, else 0.
## usecases of Logictic regression: Waeather prediction like sunny of not,classification, illness determination etc. 

## Deriving Logistic from Linear equation
#LINEAR: y=c+m1x1+m2m2+..... hence (y) range is -inf to +inf.
#LOGISTIC:  log(y/1-y)=d+n1x1+n2m2+..... hence range is0 to 1.

## TITANIC data set ------------------------------------------------------------
# what factors made eople more likely of surviving the sinking
# Steps of logistic Regression:
    # Step1: Collecting & analysing data
    # Step2: data wrangling
    # step3: training & testing
    # Step4: Accuracy Check

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
from sklearn.cross_validation import train_test_split
from sklearn import linear_model 
## collecting & analysisng data
titanic_data= pd.read_csv("C:/Users/Amit Srimal/Desktop/DET/ML/Data/Titanic/train.csv")
titanic_data.head(10) # ten first rows of titanic data

## splitting 'name' column into first&lastname
titanic_data.loc[titanic_data['Embarked']=='S']
titanic_data.loc[titanic_data['Name']] # getting name dataframe here

pass_name= titanic_data.iloc[:,3] # creating data frame of name from parentdataframe
pass_name=pass_name.str.split(',',n=1,expand=True)
pass_name.columns=['firstName','lastName'] # renaming column names frpm 0 & 1
print(pass_name.columns) # first & last name

#plotting Emarked status
sns.countplot(x='Embarked', data=titanic_data)
sns.countplot(x='Embarked',hue='Pclass', data=titanic_data)
#plotting survived 0/1 
sns.countplot(x='Survived', data=titanic_data) 
#plotting genderwise for survival
sns.countplot(x='Survived',hue='Sex', data=titanic_data) 
sns.countplot(x='Sex',hue='Survived', data=titanic_data) #swapping the gender & survival
sns.countplot(x='Survived',hue='Pclass', data=titanic_data) # plotting survival with passenger class
titanic_data["Age"].plot.hist() # age plot for data
titanic_data["Fare"].plot.hist() # fare plot for data
titanic_data["Fare"].plot.hist(binsize=20,figsize=(10,5)) # chnaging bin figure size for plot
#getting insights into the data information
titanic_data.info()
sns.countplot(x='SibSp',data=titanic_data) # plot for sibling & spouse
sns.countplot(x='Parch',data=titanic_data) # plot for parent & children

## data wrangling: data cleaning, remove unnecessary column with nan values
titanic_data.isnull() # retuns boolean T/F
titanic_data.isnull().sum() # retuns sum of True . ie. Nan values
sns.heatmap(titanic_data.isnull(),yticklabels=False) # represent graphically nan instances
sns.heatmap(titanic_data.isnull(),yticklabels=True) # same as above but with y marked index
sns.heatmap(titanic_data.isnull(),yticklabels=False,cmap='viridis') # colorcode the map for above
#IMP: nan needs to replaced with dummy values or columns needs removing.
sns.boxplot(x='Pclass',y='Age',data=titanic_data) # comparison of age vs class for titanic data
# dropping cabin column having nan values
titanic_data.drop('Cabin', axis=1,inplace=True) # update df with cabin removed
titanic_data.dropna(inplace=True) # removing nan from resultant df 
sns.heatmap(titanic_data.isnull(),yticklabels=False,cbar=False) # this has ZERO nan as graph is all BLACK.
titanic_data.isnull().sum() # check for sum of NULL
titanic_data.info() # earlier 891 & now 712 rows
## next we need to convert string to categorical like name,sex field etc.
pd.get_dummies(titanic_data['Sex'])
sex=pd.get_dummies(titanic_data['Sex'],drop_first=True) # male is 1 &female is 0
embark=pd.get_dummies(titanic_data['Embarked'])
# we can drop one C as 0/1 from Q,S is enough to tell embarkment place.
embark=pd.get_dummies(titanic_data['Embarked'],drop_first=True)
embark.head(10)
Pcl=pd.get_dummies(titanic_data['Pclass'],drop_first=True)
Pcl.head(10)
## Next we need to concatenate these columns
titanic_data=pd.concat([titanic_data,sex,embark,Pcl],axis=1)
# lets drop duplicated column +other which are not needed as below
titanic_data.drop(['Sex','PassengerId','Embarked','Name','Ticket','Pclass'],axis=1,inplace=True)

## getting training & testing data set
# take out Survived from data set as mark X i/p, take survided as Y o/p
x=titanic_data.drop('Survived',axis=1)
y=titanic_data['Survived']
# splitting training & testing on some ratio &seed size
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.30,random_state=42) # 498+214 SPLIT ratio
# creating logistic model 
logmodel=linear_model.LogisticRegression()
# passing & fitting training data to model
logmodel.fit(x_train,y_train)
# passing testing data to predict outconmes based on crated model behaviour
predictions=logmodel.predict(x_test) # predicted values obained from x_test
print(predictions)

## checking precision of predictions
from sklearn.metrics import classification_report
classification_report(y_test,predictions) # comparing acuracy by y_test vs predicted values
print(classification_report(y_test,predictions)) # gets precision, recall,f1score,support

## checking accuracy of predictions: CONFUSION MATRIX
from sklearn.metrics import confusion_matrix
cm= confusion_matrix(y_test,predictions)
print(confusion_matrix(y_test,predictions))
107+62+15+30 # 214 which is test data count
#accuracy= 169/214
# Alternatively from library
from sklearn.metrics import accuracy_score
score=accuracy_score(y_test,predictions) # ~ 79 %
print(score)

## Logistic regresson from MTCARS cars data set --------------------------------
# ref link: https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import scipy
from scipy.stats import spearmanr
from pylab import rcParams
import seaborn as sb
import sklearn
from sklearn.preprocessing import scale
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing 

rcParams('figure.figsize')=5,4
sb.set_style('whitegrid')
# reading mtcars data set from local
cars= pd.read_csv('D:/PythonReferences/mtcars.csv') 
cars.columns # returns name of all columns from imported data set
cars.head(3) # retruns first 3 row data from cars data set.
# randomly select variable & see the correlation.
cars_data = cars.iloc[:,[5,11]].values
cars_data_names = ['drat','carb']
y= cars.iloc[0:9].values
# Checking for Independecne between featires
sb.regplot(x='drat',y='carb', data=cars, scatter=True)
drat=cars['drat']
carb=cars['carb']
spearmanr_coefficient, p_value=spearmanr(drat,carb)
print('Sparmanr coefficient  %0.3f' % (spearmanr_coefficient)) # o/p says Sparmanr coefficient  -0.125
# IMP: States almost No corelation b.w drat & carb.
##checking fr missing values
cars.isnull().sum()  #all columns says ZERO.
## Checking that target is binary or ordinal
sb.countplot(x='am', data=cars, palette='hls')
## Chekcing size of data set used for modelling.
cars.info()

## Deploying & model evaluation
X= scale(cars_data)
logmodel=LogisticRegression()
logmodel.fit(X,y)
# checking R squared value 
logmodel.score(X,y)
print(logmodel.score(X,y)) # printing log model R squered value. O/p is .8125
# checking accuracy for model
y_pred= logmodel.predict(X)
from sklearn.metrics import classfication_report 
print( classification_report(y, y_pred))
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y,y_pred))


## Direct marketing campaigns (phone calls) of a Portuguese banking  -----------
# Goalisto predict whether the client will subscribe (1/0) to a term deposit (variable y).
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

# reading & analysing data
data=pd.read_csv('D:/PythonReferences/banking.csv', header=0) # header value 0 takes firt row as header of dataset.
data.isnull().sum() # checking for nan in data set
data=data.dropna() # dropping nan from data set
print(data.shape) # 41188 by 21 
print(list(data.columns)) # gets name of 21 column fields

# The education column has many categories and we need to reduce the categories for a better modelling
data['education'].unique() # returns unique educational values from column
# Let us group “basic.4y”, “basic.9y” and “basic.6y” together and call them “basic”.
data['education']=np.where(data['education'] =='basic.9y', 'Basic', data['education'])
data['education']=np.where(data['education'] =='basic.6y', 'Basic', data['education'])
data['education']=np.where(data['education'] =='basic.4y', 'Basic', data['education'])
data['education'].unique() # reurns unique with BASIC inplace of 4,9,6 basic.py

# Data exploration
data['y'].value_counts() # y refers loan accpeted Y or N. o/p is 1-36548 & 0-4640.
sns.countplot(x='y', data=data, palette='hls')
# Let’s get a sense of the numbers across the two classes.
data.groupby('y').mean() # returns mean value of all coluns grouped by Y target.
## calculate categorical means for other categorical variables say education, job, status etc.
data.groupby('job').mean()
data.groupby('marital').mean()
data.groupby('education').mean()

## Visualizations
# Here we will plot the above grouped data & check strength of varibales.
pd.crosstab(data.job,data.y).plot(kind='bar') # plotting job based on 0/1
plt.title('Purchase Frequency for Job Title')
plt.xlabel('Job')
plt.ylabel('Frequency of Purchase')
#plt.savefig('purchase_fre_job')
plt.show()
#  frequency of purchase of the deposit depends a great deal on the job title. 
# Thus, the job title can be a good predictor of the outcome variable.
# Now checking for Marital status
table=pd.crosstab(data.marital,data.y)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of Marital Status vs Purchase')
plt.xlabel('Marital Status')
plt.ylabel('Proportion of Customers')
plt.show() #conclusion its nota good predictor.
# Now checking for educatin
table=pd.crosstab(data.education,data.y)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of Education Status vs Purchase')
plt.xlabel('Education level')
plt.ylabel('Proportion of Customers')
plt.show() #conclusion its a good predictor.
# checking for day of week.
pd.crosstab(data.day_of_week,data.y).plot(kind='bar')
plt.title('Purchase Frequency for Day of Week')
plt.xlabel('Day of Week')
plt.ylabel('Frequency of Purchase')
plt.show() # may not be a good predictor
# llly checking for month:
pd.crosstab(data.month,data.y).plot(kind='bar')
plt.title('Purchase Frequency for Month')
plt.xlabel('Month')
plt.ylabel('Frequency of Purchase')
plt.show() # good signal for modelling
# lets create hostogram on AGE parameter.
data.age.hist()
plt.title('Histogram of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()
# checking for poutcome
pd.crosstab(data.poutcome,data.y).plot(kind='bar')
plt.title('Purchase Frequency for Poutcome')
plt.xlabel('Poutcome')
plt.ylabel('Frequency of Purchase')
plt.show() # hence poutocme os good predictor here.
 
## Create dummy variables
# That is variables with only two values, zero and one.
cat_vars=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
for var in cat_vars:
    cat_list='var'+'_'+var
    cat_list = pd.get_dummies(data[var], prefix=var)
    data1=data.join(cat_list)
    data=data1

cat_vars=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
data_vars=data.columns.values.tolist()
to_keep=[i for i in data_vars if i not in cat_vars]    
data_final=data[to_keep]
data_final.columns.values

data_final_vars=data_final.columns.values.tolist()
y=['y']
X=[i for i in data_final_vars if i not in y]

## Feature Selection
# RFE: recursive feature eimination : based on the idea to repeatedly construct a model and choose either the best or worst performing feature, setting the feature aside and then repeating the process with the rest of the features
# This process is applied until all features in the dataset are exhausted.
# The goal of RFE is to select features by recursively considering smaller and smaller sets of features.
from sklearn import datasets
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
rfe = RFE(logreg, 18)
rfe = rfe.fit(data_final[X], data_final[y] )
print(rfe.support_)
print(rfe.ranking_)

cols=["previous", "euribor3m", "job_blue-collar", "job_retired", "job_services", "job_student", "default_no", 
      "month_aug", "month_dec", "month_jul", "month_nov", "month_oct", "month_sep", "day_of_week_fri", "day_of_week_wed", 
      "poutcome_failure", "poutcome_nonexistent", "poutcome_success"] 
X=data_final[cols]
y=data_final['y']

## Implementing the model
import statsmodels.api as sm
logit_model=sm.Logit(y,X)
result=logit_model.fit()
print(result.summary())
# IMP: The p-values for most of the variables are smaller than 0.05, therefore, 
# most of them are significant to the model.

## Logistic Regression Model Fitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
# Predicting the test set results and calculating the accuracy
y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
# Accuracy of logistic regression classifier on test set: 0.90

## Cross Validation
# Cross validation attempts to avoid overfitting while still producing a prediction for each observation datase
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
kfold = model_selection.KFold(n_splits=10, random_state=7)
modelCV = LogisticRegression()
scoring = 'accuracy'
results = model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)
print("10-fold cross validation average accuracy: %.3f" % (results.mean()))
# Here : 10-fold cross validation average accuracy: 0.897
# The average accuracy remains very close to the Logistic Regression model accuracy; hence, we can conclude that our model generalizes well.
## Confusion Matrix:
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

## Compute precision, recall, F-measure and support
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
##Interpretation: Of the entire test set, 88% of the promoted term deposit were the term deposit that the customers liked.
#Of the entire test set, 90% of the customer’s preferred term deposits that were promoted.

## ROC Curve: receiver operating characterstics
# ROC is a common tool used with binary classifiers
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()
# The dotted line represents the ROC curve of a purely random classifier
# a good classifier stays as far away from that line as possible (toward the top-left corner)



































