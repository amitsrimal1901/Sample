########  ** MODEL ** ########
#________________________________________________________________________________
# Regression searches for relationships among variables.
"""
In ML models, the dependent variables as RESPONSE and independent variables as FEATURES.
Consider several employees of some company and try to understand how their salaries(RESPONSE) depend on the FEATURES, such as experience, level of education, role, city they work in, and so on.
This is a regression problem where data related to each employee represent one OBSERVATION.

The dependent features are called the dependent variables, outputs, or responses.
The independent features are called the independent variables, inputs, or predictors.

Regression problems usually have one continuous and unbounded dependent variable.
The inputs, however, can be continuous, discrete, or even categorical data such as gender, nationality, brand, and so on.

# Why Regression is needed?
We need regression to answer whether and how some phenomenon influences the other or how several variables are related.
Among regression, LINEAR REGRESSION is the simplest regression methods.
One of its main advantages is the ease of interpreting results.

The estimated or predicted response, 𝑓(𝐱ᵢ), for each observation 𝑖 = 1, …, 𝑛, should be as close as possible to the corresponding actual response 𝑦ᵢ.
The differences 𝑦ᵢ - 𝑓(𝐱ᵢ) for all observations 𝑖 = 1, …, 𝑛, are called the RESIDUALS.
Regression is about determining the best predicted weights, that is the weights corresponding to the smallest residuals.

To get the best weights, you usually minimize the sum of squared residuals (SSR) for all observations 𝑖 = 1, …, 𝑛: SSR = Σᵢ(𝑦ᵢ - 𝑓(𝐱ᵢ))².
This approach is called the method of ORDINARY LEAST SQUARES.

REGRESSION PERFORMANCE:
The variation of actual responses 𝑦ᵢ, 𝑖 = 1, …, 𝑛, occurs partly due to the dependence on the predictors 𝐱ᵢ.
However, there is also an additional inherent variance of the output.
The COEFFICIENT of DETERMINATION, denoted as 𝑅², tells you which amount of variation in 𝑦 can be explained by the dependence on 𝐱 using the particular regression model.
Larger 𝑅² indicates a better fit and means that the model can better explain the variation of the output with different inputs.
The value 𝑅² = 1 corresponds to SSR = 0, that is to the perfect fit since the values of predicted and actual responses fit completely to each other.

Multiple or MULTIVARIATE linear regression is a case of linear regression with two or more independent variables.
If there are just two independent variables, the estimated regression function is 𝑓(𝑥₁, 𝑥₂) = 𝑏₀ + 𝑏₁𝑥₁ + 𝑏₂𝑥₂.
It represents a regression plane in a three-dimensional space.
The goal of regression is to determine the values of the weights 𝑏₀, 𝑏₁, and 𝑏₂ such that this plane is as close as possible to the actual responses and yield the minimal SSR.
SAMPLE euqation: 𝑓(𝑥₁, …, 𝑥ᵣ) = 𝑏₀ + 𝑏₁𝑥₁ + ⋯ +𝑏ᵣ𝑥ᵣ

GOLDEN RULE for count of coefficient in any type of regression expression:
If r=1 denotes simple linear regression, then no of coeff, is r+1.
Similarly for binary its 2+1, and for higher order its r+1.

POLYNOMIAL REGRESSION
Polynomial regression are generalized case of linear regression.
Here in addition to LINEAR terms like 𝑏₁𝑥₁, the regression function 𝑓 can include NON-LINEAR terms such as 𝑏₂𝑥₁², 𝑏₃𝑥₁³, or even 𝑏₄𝑥₁𝑥₂, 𝑏₅𝑥₁²𝑥₂,
SAMPLE:Regression function is a polynomial of degree 2: 𝑓(𝑥) = 𝑏₀ + 𝑏₁𝑥 + 𝑏₂𝑥².

WHEN TO USE linear vs polynomial regression?
The general guideline is to use linear regression FIRST to determine whether it can fit the particular type of curve in your data.
If you can't obtain an adequate fit using linear regression, that's when you might need to choose nonlinear regression.

UNDERFITTING & OVERFITTING
The choice of the optimal degree of the polynomial regression function.There is no straightforward rule for doing this.
It depends on the case. You should, however, be aware of two problems that might follow the choice of the degree: underfitting and overfitting.
UNDERFITTING: (low 𝑅²)
occurs when a model can’t accurately capture the dependencies among data, usually as a consequence of its own simplicity.
It often yields a low 𝑅² with known data and bad generalization capabilities when applied with new data.
OVERFITTING:(high 𝑅²)
happens when a model learns both dependencies among data and random fluctuations.
In other words, a model learns the existing data too well.
When applied to known data, such models usually yield high 𝑅².
However, they often don’t generalize well and have significantly lower 𝑅² when used with new data.

PYTHON PACKEGS FOR LINEAR REGRESSION
1. NumPy is a fundamental Python scientific package that allows many high-performance operations on single- and multi-dimensional arrays.
2. Package scikit-learn is a widely used Python library for machine learning, built on top of NumPy.
It provides the means for preprocessing data, reducing dimensionality, implementing regression, classification, clustering, etc
3. Consider statsmodels if ypu wamt to go beyond the scope of scikit-learn. It’s a powerful Python package for the estimation of statistical models, performing tests etc.

FIVE BASIC STEPS for implementing linear regression:
    step1: Import the packages and classes you need.
    step2: Provide data to work with and eventually do appropriate transformations.
    step3: Create a regression model and fit it with existing data.
    step4: Check the results of model fitting to know whether the model is satisfactory.
    step5: Apply the model for predictions. """


#----------------------Simple Linear Regression With scikit-learn ----------------------#
########## step1: Importing required class, packages
import numpy as np
from sklearn.linear_model import LinearRegression
# class sklearn.linear_model.LinearRegression will be used to perform linear and polynomial regression and make predictions accordingly.
##########  step2: Data to work on
x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1)) # <class 'numpy.ndarray'> FEATURE
# we called .reshape() on x because this array is required to be two-dimensional, or to be more precise, to have one column and as many rows as necessary.
# That’s exactly what the argument (-1, 1) of .reshape() specifies.
print(x)
'''[[ 5]
    [15]
    [25]
    [35]
    [45]
    [55]]'''
y = np.array([5, 20, 14, 32, 22, 38]) # <class 'numpy.ndarray'> TARGET
print(y) # [ 5 20 14 32 22 38]
print(y.shape) # (6,)
##########  step3: create model
model = LinearRegression()
# this create an instance of the class LinearRegression, which will represent the regression model.
# Has optional parameters to LinearRegression like fit_intercept, normalize etc.
# Step4: start using the model.
# First, you need to call .fit() on model:
model.fit(x, y)
# With .fit(), we calculate the optimal values of the weights 𝑏₀ and 𝑏₁, using the existing input and output (x and y) as the arguments.
# It returns self, which is the variable model itself. That’s why you can replace the last two statements with this one:
model = LinearRegression().fit(x, y)
print(type(model)) # <class 'sklearn.linear_model._base.LinearRegression'>
########## step4: get results
# obtain the coefficient of determination (𝑅²) with .score() called on model
r_sq = model.score(x, y)
print('coefficient of determination:', r_sq) #  0.7158756137479542
# the model's intercept and coeff as below
print('intercept:', model.intercept_) #  .intercept_ is a scalar;  5.633333333333329
print('slope:', model.coef_) # .coef_ is an array as it handles b0, b1, b2 like # [0.54]
#-------------------------
##ADDITIONALLY we may provide y as a two-dimensional array instead of single as below
new_model = LinearRegression().fit(x, y.reshape((-1, 1)))
print('intercept:', new_model.intercept_) # .intercept_ is a one-dimensional array
print('slope:', new_model.coef_) # .coef_ is a two-dimensional array with the single element 𝑏₁.
#-------------------------
######## Step 5: Predict response
#Now use the model for predictions with either existing or new data using .predict() method.
y_predict=model.predict(x) #----------------A
print('predicted response:', y_predict, sep='\n') # [ 8.33333333 13.73333333 19.13333333 24.53333333 29.93333333 35.33333333]
# When cmapred to linear equation :
y_pred_equation = model.intercept_ + model.coef_ * x#-----------------B
print('predicted response:', y_pred_equation, sep='\n')
print(type(y_pred_equation)) # <class 'numpy.ndarray'>
# Note: The response array of A& B would be the same.

######## Step 6: Plotting
import matplotlib.pyplot as plt
fig = plt.figure(figsize = (10, 5))
plt.plot(x, y_predict,label='Predicted')
plt.plot(x, y, 'r-.',label='Actaul') # r-. is for dotted, red color, 'b' for blue & 'g:' for green
plt.title('Actual vs Predicted')
plt.xlabel('x-axis for estimators')
plt.ylabel('y-axis for responses')
plt.show()

#USE CASE:
# In practice, regression models are often applied for forecasts.
# This means that you can use fitted models to calculate the outputs based on some other, new inputs

#----------------------Multiple Linear Regression With scikit-learn ----------------------#











#*************************************************************************************************************************
## Linear regression on Edureka## 
    # Step 1: Find best fit regression line using Least Square Method
    # Step 2: Check goodness of fit using R squared method
    # Step 3: Implement model using python
        # 3.1. Linear regression using Python from scratch
        # 3.2. Linear regression using Python (scikit lib)
## Diff. b/w L&R based on accuracy & fitness of model 
# linear(continuous):measured by loss, R squared, adjusted r squared etc
# regression(categorical):measured by accuracy, precision, recall, F1 score, ROC curve, confusion matrix etc
# supervised learning algorithm for solving regression based tasks.

# For linear regression, it is assumed that the two variables are LINEARLY RELATED.
# Hence, we try to find a linear function that predicts the response value(y) as accurately as possible as a function of the feature or independent variable(x).

## linear Equation: y=mx+c where c is y intercept of line
# m is positive and hence positive relationship b/w x & y
# m is negative and hence negative relationship b/w x & y
# Error= predicted value - actual error: goal is to REDUCE Error in model behaviour 
        
## Using Least Squared method
# m (slope)= Summation(x-x')(y-y')/(x-x')**2
# lets (x,y):[(1,3),(2,4),(3,2),(4,4),(5,5)]  
# mean (x',y'):[(3,3.6)]        
# calculate x-x',y-y',x-x**2 &  push it into formula        
# gives m= 0.4 & c= 2.4 , after calculation from y=mx+c
# now out x values from bundle (x,y) to get y as (2.8,3.2,3.6,4.0,4,4)
# Calculate distance b/w actual y and predicted y value & try reducing this GAP/ error for independent variales
# line with least error is regression line/ Best Fit Line.

## Check model fitness: with R squared method: MUST be higher value closing to 1 for best FIT
# Its a statistical measure of how close the data are to the fitted line.
# Known as coeff.of determination/ multiple determination
# generally hgher the R sq. value, most its fitted, BUT may vary also.
# calculate y(actaul)-mean VS y(predicted)-mean
# Hence : R**2 = summation(yp-ymean)**2/summation(ya-ymean)**2         
# Rsq=1.6/5.2 ~ 0.3 & hence not a Good Fit.suggesting data points are FAR away from regression.
# Increasing rsq to 0.7,0.9, 1 respectively will make the FIT closer        
        
# Imp: Are low Rsq always BAD?
# Not always as they tell that something's is very HARD to predict, like human behaviour        

# ***** Relationship b/w mean, median, mode & range **** #
# mean: regular meaning of "average"
# median: middle value
# mode: most often/ frequency of most recurring number
# say 13, 18, 13, 14, 13, 16, 14, 21, 13
# mean= average ((13 + 18 + 13 + 14 + 13 + 16 + 14 + 21 + 13) ÷ 9 = 15)        
# median= is the "middle" value in the list of ascending number ( (9 + 1) ÷ 2 = 10 ÷ 2 = 5th number)
# mode: number that is repeated more often than any other, so 13 is the mode. Can be multiple based on equal repetition.
# range: The largest value in the list is 21, and the smallest is 13, so the range is 21 – 13 = 8
#_______________________________________________________________________________
#### Linear regression using sample excel file having Weight, Height data 
## METHOD ONE: USING REGULAR FORMULA--------------------------------------------        
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Step1: data loading & idenfying the variables
data= pd.read_excel('C:/Users/akmsrimal/.spyder/datasets/brain.xls')
print(data.shape)
data.head(n=3)
# collecting x,y from data set whose Linear relationhip is to be established.
X= data['Height'].values
Y= data['Weight'].values
# mean of x,y
mean_x= np.mean(X)
mean_y=np.mean(Y)
# total number of values
l=len(X)
# Step2: calculate coef m & c for y=mx+c line
numer=0
denom=0
for i in range (l):
    numer += (X[i]-mean_x) * (Y[i]-mean_y)
    denom += (X[i]-mean_x)**2
    m= numer/denom
    c= mean_y-(m*mean_x)
# print coefficients
print(m,c)  # o/p is (3.772198432991385, -108.69844017949745)
# Step3: Plotting
# plotting values & regression line
max_x=np.max(X)+10 # random 10 selected to check scatteredness of points
min_x=np.min(X)-10
#calculating line value with m,c
x= np.linspace(min_x, max_x,100) # using 100 linspace in plot
y= c+m*x
#plotting lines
plt.plot(x,y,color='red',label='regression line')
# plotting scatter points
plt.scatter(X,Y,c='blue',label='scatter plot')
plt.xlabel('Height')
plt.ylabel('Weight')
plt.legend()
plt.show()
# Step4: Checking Fitness of model using R squared
ss_t=0 # total sum of squares
ss_r=0 # total sum of residuals
for i in range(l):
    y_pred=c+m*X[i]
    ss_t+= (Y[i]-mean_y)**2
    ss_r+= (Y[i]-y_pred) **2
r2= 1-(ss_r/ss_t)
print(r2) # o/p is 0.3913001200584526

## METHOD TWO:  USING SCIKIT LIBRARY--------------------------------------------   
from sklearn import linear_model
# Can not use rank 1 matrix in scikit learn
X=X.reshape((l,1))
# creating model
model = linear_model.LinearRegression()
# fitting training data
model=model.fit(X,Y)
# Calculate r2 score
r2_score=model.score(X,Y)
print(r2_score) # o/p is 0.3913001200584527
# Y prediction
Y_pred= model.predict(X) # predict Y with X & has no role in calculating r2 score value
# compare Y_pred with Y value at X points and can measure deviation in predition
len(Y_pred)

#_______________________________________________________________________________
#### Linear regression using Boston dataset of sklearn
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_boston
boston=load_boston()
boston
# check for data, feature_names, target etc here which will be used below as independent & dependent variables
df_x=pd.DataFrame(boston.data,columns=boston.feature_names) 
df_x.describe()
df_y=pd.DataFrame(boston.target)
df_y.describe()
# creating model now
model= linear_model.LinearRegression()
# spliting data in training & testing set
x_train,x_test,y_train,y_test=train_test_split(df_x,df_y,test_size=0.2,random_state=4) # random state is basically SEEDING
# fitting training in model
model.fit(x_train,y_train)
model.coef_ # checking weights of each coeff. of data-set
# prediction part now
a=model.predict(x_test) 
a[0] # gives predicted value of 0th index set as array([12.06508881])
# now comparing predicetd value vs actual value
y_test[0] # actual value is 16.5  and hence error 12.06-16.5
# similarly check for 1,2,3,n randomly index for idea of performance of model.
## mean square error 
np.mean((a-y_test)**2) # a-y_test gives error in individual index values
# error is around 25. use other models and tech to reduce this value for better perfrmance.

#_______________________________________________________________________________
## Concepts of SPLINE from Analytical Vidhya
## link: https://www.analyticsvidhya.com/blog/2018/03/introduction-regression-splines-python-codes/
# It assumed a linear relationship between the dependent and independent variables, which was rarely the case in reality.
# As an improvement over this model, we have Polynomial Regression which generated better results (most of the time).
# But using Polynomial Regression on datasets with high variability chances to result in over-fitting.
# SOLUTION: is  Regression Splines which uses a combination of linear/polynomial functions to fit the data.
# wage.csv to predict relation b/w wage & age.

# import modules
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt 
# %matplotlib inline
# read data_set
data = pd.read_csv("D:/PythonReferences/Wage.csv")
data.head()
data.shape # 3000 rows by 13 columns
# pulling dependent & independent variables, with VALUES, else error will be there.
data_x = data['age'].values
data_y = data['wage'].values
#Dividing data into train and testing datasets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.33, random_state = 1)
x_train.shape # 2010 rows
x_test.shape # 990 rows
# Visualize the relationship b/w age and wage
import matplotlib.pyplot as plt
plt.scatter(x_train, y_train, facecolor='None', edgecolor='k', alpha=0.3)
plt.show()

# Introduction to Linear Regression
# linear model as it establishes a linear relationship between the dependent and independent variables.
# y=m1x1+m2x2+....+c
# m1,m2 etc . Coefficients are the weights assigned to the features
# If it involves only one independent variable. It is called Simple Linear Regression
from sklearn import linear_model
# Fitting linear regression model
x = x_train.reshape(-1,1)
model = linear_model.LinearRegression()
model.fit(x,y_train)
print(model.coef_) # o/p is [0.72190831]
print(model.intercept_) # o/p is 80.65287740759283

# Prediction on validation dataset
x_test = x_test.reshape(-1,1)
pred = model.predict(x_test)
# Visualisation
# We will use 70 plots between minimum and maximum values of valid_x for plotting
xp = np.linspace(x_test.min(),x_test.max(),70)
xp = xp.reshape(-1,1)
pred_plot = model.predict(xp)
plt.scatter(x_test, y_test, facecolor='None', edgecolor='k', alpha=0.3)
plt.plot(xp, pred_plot)
plt.show()
## calculate the RMSE on the predictions to check model FIT
from sklearn.metrics import mean_squared_error
from math import sqrt
rms = sqrt(mean_squared_error(y_test, pred))
print(rms) # o/p is 40.436
# Note:linear regression is not capturing all the signals available and is not the best method for solving this wage prediction.

## Improvement over Linear Regression: Polynomial Regression

#  regression technique, which uses a non linear function, is called Polynomial regression. its nature is CURVE.
# polynomial eqas y= c+m1x1+m2x2**2+....+mnxn**2
  # Note: As we increase the power value, the curve obtained contains high oscillations which will lead to shapes that are over-flexible.
  # Such curves lead to over-fitting.
  
# Generating weights for polynomial function with degree =2
weights = np.polyfit(x_train, y_train, 2)
print(weights) # array([ -0.05194765,   5.22868974, -10.03406116])
# Generating model with the given weights
model = np.poly1d(weights)
# Prediction on test set
pred = model(x_test)
pred
# We will plot the graph for 70 observations only
xp = np.linspace(x_test.min(),x_test.max(),70)
pred_plot = model(xp)
plt.scatter(x_test, y_test, facecolor='None', edgecolor='k', alpha=0.3)
plt.plot(xp, pred_plot)
plt.show()

## SIMILARLY try to plot with other degree say 4,12,16,25 etc
# As we increase the complexity of the formula, the number of features also increases which is sometimes difficult to handle.
# polynomial regression has a tendency to drastically over-fit, even on this simple one dimensional data set

## Walk-through of Regression Splines along with its Implementations
# In order to overcome the disadvantages of polynomial regression, we can use an improved regression technique which,
# instead of building one model for the entire dataset, divides the dataset into multiple bins and fits each bin with a separate model. 
# Such a technique is known as REGRESSION SPLINES.
# Regression splines is one of the most important non linear regression techniques.
# In polynomial regression, we generated new features by using various polynomial functions on the existing features which imposed a global structure on the dataset.
# The points where the division occurs are called Knots
# Functions which we can use for modelling each piece/bin are known as PIECEWISE functions. 
# There are various piecewise functions that we can use to fit these individual bins.

## PIECEWISE FUNCTION: Type STEP ****************
# One of the most common piecewise functions is a Step function.
# Step function is a function which remains constant within the interval.
# We can fit individual step functions to each of the divided portions in order to avoid imposing a global structure.

# Dividing the data into 4 bins
df_cut, bins = pd.cut(x_train, 4, retbins=True, right=True)
df_cut.value_counts() # gives 4 bin having frequency in each BIN
#df_cut =pd.DataFrame([df_cut])
#x_train =pd.DataFrame([x_train])
#y_train =pd.DataFrame([y_train]) # converting category to dataframe
#type(df_cut) # categorical data type
#type(x_train) # dataframe
#type(y_train) # dataframe
df_steps = pd.concat([x_train, df_cut, y_train], keys=['age','age_cuts','wage'], axis=1)
type(df_steps) # dataframe object
# Create dummy variables for the age groups
df_steps_dummies = pd.get_dummies(df_cut)
df_steps_dummies.head()
df_steps_dummies.columns = ['17.938-33.5','33.5-49','49-64.5','64.5-80'] 

# Fitting Generalised linear models
fit3 = sm.OLS(df_steps.wage, df_steps_dummies).fit() # using OLS in place of GLM
# Binning validation set into same 4 bins
bin_mapping = np.digitize(x_test, bins) 
X_test = pd.get_dummies(bin_mapping)
# Removing any outliers
X_test = pd.get_dummies(bin_mapping).drop([5], axis=1)
# Prediction
pred2 = fit3.predict(X_test)
# Calculating RMSE
from sklearn.metrics import mean_squared_error 
from math import sqrt 
rms = sqrt(mean_squared_error(y_test, pred2)) 
print(rms) # o/p is 39.9 
# We will plot the graph for 70 observations only
xp = np.linspace(x_test.min(),x_test.max()-1,70) 
bin_mapping = np.digitize(xp, bins) 
X_test_2 = pd.get_dummies(bin_mapping) 
pred2 = fit3.predict(X_test_2)
# Visualisation
fig, (ax1) = plt.subplots(1,1, figsize=(12,5))
fig.suptitle('Piecewise Constant', fontsize=14)
# Scatter plot with polynomial regression line
ax1.scatter(x_train, y_train, facecolor='None', edgecolor='k', alpha=0.3)
ax1.plot(xp, pred2, c='b')
ax1.set_xlabel('age')
ax1.set_ylabel('wage')
plt.show()
# Binned regression does not create continuous functions of the predictor,
# so in most cases we would expect no relationship between the input and output.
# For example, in the above graph, we can see that the first bin clearly misses the increasing trend of wage with age.

## PIECEWISE FUNCTION: Type BASIS ****************
# To capture non-linearity in regression models, we need to transform some, or all of the predictors.
#  family of transformations that can fit together to capture general shapes is called a basis function.
# IMP: Most preferred choice for basis function is PIECEWISE POLYNOMIAL 
# Instead of fitting a constant function over different bins across the range of X, piecewise polynomial regression involves fitting separate low-degree polynomials over different regions of X
# As we use lower degrees of polynomials, we don’t observe high oscillations of the curve around the data.
# For example, a piecewise quadratic polynomial works by fitting a quadratic regression equation. y=mx**2+nx+c
# Each of these polynomial functions can be fit using the least squares error metric.
#  In general, if we place K different knots throughout the range of X, we will end up fitting K+1 different cubic polynomials.
# We can use any low degree polynomial to fit these individual bins.
# NOTE: Stepwise functions used above are actually piecewise polynomials of degree 0.

## necessary conditions/constraints for piecewise polynomials.
# 1. we may encounter certain situations where the polynomials at either end of a knot are not continuous at the knot.
# 2. to avoid this, we should add an extra constraint/condition that the polynomials on either side of a knot should be continuous at the knot.
# 3. Now after adding that constraint, we get a continuous family of polynomials.
# 4. To further smoothen the polynomials at the knots, we add an extra constraint/condition: the first derivative of both the polynomials must be same.
# 5. One thing we should note: Each constraint that we impose on the piecewise cubic polynomials effectively frees up one degree of freedom.
# 6. Going ahead we will impose an extra constraint: that the double derivatives of both the polynomials at a knot must be same.
# 7. piecewise polynomial of degree m with m-1 continuous derivatives is called a SPLINE.

## Cubic spline is a piecewise polynomial with a set of extra constraints (continuity, continuity of the first derivative, and continuity of the second derivative).
# In general, a cubic spline with K knots uses cubic spline with a total of 4 + K degrees of freedom.
# PS: There is seldom any good reason to go beyond cubic-splines.
from patsy import dmatrix
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Generating cubic spline with 3 knots at 25, 40 and 60
transformed_x = dmatrix("bs(train, knots=(25,40,60), degree=3, include_intercept=False)", {"train": x_train},return_type='dataframe')
# Fitting Generalised linear model on transformed dataset
fit1 = sm.GLM(y_train, transformed_x).fit()
# Generating cubic spline with 4 knots
transformed_x2 = dmatrix("bs(train, knots=(25,40,50,65),degree =3, include_intercept=False)", {"train": x_train}, return_type='dataframe')
# Fitting Generalised linear model on transformed dataset
fit2 = sm.GLM(y_train, transformed_x2).fit()
# Predictions on both splines
pred1 = fit1.predict(dmatrix("bs(valid, knots=(25,40,60), include_intercept=False)", {"valid": x_test}, return_type='dataframe'))
pred2 = fit2.predict(dmatrix("bs(valid, knots=(25,40,50,65),degree =3, include_intercept=False)", {"valid": x_test}, return_type='dataframe'))

# Calculating RMSE values
rms1 = sqrt(mean_squared_error(y_test, pred1))
print(rms1)
rms2 = sqrt(mean_squared_error(y_test, pred2))
print(rms2)

# We will plot the graph for 70 observations only
xp = np.linspace(x_test.min(),x_test.max(),70)
# Make some predictions
pred1 = fit1.predict(dmatrix("bs(xp, knots=(25,40,60), include_intercept=False)", {"xp": xp}, return_type='dataframe'))
pred2 = fit2.predict(dmatrix("bs(xp, knots=(25,40,50,65),degree =3, include_intercept=False)", {"xp": xp}, return_type='dataframe'))

# Plot the splines and error bands
plt.scatter(data.age, data.wage, facecolor='None', edgecolor='k', alpha=0.1)
plt.plot(xp, pred1, label='Specifying degree =3 with 3 knots')
plt.plot(xp, pred2, color='r', label='Specifying degree =3 with 4 knots')
plt.legend()
plt.xlim(15,85)
plt.ylim(0,350)
plt.xlabel('age')
plt.ylabel('wage')
plt.show()

## NATURAL SPLINE: for boundary outliers: 
### Behavior of polynomials that are fit to the data tends to be erratic near the boundaries. Such variability can be dangerous
## To smooth the polynomial beyond the boundary knots, we will use a special type of spline known as Natural Spline.
# A natural cubic spline adds additional constraints, namely that the function is linear beyond the boundary knots. 
# This constrains the cubic and quadratic parts there to 0, each reducing the degrees of freedom by 2.
# That’s 2 degrees of freedom at each of the two ends of the curve, reducing K+4 to K.

# Generating natural cubic spline
transformed_x3 = dmatrix("cr(train,df = 3)", {"train": x_train}, return_type='dataframe')
fit3 = sm.GLM(y_train, transformed_x3).fit()
# Prediction on validation set
pred3 = fit3.predict(dmatrix("cr(valid, df=3)", {"valid": x_test}, return_type='dataframe'))
# Calculating RMSE value
rms = sqrt(mean_squared_error(y_test, pred3))
print(rms)
# We will plot the graph for 70 observations only
xp = np.linspace(x_test.min(),x_test.max(),70)
pred3 = fit3.predict(dmatrix("cr(xp, df=3)", {"xp": xp}, return_type='dataframe'))

# Plot the spline
plt.scatter(data.age, data.wage, facecolor='None', edgecolor='k', alpha=0.1)
plt.plot(xp, pred3,color='g', label='Natural spline')
plt.legend()
plt.xlim(15,85)
plt.ylim(0,350)
plt.xlabel('age')
plt.ylabel('wage')
plt.show()

### Choosing the Number and Locations of the Knots
# One potential place would be the area of high variability, because in those regions the polynomial coefficients can change rapidly.
# one option is to place more knots in places where we feel the function might vary most rapidly, and to place fewer knots where it seems more stable.
# While this option can work well, in practice it is common to place knots in a uniform fashion.
# Another option is to try out different numbers of knots and see which produces the best looking curve.
# A more objective approach is to use cross-validation. With this method:
    #1.we remove a portion of the data,
    #2.fit a spline with a certain number of knots to the remaining data, and then,
    #3.use the spline to make predictions for the held-out portion.
    # We repeat this process multiple times & calculate cross validated RMSE. 
    # value of K giving lowest RMSE is chosen as factor here.
    
## Comparison of Regression Splines with Polynomial Regression
# Regression splines often give better results than polynomial regression.
# splines introduce flexibility by increasing the number of knots but keep the degree fixed. 
# WHILE polynomials uses a high degree polynomial to produce flexible fits
# trying to get xtra flexibility in the polynomial produces undesirable results at the boundaries,whereas the natural cubic spline still provides a reasonable fit to the data.





























       
