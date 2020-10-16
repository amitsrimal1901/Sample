# Data Preprocessing
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('https://raw.githubusercontent.com/amitsrimal1901/Machine_Learning_A-Z/master/Part%201%20-%20Data%20Preprocessing/Data.csv')
# dataset is data-frame type. dataset.shape (10,4)
X = dataset.iloc[:, [0,1,2]].values  ## <class 'numpy.ndarray'>
y = dataset.iloc[:, 3].values  # <class 'numpy.ndarray'>

# GeT to know your data
dataset.describe()
dataset.info()

# Check if missing values
dataset.isnull() # checks if null present
dataset.isnull().any() #  # returns count of count in each column
dataset.isnull().values # retruns array with null values from dataframe
dataset.isnull().values.any() #retruns TRUE if a single NAN is found

# Check number of NaNs
dataset.isnull().sum() # rteturns count of count in each column
dataset.isnull().sum().sum() # retruns total count of NULL i.e. 2 here

# Taking care of missing data
"""
Imputer is used to handle missing data in your dataset. 
Imputer gives you easy methods to replace NaNs and blanks with something like the mean of the column or even median. 
But before it can replace these values, it has to calculate the value that will be used to replace blanks.

The SimpleImputer class also supports categorical data represented as string values or pandas categoricals when using the 'most_frequent' or 'constant' strategy.
This means we can use to replace most frequenct ords with something we want
# Example: imp = SimpleImputer(strategy="most_frequent")
"""
# from sklearn.preprocessing import Imputer....deprectaed now
from sklearn.impute import SimpleImputer
simpleimputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
simpleimputer = simpleimputer.fit(X[:, 1:3])
X[:, 1:3] = simpleimputer.transform(X[:, 1:3]) # array object

# Encoding categorical data: Categorical encoding is a process of converting categories to numbers.
"""
LABEL ENCODING is a popular encoding technique for handling categorical variables. 
In this technique, each label is assigned a unique integer based on alphabetical ordering.
 -- Challenges with Label Encoding
In the scenario fopr country code data set, the Country names do not have an order or rank. 
But, when label encoding is performed, the country names are ranked based on the alphabets. 
Due to this, there is a very high probability that the model captures the relationship between countries such as India < Japan < the US.
--We apply Label Encoding when:
    1. The categorical feature is ordinal (like Jr. kg, Sr. kg, Primary school, high school)
    2. The number of categories is quite large as one-hot encoding can lead to high memory consumption

ONE HOT ENCODING One-Hot Encoding is another popular technique for treating categorical variables.
It simply creates additional features based on the number of unique values in the categorical feature. 
Every unique value in the category will be added as a feature.
-- Challenges of One-Hot Encoding: Dummy Variable Trap
One-Hot Encoding results in a Dummy Variable Trap as the outcome of one variable can easily be predicted with the help of the remaining variables.
he Dummy Variable Trap leads to the problem known as multicollinearity. 
Multicollinearity occurs where there is a dependency between the independent features. 
In order to overcome the problem of multicollinearity, one of the dummy variables has to be dropped
--We apply One-Hot Encoding when:
    1. The categorical feature is not ordinal (like the countries above)
    2. The number of categorical features is less so one-hot encoding can be effectively applied
"""
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder
#Step1: Encoding the categorcial data
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0]) # applied labelEncoder to NO-ORDINAL data country here
# COuntry gets 0/1/2/ type of categorial values
#Onehotencoder = OneHotEncoder(categorical_features = [0])
#X = onehotencoder.fit_transform(X).toarray()
#Step2: Creating dummy variables
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(), [0])],    # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
    remainder='passthrough')                      # Leave the rest of the columns untouched
X = np.array(ct.fit_transform(X), dtype=np.float) # gets to create addtiona columns based on categorical code 1/2/3 etc
#NOTE: This creates dummy variable and we need to remove it EXPLICITLY
# Country code 0,1,2, hence 3 columns are added to show country, instead of single country column

#Avoiding the Dummy Variable Trap by dropping any of the column say 0th column in our case
X = X[:, 1:] #<class 'numpy.ndarray'> having shape (4 olumns instead of 5 columns earlier )

# Encoding the Dependent Variable
# Similarly Labelling for TARGET variable, no encoding needed as its a single column only
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y) # o/1 for the case of TRUE/ FALSE

#*******************************************************************************