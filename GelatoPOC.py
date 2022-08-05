import numpy as np
low_memory=False
import pandas as pd
# Importing the dataset
dataset = pd.read_csv(r'/Users/amitshrimal/Downloads/allCountriesCSV.csv')
##type(dataset) ## dataframe
dataset[dataset['CITY']=='Friedrichsdorf'][['COUNTRY','POSTAL_CODE']] ## AT and DE country
dataset[dataset['CITY']=='Pak Chong'][['COUNTRY','POSTAL_CODE']] ## 30130
dataset[dataset['CITY']=='Kimstad'][['COUNTRY','POSTAL_CODE']] ## 610 20
dataset[dataset['CITY']=='Montreal'][['COUNTRY','POSTAL_CODE']] ## 65591, 54550 but correct is H3C 0T4
dataset[dataset['CITY']=='Barchem'][['COUNTRY','POSTAL_CODE']] ## 7244 but actual is 7244 BM
dataset[dataset['CITY']=='Gjøvik'][['COUNTRY','POSTAL_CODE']] # series of post code with the actual on ealso from Ops team 2818
dataset[dataset['CITY']=='London'][['COUNTRY','POSTAL_CODE']] # IP24 but Ops team used IPS25 which is Holme Hale
## based on multiple condition search
dataset.query('COUNTRY=="DE" & CITY=="Friedrichsdorf"')[['COUNTRY','POSTAL_CODE']]

## Model DTP
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# Importing the dataset
dtpRaw = pd.read_csv(r'/Users/amitshrimal/Downloads/DtpMLFedExCleaned.csv')
dtpExtract = dtpRaw[['TotalDutyTax', 'ServiceType', 'WeightPounds', 'RecipientState','RecipientCountry','ShipperState', 'ShipperCountry', 'CustomsValueUSD','MultiplierPercent']]
X= dtpExtract[['ShipperCountry','ShipperState', 'RecipientCountry','RecipientState','ServiceType', 'WeightPounds', 'CustomsValueUSD']]
# Adding dummy variable for encoding to 0/1
# Multiple Independent variable
X = pd.get_dummies(data=X, drop_first=True)
X.head() # showing first 5 rows
# Dependent varibale
Y = dtpExtract['TotalDutyTax']
Y.head() # showing first 5 rows

# Creating Training and test data set from X/ Y
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=101)
print(X_train.shape) # (151, 34)
print(X_test.shape) # (102, 34)
print(y_train.shape) # (151,)
print(y_test.shape) # (102,)

# Making Linear regression Multivariate model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)
# print the coefficient and intercept
print(model.coef_)
print(model.intercept_)
# Looking into all parameters coefficient
coeff_parameter = pd.DataFrame(model.coef_,X.columns,columns=['Coefficient'])
print(coeff_parameter) # printing coefficient of all dummy variable

# Testing model performance using test data set
predictions = model.predict(X_test)
print(predictions)
# Plot of Predicted vs Actual values
sns.regplot(y_test,predictions)

# Plotting R squared
import statsmodels.api as sm
X_train_Sm= sm.add_constant(X_train)
X_train_Sm= sm.add_constant(X_train)
ls=sm.OLS(y_train,X_train_Sm).fit()
print(ls.summary())

# Enter real world data now here:
# Predicting buy Yes/ No for a given Age, Salary
ShipperCountry=int(input("Enter Shipper Country IsoCode: "))
ShipperState=int(input("Enter Shipper State IsoCode: "))
RecipientCountry= int(input("Enter Recipient Country IsoCode: "))
RecipientState= int(input("Enter Recipient State IsoCode: "))
ServiceType= int(input("Enter Service Type: "))
WeightPounds= int(input("Enter Weight in pounds: "))
CustomsValueUSD= int(input("Enter Custom values in USD: "))
# creating array to be fetched model
data_by_user= np.array([ShipperCountry, ShipperState, RecipientCountry,RecipientState, ServiceType, WeightPounds, CustomsValueUSD ]).reshape(1,-1)
print(type(data_by_user))
expectedDutyTax= model.predict(data_by_user)
print("Expected DutiesTaxes amount (USD) is:", expectedDutyTax)
##-----------------------------------------------------