6# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 13:29:52 2018

@author: BA3
"""

age= int(input("Enter your age: ")) #1. Step 1 i/p numeric 
def cube_age(age): #2. Step2. function to perform certain operation
    return age**3
y = cube_age(age) #3. Step 3.assigning function o/p to some other variable
print ('Cube of your age',age,'is', y/4) # step 4. getting overall o/p as her

------------------------------------------------------------------------------------------------------------
##TIME SERIES with airpassengers dataset
import pandas as pd
from fbprophet import Prophet
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
apd = pd.read_csv("C:/Users/BA3/Desktop/DETfiles/ML/Data/AirPassengers.csv")
apd.head(9)
# data needs to be time & metric value
apd.dtypes # o/p is obj & int64
# Because the Month column is not of the datetime type, we'll need to convert it:
apd['Month'] = pd.DatetimeIndex(apd['Month'])
apd.dtypes # o/p is datime64 & int64. Good for time series now.
# Prophet also imposes the strict condition that the input columns be named ds (the time column) and y (the metric column), 
#so let's rename the columns in our DataFrame:
apd = apd.rename(columns={'Month': 'ds','#Passengers': 'y'})
apd.head(5)

# visualize the data
ax = apd.set_index('ds').plot(figsize=(20, 10))
ax.set_ylabel('Monthly Number of Airline Passengers')
ax.set_xlabel('Date')
plt.show()

## Time Series Forecasting with Prophet
# specify the desired range of our uncertainty interval by setting the interval_width parameter.
# set the uncertainty interval to 95% (the Prophet default is 80%)
apd_model = Prophet(interval_width=0.95)
# Model initialized & now data fitting into model
apd_model.fit(apd)

# In order to obtain forecasts of our time series, we must provide Prophet with a new DataFrame containing a ds column.
# lets use Prophet's data making utility make_future_dataframe for this
future_dates = apd_model.make_future_dataframe(periods=12, freq='MS') # 36 months created from apd date
future_dates.head()
future_dates.tail()
# NOTE: future dates has all dates starting from apd, hence 144th row date is added.
# The DataFrame of future dates is then used as input to the predict method of our fitted model
forecast = apd_model.predict(future_dates)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail() # ds,yhat,_lowe,_upper are traditional naming 

## IMP: A variation in values from the output presented above is to be expected as Prophet relies on Markov chain Monte Carlo (MCMC) methods to generate its forecasts. 
# MCMC is a stochastic process, so values will be slightly different each time

## plotting the result
apd_model.plot(forecast, uncertainty=True)
# Prophet plots the observed values of our time series (the black dots), the forecasted values (blue line) and the uncertainty intervals of our forecasts (the blue shaded regions).

# Visualing components if time series
apd_model.plot_components(forecast)

# Conclusion:
# The first plot shows that the monthly volume of airline passengers has been linearly increasing over time. 
# The second plot highlights the fact that the weekly count of passengers peaks towards the end of the week and on Saturday.
# The third plot shows that the most traffic occurs during the holiday months of July and August.


------------------------------------------------------------------------------------------------------------
##Timeseries using airpassengers but with divided data 80 & 20 sort of.(115+29 rows)
import pandas as pd
from fbprophet import Prophet
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
apd = pd.read_csv("C:/Users/BA3/Desktop/DETfiles/ML/Data/AirPassengers.csv")
apd.head(9)
# data needs to be time & metric value
apd.dtypes # o/p is obj & int64
# Because the Month column is not of the datetime type, we'll need to convert it:
apd['Month'] = pd.DatetimeIndex(apd['Month'])
apd.dtypes # o/p is datime64 & int64. Good for time series now.
# Prophet also imposes the strict condition that the input columns be named ds (the time column) and y (the metric column), 
#so let's rename the columns in our DataFrame:
apd = apd.rename(columns={'Month': 'ds','#Passengers': 'y'})
apd.head(5)

# visualize the data
ax = apd.set_index('ds').plot(figsize=(20, 10))
ax.set_ylabel('Monthly Number of Airline Passengers')
ax.set_xlabel('Date')
plt.show()

## creating test & training data set 
apd_train= apd.head(115)
apd_train.describe()
apd_test= apd.tail(29)
apd_test.describe()

# creating model now
apd_model_95 = Prophet(interval_width=0.95)
# lets train model on training dataset
apd_model_95.fit(apd_train)
#AND Now extract date from test table
#future_date_95= apd_test.iloc[:,0] # gives series date & prophet expected ds format
# so will use of prophet's in built date making function for next 60 months
future_date_95=apd_model_95.make_future_dataframe(periods=60, freq='MS')
# Forecasting using the available test date that's exracted above
forecast_95 = apd_model_95.predict(future_date_95)
future_date_95.dtypes ## datatype is datetime64

## Check FOUR critical attributes. Though forecasthas 16 columns from Prophet model.
forecast_95[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10)
apd_test.tail(10) ##printing last 10 values for comparsion purpose
## Compare element by element ro check the DISPCREPANCY in value.

## plotting the result
apd_model_95.plot(forecast_95, uncertainty=True)
# plot shows preidcted grpah starting from End of training date.No dots &only predicted line is displayed.
# Trsining data has end line as 1958-07-01 DATE
# Visualing components if time series
apd_model_95.plot_components(forecast_95)


## FUNCTION to check for particular predicted date & see difference. 
##NOTE: Select date from 1949 to 1960 oNLY
# First convert to datetime64 for prophet, else error will be thrown
enter_for_date= str(input("Enter prediction date b/w 1949 to 1960: ")) 
for_date = {'ds': [enter_for_date]} #also can use '1960-10-01 or 1960-10' for same result coz of time format syntex
for_date_shaped=pd.DataFrame(data=for_date) # datetime given shape of dataframe for prophet prediction.
#passing shaped date to model for prediction.
forecast_95_for_date = apd_model_95.predict(for_date_shaped)
forecast_95_for_date_output=forecast_95_for_date[['yhat']] # extracting ds & yhat values form predicted metrics
print("forecasted value is: ",forecast_95_for_date_output.loc[0,'yhat']) # reading specific cell value

# Reading individual row data from original set
# create index first on ds
apd_indexed= apd.set_index("ds",drop=False)
#reading desired cell data say date & y value
actual_for_date_output= apd_indexed.loc[enter_for_date,"y"]
print("actual value is: ",actual_for_date_output) 

#PRINTING Difference between prediction & actual value is:
print("Actual- Predicted is:",actual_for_date_output - forecast_95_for_date_output.loc[0,'yhat'])
------------------------------------------------------------------------------------------------------------

from textblob import TextBlob
polarity:
Subjectivity:

review1='The movie was okay'
review2='The movie was good'
review3='The movie was very okay'
review4='The movie was awesome'
review5='The movie was pathetic'

blob1= TextBlob(review1)
blob2= TextBlob(review2)
blob3= TextBlob(review3)
blob4= TextBlob(review4)
blob5= TextBlob(review5)

print('Review 1's 'blob1.sentiments)
print('Review 2's 'blob2.sentiments)
print('Review 3's 'blob3.sentiments)
print('Review 4's 'blob4.sentiments)
print('Review 5's 'blob5.sentiments)


%##
Features of TextBlob:
Noun phrase extraction
Part-of-speech tagging
Sentiment analysis
Classification (Naive Bayes, Decision Tree)
Language translation and detection powered by Google Translate
Tokenization (splitting text into words and sentences)
Word and phrase frequencies
Parsing
n-grams
Word inflection (pluralization and singularization) and lemmatization
Spelling correction
Add new models or languages through extensions
WordNet integration
%##















