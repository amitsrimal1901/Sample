# TimeSeries Forecasting -------------------------------------------------------
## Using Tableau Sample Superstore data
import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
import pandas as pd
import statsmodels.api as sm
import matplotlib

## start from time series analysis and forecasting for furniture sales.
df = pd.read_excel("C:/Users/BA3/Desktop/DETfiles/ML/Data/SuperstoreData.xls")
df.head(3)
furniture = df.loc[df['Category'] == 'Furniture'] ## getting data for category Furniture only
furniture.head()
## get extreme date range for furniture sale data
furniture['Order Date'].min() ##2014-01 yyyy-mm
furniture['Order Date'].max() ##2017-12 yyyy-mm

##Data Preprocessing
list(furniture) ## returns list of column name
list(furniture.columns.values) ## ALITER way of doing above
#Now below steps includes removing columns we do not need, check missing values, aggregate sales by date and so on.
# lets say cols define column that we need to omit.
cols = ['Row ID', 'Order ID', 'Ship Date', 'Ship Mode', 'Customer ID', 'Customer Name', 'Segment', 'Country', 'City', 'State', 'Postal Code', 'Region', 'Product ID', 'Category', 'Sub-Category', 'Product Name', 'Quantity', 'Discount', 'Profit']
furniture.drop(cols, axis=1, inplace=True)
list(furniture)  ## only date & sale is left
##lets sort data as per Sales date
furniture = furniture.sort_values('Order Date')
# Check count of null data in furniture data set
furniture.isnull().sum()
# creating custom index now for dataframe.
furniture = furniture.groupby('Order Date')['Sales'].sum().reset_index()

## Indexing with Time Series Data
furniture = furniture.set_index('Order Date') ## set order date as Index of data frame
furniture.index
## Lets use the AVERAGE DAILY SALES value for that month instead of daily sales
y = furniture['Sales'].resample('MS').mean() ## MS refers to monthly frequency
# Lets have a quick peek 2017 furniture sales data
y['2016':]


## Visualizing Furniture Sales Time Series Data
y.plot(figsize=(15, 6))
plt.show()

# visualize our data using a method called time-series decomposition that allows us to decompose our time series into three distinct components: trend, seasonality, and noise.
from pylab import rcParams
rcParams['figure.figsize'] = 18, 8

decomposition = sm.tsa.seasonal_decompose(y, model='additive')
fig = decomposition.plot()
plt.show()
# plot above clearly shows that the sales of furniture is unstable, along with its obvious seasonality.


## Time series forecasting with ARIMA "Autoregressive Integrated Moving Average."
#ARIMA models are denoted with the notation ARIMA(p, d, q). These three parameters account for seasonality, trend, and noise in data.
p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

# This step is parameter Selection for our furniture’s sales ARIMA Time Series Model. 
# Our goal here is to use a “grid search” to find the optimal set of parameters that yields the best performance for our model.
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

            results = mod.fit()

            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue

# NOTE: The above output suggests that SARIMAX(1, 1, 1)x(1, 1, 0, 12) yields the lowest AIC value of 297.78. 
# Therefore we should consider this to be optimal option.

## Fitting the ARIMA model:
mod = sm.tsa.statespace.SARIMAX(y,
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 0, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit()
print(results.summary().tables[1])            
# We should always run model diagnostics to investigate any unusual behavior.
results.plot_diagnostics(figsize=(16, 8))
plt.show()
# It is not perfect, however, our model diagnostics suggests that the model residuals are near normally distributed.

## Validating Forecast results:
# compare predicted sales to real sales of the time series, and we set forecasts to start at 2017–01–01 to the end of the data.
pred = results.get_prediction(start=pd.to_datetime('2017-01-01'), dynamic=False)
pred_ci = pred.conf_int()
ax = y['2014':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('Furniture Sales')
plt.legend()
plt.show()

# Line plot is showing the observed values compared to the rolling forecast predictions
y_forecasted = pred.predicted_mean
y_truth = y['2017-01-01':]
mse = ((y_forecasted - y_truth) ** 2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))
# The Mean Squared Error of our forecasts is 22993.58

print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))
# The Root Mean Squared Error of our forecasts is 151.64
# In statistics, the mean squared error (MSE) of an estimator measures the average of the squares of the errors — that is, the average squared difference between the estimated values and what is estimated.
# he MSE is a measure of the quality of an estimator — it is always non-negative, and the smaller the MSE, the closer we are to finding the line of best fit.

## Producing and visualizing forecasts:
pred_uc = results.get_forecast(steps=100)
pred_ci = pred_uc.conf_int()
ax = y.plot(label='observed', figsize=(14, 7))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('Furniture Sales')
plt.legend()
plt.show()

## Our model clearly captured furniture sales seasonality. 
# As we forecast further out into the future, it is natural for us to become less confident in our values. 
# This is reflected by the confidence intervals generated by our model, which grow larger as we move further out into the future

#-- to be continued




#########-------------------------------------------------------------------------------####################
#TimeSeries on EDUREKA youtube @ https://www.youtube.com/watch?v=e8Yw4alG16Q
# Other link as: https://www.kaggle.com/rakannimer/air-passenger-prediction/notebook
# Time series: data values spread with EQUAL intervals
# Components of Time series:
    #1: Seasonality
    #2: Trend
    #3: Cyclic
    #4: Irregularity (also called NOISE, or Residual in some cases)
    
# When not to use TIme Series
    #1: Values are constant
    #2: values are in form of functions (known or derived)
    
# Whats STAIONARITY: 
# A stationary time series is one whose statistical properties such as mean, variance, autocorrelation, etc. are all constant over time.
# Time series needs data to be Stationary   
# Criteria's of stationarity: 
    #`1: Constant mean
    #2: Constant variance
    #3: Autocovariance that does not depends on time   
    
# Tests to check Stationarity:
    #1: Rolling Statistics: plot moving avg or moving variance & see if it varies with time. Visual tech.
    #2: ADCF test: Null hypothesis saying TS is non-stationary. Test result consist of Test Statistics & some Critical values.

## ARIMA model: AutoRegressiove Integated Moving Average model
# AR(p=autoregressive lags)+I(d=order of differentiation)+MA(q= moving average)    
    
## FORECAST model
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 10, 6

dataset = pd.read_csv("C:/Users/BA3/Desktop/DETfiles/ML/Data/AirPassengers.csv")
# Parse strings to datetime type
dataset['Month'] = pd.to_datetime(dataset['Month'], infer_datetime_format=True)
indexedDataset = dataset.set_index(['Month']) # setting index on datetime- month
indexedDataset.head()

from datetime import datetime
indexedDataset['1949-03']
##indexedDataset['1949-03-01'] # returns key error as index defined on Month
indexedDataset['1949-03':'1949-06']
indexedDataset['1949']    

# Plotting high level relation
plt.xlabel("Date")
plt.ylabel("Number of air passengers")
plt.plot(indexedDataset)

##Check for Stationarity of time Series
    #Method 1: Visual Mean at lets say year 1961 is different from mean at 1949, hence data is NOT statinary
    #Method 2: Rolling Statistics 
rolmean = indexedDataset.rolling(window=12).mean() # starting 11 valueas NaN as 12 is set for Rolling calculation.
rolstd = indexedDataset.rolling(window=12).std()
print(rolmean, rolstd)   
#Plot rolling statistics: Run all 3 commands to plot on same canvas
orig = plt.plot(indexedDataset, color='blue',label='Original')
mean = plt.plot(rolmean, color='red', label='Rolling Mean')
std = plt.plot(rolstd, color='black', label = 'Rolling Std')
plt.legend(loc='best')
plt.title('Rolling Mean & Standard Deviation')
plt.show(block=False)
##Note: from graph, mean & std deviation are not constant & hence data is non stationary in nature.
    #Method3: Dickey Fuller test
from statsmodels.tsa.stattools import adfuller

print ('Results of Dickey-Fuller Test:')
dftest = adfuller(indexedDataset['#Passengers'], autolag='AIC') # Akaike Information Criterion
# AIC technique is based on in-sample fit to estimate the likelihood of a model to predict/estimate the future values. A good model is the one that has minimum AIC among all the other models.
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key,value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value
print(dfoutput)
#NOTE:  check p value, test stats value & confer that TS is not statinary.

# Estimating trend
indexedDataset_logScale = np.log(indexedDataset)
plt.plot(indexedDataset_logScale)

# plotting for moving average
movingAverage = indexedDataset_logScale.rolling(window=12).mean()
movingSTD = indexedDataset_logScale.rolling(window=12).std()
plt.plot(indexedDataset_logScale)
plt.plot(movingAverage, color='red')

## Transformation for making TS stationary
#This can be done using log, exp, sq etc. Any random method which futs BEST.
#Lets say we use LOG method: Get the difference between the moving average and the actual number of passengers
datasetLogScaleMinusMovingAverage = indexedDataset_logScale - movingAverage
datasetLogScaleMinusMovingAverage.head(12)
#Remove Nan Values
datasetLogScaleMinusMovingAverage.dropna(inplace=True)
datasetLogScaleMinusMovingAverage.head(10)

#Now check whther data has been made Satinaory or NOT. So perform all three test
from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    
    #Determing rolling statistics
    movingAverage = timeseries.rolling(window=12).mean()
    movingSTD = timeseries.rolling(window=12).std()

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(movingAverage, color='red', label='Rolling Mean')
    std = plt.plot(movingSTD, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries['#Passengers'], autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)

##Run the function & check the concurrent graph
test_stationarity(datasetLogScaleMinusMovingAverage)
#1. p value is almost ZERO with low value for test statistics
#2. Critical value & Test statistics value are almost EQUAL.
## form 1 & 2, we get that Mean & std are getting COnstant &hence TS has been made stationary.

#Now ltes chekc TREND and hence weighted avrage will be used.
exponentialDecayWeightedAverage = indexedDataset_logScale.ewm(halflife=12, min_periods=0, adjust=True).mean()
plt.plot(indexedDataset_logScale)
plt.plot(exponentialDecayWeightedAverage, color='red')
#hence trend is UPWARD & increasing with time.

## Now Lets subrtract weighted mean, Earlier we used simple mean. & check how much statinarty is achieved.
datasetLogScaleMinusMovingExponentialDecayAverage = indexedDataset_logScale - exponentialDecayWeightedAverage
test_stationarity(datasetLogScaleMinusMovingExponentialDecayAverage) #passing TS is test_stationarity function SIMPLY.
#Note: results are much flat & hence trnaformation rendered got Statinarity.

###Now ltes focus to SHIFT the data whch will be used for forecasting.
# Shift is used here
datasetLogDiffShifting = indexedDataset_logScale - indexedDataset_logScale.shift()
plt.plot(datasetLogDiffShifting)
#Now check again for stationairty after data SHIFT
datasetLogDiffShifting.dropna(inplace=True)
test_stationarity(datasetLogDiffShifting) # using previus function & passing TS value.
##Note: no trend, mean & dev are constant & hence statinarity is acheieved

## Components of TS
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(indexedDataset_logScale)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.subplot(411)
plt.plot(indexedDataset_logScale, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()

## Now check whther Noise/ residuals are Stationary of not.
decomposedLogData = residual
decomposedLogData.dropna(inplace=True)
test_stationarity(decomposedLogData)
# Here graph is NOT STATINARY in nature. & we have d value for ARIMA.

# Lets calcultae p & q for ARIMA using PACF(partial autocoorelation function) & ACF graphs respectively
# Preferred method: Ordinary Least Square ols
#ACF and PACF plots:
from statsmodels.tsa.stattools import acf, pacf

lag_acf = acf(datasetLogDiffShifting, nlags=20)
lag_pacf = pacf(datasetLogDiffShifting, nlags=20, method='ols')


#Plot ACF: 
plt.subplot(121) 
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(datasetLogDiffShifting)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(datasetLogDiffShifting)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')

#Plot PACF:
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(datasetLogDiffShifting)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(datasetLogDiffShifting)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()

# to get value of p,q, check graph point where Y gets ZERO for the first time.
#hence corresponding x axis value becomes the p(PACF),q values for ARIMA.

## Now lets plot ARIMA with p,d,q values ie.(p=2,d=1,q=2)
from statsmodels.tsa.arima_model import ARIMA
##Note: try using random combination of pdq values
## With graph check RSS: Greater the RSS, its bad for prediction. Make it as small as possible.
#AR MODEL(gives p value) having (pdq as 210 as its just first half of ARIMA)
##AR model basically forecast a series without trend or seasonality. Its similar to linear regression where predictors are lagged version of the series.
# major assumption in AR is STATIONARITY.
model = ARIMA(indexedDataset_logScale, order=(2, 1, 0))  
results_AR = model.fit(disp=-1)  
plt.plot(datasetLogDiffShifting)
plt.plot(results_AR.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_AR.fittedvalues-datasetLogDiffShifting["#Passengers"])**2))
print('Plotting AR model') ## RSS value is ~1.5

#MA MODEL having (pdq as 012 as its just second half of ARIMA)
model = ARIMA(indexedDataset_logScale, order=(0, 1, 2))  
results_MA = model.fit(disp=-1)  
plt.plot(datasetLogDiffShifting)
plt.plot(results_MA.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_MA.fittedvalues-datasetLogDiffShifting["#Passengers"])**2))
print('Plotting AR model') ## RSS value is ~1.4

## Combining the AR+MA model as INTEGRATED now:
model = ARIMA(indexedDataset_logScale, order=(2, 1, 2))  
results_ARIMA = model.fit(disp=-1)  
plt.plot(datasetLogDiffShifting)
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-datasetLogDiffShifting["#Passengers"])**2))
## RSS value is 1.02, which is far smaller that AR, MA 

## now lets try fitting the values(PREDICTING the values it means), in series format                                                                          
predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
print (predictions_ARIMA_diff.head())

#Convert to cumulative sum
predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
print (predictions_ARIMA_diff_cumsum.head())

##Predicting using fitted values:
#predictions_ARIMA_log = pd.Series(ts_log.ix[0], index=ts_log.index)
#predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
#predictions_ARIMA_log.head()

predictions_ARIMA_log = pd.Series(indexedDataset_logScale['#Passengers'].ix[0], index=indexedDataset_logScale['#Passengers'].index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
predictions_ARIMA_log.head()

##Chnaging values back to original as we did it for log normalization.
predictions_ARIMA = np.exp(predictions_ARIMA_log)
plt.plot(indexedDataset)
plt.plot(predictions_ARIMA)
plt.title('RMSE: %.4f'% np.sqrt(sum((predictions_ARIMA-indexedDataset["#Passengers"])**2)/len(indexedDataset["#Passengers"])))

## Now PREDICTING outocmes
indexedDataset_logScale #returns dataset of 144 rows

results_ARIMA.plot_predict(1,264) ##(264= 144datarows+10*12 to predict) ie 10 yrs prediction expected.
#gets predicted valuel instead of graph
x=results_ARIMA.forecast(steps=5) # ie 10 years (10*12 months data each we have)
print(x)


#------------------------------------------------------------------------------------------------------------

##TIME SERIES with airpassengers dataset with facebook PROPHET package
import pandas as pd
from fbprophet import Prophet
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
apd = pd.read_csv("C:/DET/ML/Data/AirPassengers.csv")
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
future_dates = apd_model.make_future_dataframe(periods=60, freq='MS') # 36 months created from apd date
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


#------------------------------------------------------------------------------------------------------------
##Timeseries using airpassengers but with divided data 80 & 20 sort of.(115+29 rows)
import pandas as pd
from fbprophet import Prophet
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
apd = pd.read_csv("C:/Users/Amit Srimal/Desktop/DET/ML/Data/AirPassengers.csv")
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

#--------------------------------------------------------------------------------------------------------
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
#------------------------------------------------------------------------------------------------------------

## WALMART DATA SET time series forecasting:
import pandas as pd
from fbprophet import Prophet
from matplotlib import pyplot as plt
wallmart_data= pd.read_csv("C:/Users/Amit Srimal/Desktop/DET/ML/Data/WalmartData/train.csv")
wallmart_data.shape  # (4617600, 4)
# getting data range in terms of Date ie around 22 MONTHS data
wallmart_data.head() # data starts from 1-1-2012
wallmart_data.tail() # data ends to 31-10-2014
# data from 
wallmart_data.head()
wallmart_data.describe
list(wallmart_data) #get header name as ['saledate', 'store_nbr', 'item_nbr', 'units']
# count of Unuqie STORES
wallmart_data.store_nbr.unique() # list of unique values in column StoreNumber
wallmart_data.store_nbr.unique().shape # count of unique storenumber as 45
# count of Unuqie ITEMS
wallmart_data.item_nbr.unique() # list of unique values in column StoreNumber
wallmart_data.item_nbr.unique().shape # count of unique storenumber as 111

##LETs plot few graphs for Store_nbr,Item_nmbr,MonthOfSale, DailySale against the count SOLD
# Plot 1:  the store_nbr with count of units sold.
wallmart_data.iloc[:,[1,3]] # segregating only required columns STORE_NBR & UNITS
wmdp_str= wallmart_data.iloc[:,[1,3]].groupby(['store_nbr']).sum() # applying sum 
wmdp_str.plot(kind='bar') # shows Store wise Sale count SUM valaue
wmdp_str.sort_values(by='units', ascending=False) # retrunds sorted SALE UNIT
# Plot 2:  the item_nbr with count of units sold.
wallmart_data.iloc[:,[2,3]] # segregating only required columns STORE_NBR & UNITS
wmdp_item= wallmart_data.iloc[:,[2,3]].groupby(['item_nbr']).sum() # applying sum 
wmdp_item.plot(kind='bar', color='red') # shows Item wise Sale count SUM valaue
wmdp_item.sort_values(by='units', ascending=False) # retrunds sorted SALE UNIT based on ITEM NUMBER
# Plot 3:  the Monthly Sale with count of units sold.
wmdp_md= wallmart_data.iloc[:,[0,3]]
import datetime
#Getting Month new column in existing data frame
wmdp_md['Month'] = pd.DatetimeIndex(wmdp_md['saledate']).month  
wmdp_md.head()
wmdp_md2= wmdp_md.iloc[:,[2,1]].groupby(['Month']).sum() # applying sum 
wmdp_md2.plot(kind='bar',color='black') # shows Monthly Sale count SUM valaue
wmdp_md2.sort_values(by='units', ascending=False)
# Plot 4:  the Daily Sale with count of units sold.
wmdp_md['Day'] = pd.DatetimeIndex(wmdp_md['saledate']).day
wmdp_md.head()
wmdp_d= wmdp_md.iloc[:,[1,3]].groupby(['Day']).sum() # applying sum 
wmdp_d.plot(kind='bar',color='orange') # shows Monthly Sale count SUM valaue
wmdp_d.sort_values(by='units', ascending=False)

## Now the Actual time series forecasting begins here
# lets create df whch shows monthly sale for these two years.
wallmart_data_frcst= wallmart_data.iloc[:,[0,3]].groupby(['saledate'], as_index=False ).sum() # month sale for two years
wallmart_data_frcst.shape # 1034 x 2 dataframe
# Writing file to system for comparison purpose
wallmart_data_frcst.to_excel("C:/Users/Amit Srimal/Desktop/DET/ML/Data/WalmartData/Sale_Actual.xlsx",sheet_name='sheet1', index=False)

#setting INDEX is critical here else dataframe will loose ds, y format 
wallmart_data_frcst.dtypes # object & int datatypes
# converting to other format as equored by prophet
wallmart_data_frcst['saledate'] = pd.DatetimeIndex(wallmart_data_frcst['saledate'])
wallmart_data_frcst.dtypes # converted to datetime & int datatypes
## Renaming coilumn value as expected by PROPHET
wall_data = wallmart_data_frcst.rename(columns={'saledate': 'ds','units': 'y'})

# visualize the data
ax = wall_data.set_index('ds').plot(figsize=(20, 10))
ax.set_ylabel('Sales Unit')
ax.set_xlabel('Year- Month')
plt.show()

## creating test & training data set  in SEQUENTIAL format.
wm_train= wall_data.head(827)
wm_train.describe()
wm_test= wall_data.tail(207)
wm_test.describe()
wm_train.shape # 827 rows
wm_test.shape # 207 rows

# creating MODEL now
wallmart_model_95 = Prophet(interval_width=0.95)
# lets train model on training dataset
wallmart_model_95.fit(wm_train)
#AND Now extract date from test table
#future_date_95= apd_test.iloc[:,0] # gives series date & prophet expected ds format
# so will use of prophet's in built date making function for next 207 days(so as to verify from test data set)
wm_future_date_95=wallmart_model_95.make_future_dataframe(periods=207) 
# Forecasting using the available test date that's exracted above
wm_forecast_95 = wallmart_model_95.predict(wm_future_date_95)
wm_future_date_95.dtypes ## datatype is datetime64

## Check FOUR critical attributes. Though forecasthas 16 columns from Prophet model.
wm_forecast_95[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
wm_forecast_95[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head() #includes all 827+700 data rows
wm_forecast_95[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail() # ends by 2016-03-07
# lets have a look into last few rows of test set
wm_test.tail() ##printing last 10 values for comparsion purpose
## Compare element by element ro check the DISPCREPANCY in value.

## plotting the result
wallmart_model_95.plot(wm_forecast_95, uncertainty=True)
# plot shows preidcted grpah starting from End of training date.No dots &only predicted line is displayed.
# Visualing components if time series
wallmart_model_95.plot_components(wm_forecast_95)

# LASTLY,Lets take Predicted data for 1034(827+207 predicted) & compare with Actual Sale unit
predicted_sale= wm_forecast_95[['ds', 'yhat']] # just ds & y values
predicted_sale.to_excel("C:/Users/Amit Srimal/Desktop/DET/ML/Data/WalmartData/Sale_Predicted.xlsx",sheet_name='sheet1', index=False)
# Now concatenating two dataframes to view side by side result
side_by_side_comparison = pd.concat([wallmart_data_frcst, predicted_sale], axis=1, sort=False)
side_by_side_comparison.shape # has 1034 by 4 data rows
side_by_side_comparison.head() # units is Actual & yhat is Predicted.
side_by_side_comparison.to_csv("C:/Users/Amit Srimal/Desktop/DET/ML/Data/WalmartData/Sale_ActualVsPredicted.csv", index=False)

#side by side graph
side_by_side_comparison.plot(x='saledate', y=['units', 'yhat'], figsize=(10,5), grid=True)
#------------------------------------------------------------------------------------------------------------













