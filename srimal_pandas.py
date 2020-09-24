######Pandas##### Source: meher krishna patel.pdf & pandas pdf 2000+pages in google drive
##Pandas can handle SERIES, DATAFRAME & PANEL
#Warning: In 0.20.0, Panel is deprecated and will be removed in a future version.
#SERIES: The Series is a one-dimensional array that can store various data types, including mix data types. 
# The row labels/axis labels in a Series are called the index. 

# Pandas, Series is a one-dimensional labeled array capable of holding any data type.
# In layman terms, Pandas Series is nothing but a column in an excel sheet eg. columns with Name, Age and Designation representing a Series

# Any list, tuple and dictionary can be converted in to Series using ’series’ method

import pandas as pd
import numpy as np
# converting tuple to Series
t1 = ('AA', '2012-02-01', 100, 10.2)
s1= pd.Series(t1)
#generating series of random 5 numbers with index
s2 = pd.Series(np.random.randn(5), index=['a', 'b', 'c', 'd', 'e']) # uses numpy & pandas library
#Note: pandas supports non-unique index values. like for group by functions
s2[3:] # gives element after third index value HAVING index start at 1
s2[:3] # gives initial three values of index s2, having index start at 0
s2[-1:] # keeps last index value 
s2[:-1] # removes last index value
s2[3] # gives third index value ONLY.ie 4th place
s2[1:3] # gives index 1 to (3-1) value as o/p i.e. start index from ZERO
s2[[4, 3, 1]] # indexed order is o/p. s2[4, 3, 1] is WRONG & gives key error.
s2[s2>s2.min()] # gives value with greater than min of series.llly max,mean etc can be used.
#s2>s2.min() only gives T/F,implying the index value
s2 + s2 #vector addition
s2 * 2 # vector multiplication
# Series operation automatically align the data based onlabel.
#If a label is not found in one Series or the other, the result will be marked as missing NaN
s2[1:] + s2[:-1]  #result has NaN values
np.exp(s2) # exponential value 
s2['d'] # gives d index value
'e' in s2 # checks if e present in s2, gets T/F value as o/p
s2.get('f', '23') #check for f & if not present,assign 23. if abcde, no assignment is done
#checking index of dictionary and assigned index value for series
#Naming attributes for Series:
s3 = pd.Series(np.random.randn(5), name='something')
s3.name # gives something
s4 = s3.rename("different") # something remaned as different

h2 = {'a' : 0., 'b' : 1., 'c' : 2.} # dictionary data set
pd.Series(h2) # produces series having a,b,c as index
pd.Series(h2, index=['b', 'c', 'd', 'a']) # gives list against b,c,d,a with d as NaN.
#scalar values below
pd.Series(5., index=['a', 'b', 'c', 'd', 'e']) # gives repated 5 aginst index a,b,c,d,e

# converting dictionary to Series
data1={'brand':['Hero','RE','Bajaj'],'model':['Passion','Classic350','pulsar']} ##Dictonary of 2 wheeler
df1=pd.DataFrame(data1) #create df from dictonary
print(df1)
type(df1) #check data type of final object
#adding new column to above data frame
df1['price']=[55000,150000,85000] ## set 'unjvown if value not known at this stage'
#Setting CUSTOM index to dataframe that' created
df1.index=['model1','model2','model3']
##Setting any column as index attribute
df1.set_index(['brand']) # just gives view, need to store in other dataframe as below
df2=df1.set_index(['brand'])
print(df2)
##Accessing data frame via
#1. column wise
df1['model'] # gives element against INDEX name.
#2. row wise
df1.ix['model1'] # needs to give INDEX number of desired row
#3. All rows of column
df1.ix[:,'brand']
df1.ix[0:,'model']  ## try with 0,1,2 and check option

data={'a':111,'b':22,'c':33}
s=pd.Series(data)
print(s)

data2={'Name':['Amit','Annu'],'Age':[31,28]}
df=pd.DataFrame(data2,index=['Sr No. 1','Sr No. 2'])
print(df)
df['Name'] # gives name of all column
df['Name','Age'] # gives name & Age of data frame

##dropping & Deleting column column name from data frame
del df['column name']
df.drop('column name', axis = 1) # drop column shares from data frame

#2D dataframe using DICTIONARY with indexed approach
d = {'one' : pd.Series([1., 2., 3.], index=['a', 'b', 'c']),
     'two' : pd.Series([1., 2., 3., 4.], index=['a', 'b', 'c', 'd'])}
df = pd.DataFrame(d)
df.index # gives index name of above created dataframe
df.columns # gives column name of above created dataframe
pd.DataFrame(d, index=['d', 'b', 'a']) # gives DF for index d,b,a ONLY
pd.DataFrame(d, index=['d', 'b', 'a'], columns=['two', 'three']) # gives DF with two,three. Three has NaN as values.
#inserting new column
#By default, columns get inserted at the end
df['newcolumn'] = df['existingcolumn'][:2] # df gets new column from column having with initial TWO values.
 # to insert at DESIRED location ie 2nd column with name & value respectively
df.insert(1, 'Surname', 'Srimal')

# Like DPLYR's mutate, python has ASSIGN function to compute based on ROW DATA
(iris.assign(sepal_ratio = iris['SepalWidth'] / iris['SepalLength']).head())
#ASSIGN operation always returns a copy of the data, leaving the original DataFrame untouched.
#query for length and then mutate condition is applied as below:
(iris.query('SepalLength > 5').assign(sepal_ratio = iris['SepalWidth'] / iris['SepalLength']).head())
# Other operations as:
df.loc['b'] #select row by label
df.iloc[2] # select row by integer location
df['a':'b'] # gives row a & b data
df[1:3] # gives data from 1 to 3-1 index. start index from ZERO.
#llly try df[2:],df[:4] etc to check

#Data Alignment
df1 = pd.DataFrame(np.random.randn(10, 4), columns=['A', 'B', 'C', 'D']) # df with 10row &7col.
df1 = pd.DataFrame(np.random.randn(10, 4), columns=list('ABCD')) # gives column as ABOVE 
df2 = pd.DataFrame(np.random.randn(7, 3), columns=['A', 'B', 'C'])
df1+df2 # gives NaN for D row as df2 has no D column in place
df1 - df1.iloc[0] # removes row o data from resulting df.

#creating df with datetime index
time_index = pd.date_range('1-1-2000', periods=8) # or 1/1/2000 as datetime
df3 = pd.DataFrame(np.random.randn(8, 3), index=time_index, columns=list('ABC'))
type(df3['A'])
df3.sub(df3['B'], axis=0) # to remove column B from data frame along column
df3 * 5 + 2
1/df3
df3**3
df3[:5].T # transpose operation for first 5 rows
np.asarray(df3) # dataframe to array conversion
df3.info() # gives information of dataframe...

#handling boolean data frame
df1 = pd.DataFrame({'a' : [1, 0, 1], 'b' : [0, 1, 1] }, dtype=bool)
df2 = pd.DataFrame({'a' : [0, 1, 1], 'b' : [1, 1, 0] }, dtype=bool)
df1 & df2
df1 | df2
df1 ^ df2
-df1 # inverted true false of df1

s1 = pd.Series(np.arange(5,10))
a=s1.dot(s1) # 255 as 5*5+ ans so on Element wise Multiplication
#Handling paper width of DISPLAY
#1. set max display width of entire dataframe
pd.set_option('display.width', 40) # default is 80.
pd.DataFrame(np.random.randn(3, 12))
#2. set max width of column having large data
pd.set_option('display.max_colwidth',30) # extra data visible as ... extension

##BIG DATA HANDLING
#pandas has support for accelerating certain types of binary numerical and boolean operations using the numexpr
#library and the bottleneck libraries.These libraries are especially useful when dealing with large data sets, and provide large speedups.NUMEXPR uses
#smart chunking, caching, and multiple cores. BOTTLENECK is a set of specialized cython routines that are especially
#fast when dealing with arrays that have nans.
#These are both enabled to be used by default.
#but in new. pd.set_option('compute.use_bottleneck', False)
#pd.set_option('compute.use_numexpr', False)
##

#With binary operations between pandas data structures, there are two key points of interest:
#1. broadcasting behaviour # ie. methods add(), sub(), mul(), div() and related functions radd(), rsub(), ...
#2. missing data in computation

#broadcasting
df = pd.DataFrame({'one' : pd.Series(np.random.randn(3), index=['a', 'b', 'c']),
                   'two' : pd.Series(np.random.randn(4), index=['a', 'b', 'c','d']),
                   'three' : pd.Series(np.random.randn(3), index=['b', 'c','d'])})
row = df.iloc[1] # gives all column data of row 1
column = df['two']
df.sub(row, axis='columns') # subtract columns axis value from df
df.sub(row, axis=1) # same command as above. 0 for rows & 1 for columns
df.sub(column, axis='index') # sub columns from df along index
df.sub(column, axis=0) # same command as above. 0 for rows & 1 for columns
# create multidimensional Index
multi_index = pd.MultiIndex.from_tuples([(1,'a'),(1,'b'),(1,'c'),(2,'a')],names=['first','second'])

#Missing data / operations
#Rules as here: nan +nan= nan, number +nan= number
#eplace NaN with some other value using fillna operation

##Flexible Opeartion:
#Series and DataFrame have the binary comparison methods eq, ne, lt, gt, le, and ge whose behavior is analogous
#to the binary arithmetic operations
df.gt(df2) # greater than & gives T/F value output in data frame
df.ne(df2)
##Boolean reduction
#apply the reductions: empty, any(), all(), and bool() to provide a way to summarize a boolean result.
(df > 0).all()
(df > 0).any()
(df > 0).any().any()
df.empty # check in datraframe is empty or not

#Checking objects Equivalnce
#NDFrames (such as Series, DataFrames, and Panels) have an equals() method for testing equality
df+df == df*2 # may give FALSE o/p coz of NaN if present
(df+df).equals(df*2) # best to use to check equivalent dataframes & needs to be of SAME ORDER to be TRUE.

##Comparing array like objects
#element-wise comparisons when comparing a pandas data structure with a scalar value:
pd.Series(['foo', 'bar', 'baz']) == 'foo'
pd.Series(['foo', 'bar', 'baz']) == pd.Index(['foo', 'bar', 'qux'])
pd.Series(['foo', 'bar', 'baz']) == np.array(['foo', 'bar', 'qux'])
# if we try comparing unequal length, we get valueError
##But different in Numpy as below
np.array([1, 2, 3]) == np.array([2]) # gives False True False
np.array([1, 2, 3]) == np.array([1,2]) # gives False o/p

#Combining overlapping data sets using the is combine_first()
df1.combine_first(df2) # combines using first df. create df with NaN & check result
#llly try the df2.combine_first(df1) o/p to check logic of function

## Descriptive Statistics
#Most of these are aggregations (hence producing a lower-dimensional result) like sum(), mean(), and
#quantile(), but some of them, like cumsum() and cumprod(), produce an object of the same size.
df.mean(0)
df.mean(1)
#All such methods have a skipna option signaling whether to exclude missing data (True by default):
df.sum(0, skipna=False) # considering missing data
df.sum(axis=1, skipna=True)

#Series also has a method nunique() which will return the number of unique non-NA values:
series = pd.Series(np.random.randn(500))
series.nunique() # givs 500 unique value
series[10:20]= 5 # lets assign value to sliced data
series.nunique() # gives 491 & not 500 o/p as value

#describe() function which computes a variety of summary statistics about a Series or the
#columns of a DataFrame (excluding NAs of course)
series.describe() # considering numerical series/ data frame range
series.describe(percentiles=[.05, .25, .75, .95]) # gets specific stat value extracted but By default, the median is always included.
s = pd.Series(['a', 'a', 'b', 'b', 'a', 'a', np.nan, 'c', 'd', 'a']) # non numercal series
s.describes() # gives unique & frequency related o/p
frame = pd.DataFrame({'a': ['Yes', 'Yes', 'No', 'No'], 'b': range(4)}) # mixed data
frame.describe() # will have stats for numerical data for mixed df
frame.describe(include=['object']) # only non-numerical data object
frame.describe(include=['number']) #  only numerical data object
frame.describe(include='all')

##Index of max & min values
s1.idxmin(), s1.idxmax() # gives index of min & max value considering s1 is series of random numbers
df1.idxmin(axis=0) # MIN VALUE INDEX ACORSS ROW
df1.idxmin(axis=1) # min value index across column
##Note: idxmin and idxmax are called argmin and argmax in NumPy.

#histogramming-value counting method
#The value_counts() Series method and top-level function computes a histogram of a 1D array of values.
s = pd.Series(data = np.random.randint(0, 7, size=50))
s.value_counts() # gives count fo occurence of value in above data series
#most frequently occurring value(s) (the mode) of the values in a Series or DataFrame:
s.mode() # gives directly value of Mode-max ocurence

##Continuous values can be discretized using the cut() (bins based on values) and qcut() (bins based on sample
#quantiles) functions:
arr = np.random.randn(20)
factor = pd.cut(arr, 4) # bin based values
factor = pd.cut(arr, [-5, -1, 0, 1, 5]) # predefined bin value
factor = pd.qcut(arr, [0, .25, .5, .75, 1]) #quantile cut
pd.value_counts(factor) # gives count of occurence against each bin range
factor = pd.cut(arr, [-np.inf, 0, np.inf]) # passing infinite bean for  arr of 20 numbers

##FUNCTION OPERATION and four methods of applying:
#1. Tablewise Function Application: pipe()
#2. Row or Column-wise Function Application: apply()
#3. Aggregation API: agg() and transform()
#4. Applying Elementwise Functions: applymap()

#1. Pipe method or Chaining method. just like dply >%> pipe chain call
#2. Row or Column-wise Function Application
df.apply(np.mean)
df.apply(np.mean, axis=1)
df.apply(np.cumsum)
tsdf = pd.DataFrame(np.random.randn(1000, 3), columns=['A', 'B', 'C'],index=pd.date_range('1/1/2000', periods=1000))
tsdf.apply(lambda x: x.idxmax()) # gives date of max values
#Finally, apply() takes an argument raw which is False by default, which converts each row or column into a Series
#before applying the function. When set to True, the passed function will instead receive an ndarray object, which has
#positive performance implications if you do not need the indexing functionality.
#3. A. Aggregate API: aggregate(), or the alias agg().
tsdf.agg(np.sum)
tsdf.agg('sum')
tsdf.sum()
tsdf.A.agg('sum') # single aggregation on a series by column A
 # aggregation over multiple function
tsdf.agg(['sum','mean'])
tsdf.A.agg(['sum', 'mean']) # multiple agg() for series function
tsdf.agg({'A': 'mean', 'B': 'sum'}) # agg() over DICTIONARY object
tsdf.agg({'A': ['mean', 'min'], 'B': 'sum'})
#When presented with mixed dtypes that cannot aggregate, .agg will only take the valid aggregations.
#3 B. The transform() method returns an object that is indexed the same (same size) as the original. This API allows
#you to provide multiple operations at the same time rather than one-by-one. Its API is quite similar to the .agg API.
# Explore more on pg 547-599/ 2215 for transform & related options
 # 4. Applying Elementwise Functions
 #Since not all functions can be vectorized (accept NumPy arrays and return another array or value), the methods
#applymap() on DataFrame and analogously map() on Series accept any Python function taking a single value and
#returning a single value.
# Explore more on pg 602/ 2215 for transform & related options

##Reindexing and altering labels
s = pd.Series(np.random.randn(5), index=['a', 'b', 'c', 'd', 'e'])
s.reindex(['e', 'b', 'f', 'd']) # 'f' will have NaN written against it
df.reindex(index=['c', 'f', 'b'], columns=['three', 'two', 'one']) # reindex abc to cfb & onetwothree to threetwoone
# Index objects containing the actual axis labels can be shared between objects.
rs = s.reindex(df.index)
rs.index is df.index # gives TRUE as o/p. means series & df index are similars
df.reindex(['c', 'f', 'b'], axis='index')
df.reindex(['three', 'two', 'one'], axis='columns')

# Reindexing to align with another object with REINDEX_LINE() operation
df4=pd.DataFrame(np.random.randn(3, 3))
df5=pd.DataFrame(np.random.randn(5,5))
df5.reindex_like(df4) # converts df5 to df like df4 having 3*3 design, other values are truncated

## Filling while reindexing
#reindex() takes an optional parameter method which is a filling method chosen
#pad / ffill: Fill values forward
#bfill / backfill:Fill values backward
#nearest: Fill from the nearest index value
# IMP: These methods require that the indexes are ordered increasing or decreasing.If not , it gives ValueError.
# instead use fillna() and interpolate() will not make any checks on the order of the index.
rng = pd.date_range('1/3/2000', periods=8)
ts = pd.Series(np.random.randn(8), index=rng)
ts2 = ts[[0, 3, 6]]
ts2.reindex(ts.index) # gives NaN for non 0,3,6 index values
ts2.reindex(ts.index, method='ffill') #0th index value filled in place of NaN.
ts2.reindex(ts.index, method='bfill') #same as above but from BACK.
ts2.reindex(ts.index, method='nearest')

##Aligning objects with each other with align FASTEST methos to align two objects, supports join argument.
#• join='outer': take the union of the indexes (default)
#• join='left': use the calling object’s index
#• join='right': use the passed object’s index
#• join='inner': intersect the indexes
s = pd.Series(np.random.randn(5), index=['a', 'b', 'c', 'd', 'e'])
s1 = s[:4]
s2 = s[1:]
s3=s1.align(s2) # retruns a TUPLE always
s1.align(s2, join='inner')
s1.align(s2, join='left')
# For DataFrames, the join method will be applied to both the index and the columns by default:
df.align(df2, join='inner')
df.align(df2, join='inner', axis=0)

##Filling while reindexing
rng = pd.date_range('1/3/2000', periods=8)
ts = pd.Series(np.random.randn(8), index=rng)
ts2 = ts[[0, 3, 6]]
ts2.reindex(ts.index) # ts2 with no matching valuw will have NaN
ts2.reindex(ts.index, method='ffill')
ts2.reindex(ts.index, method='bfill')
ts2.reindex(ts.index, method='nearest')
# when we need to define the limits, The limit and tolerance arguments provide additional control over filling while reindexing
#Limit specifies the maximum count of consecutive matches:
ts2.reindex(ts.index, method='ffill', limit=1) # forward fill only 1 level up & then NaN.
#tolerance specifies the maximum distance between the index and indexer values:
ts2.reindex(ts.index, method='ffill', tolerance='1 day')

## Dropping Labels from axis: A method closely related to reindex is the drop() function
df.drop(['a', 'd'], axis=0) # drops index a,d from dataframe alon row line
df.drop(['one'], axis=1) # drops column one along clolumn line

##Renaming / mapping labels
s= pd.Series(np.random.randn(5), index=['a','b','c','d','e'])
s.rename(str.upper) # renames abcde to ABCDE
# consider df having abcde as row, one two as column. These gets renamed
df.rename(columns={'one': 'foo', 'two': 'bar'},index={'a': 'apple', 'b': 'banana', 'd': 'durian'})
#If the mapping doesn’t include a column/index label, it isn’t renamed. Also extra labels in the mapping don’t throw an error.

##ITERATION OPERATION (generally slow operation in PANDAS)###########
# CYTHON or NUMBA must be used for better performance
#The behavior of basic iteration over pandas objects depends on the type. When iterating over a Series, it is regarded
#as array-like, and basic iteration produces the values. Other data structures, like DataFrame and Panel, follow the
#dict-like convention of iterating over the “keys” of the objects. In short as below:
#Series: produces values
# DataFrame: produces column labels
df = pd.DataFrame({'col1' : np.random.randn(3), 'col2' : np.random.randn(3)},index=['a', 'b', 'c'])
for col in df:
    print(col) # produces two column names
#To iterate over the rows of a DataFrame, you can use the following methods: 
#1.iterrows():Iterate over the rows of a DataFrame as (index, Series) pairs.
#2.itertuples(): Iterate over the rows of a DataFrame as namedtuples of the values & FASTER than Above
# Warning: You should never modify something you are iterating over..
#Depending on the data types, the iterator returns a copy and not a view, and writing to it will have no effect!
df = pd.DataFrame({'a': [1, 2, 3], 'b': ['a', 'b', 'c']})
for index, row in df.iterrows():
    row['a'] = 10
df 
# df remain unchanges even we tried changed a vakue to 10.hence dont chnage value iterating.
#iterrows() allows you to iterate through the rows of a DataFrame as Series objects
for row_index, row in df.iterrows():
    print('%s\n%s' % (row_index, row))
#Note: Because iterrows() returns a Series for each row, it does not preserve dtypes across the rows (dtypes are
#preserved across columns for DataFrames
df_orig = pd.DataFrame([[1, 1.5]], columns=['int', 'float'])
df_orig.dtypes
row = next(df_orig.iterrows())[1]
row
###To preserve dtypes while iterating over the rows, it is better to use itertuples() which returns namedtuples of the
#values and which is generally much faster as iterrows.
#The itertuples() method will return an iterator yielding a namedtuple for each row in the DataFrame. The first
#element of the tuple will be the row’s corresponding index value, while the remaining values are the row values.

##VECTORIZED STRING METHODS####
s = pd.Series(['A', 'B', 'C', 'Aaba', 'Baca', np.nan, 'CABA', 'dog', 'cat'])
s.str.lower() # converts everything to lower case. np.nan returns NaN.

## Sorting: two types as 'sorting by label' and 'sorting by actual values'.
#1. by Index:
unsorted_df = df.reindex(index=['a', 'd', 'c', 'b'],columns=['three', 'two', 'one'])
unsorted_df.sort_index()
unsorted_df.sort_index(ascending=False)
unsorted_df.sort_index(axis=1)
unsorted_df['three'].sort_index()
#2. By values
df1 = pd.DataFrame({'one':[2,1,1,1],'two':[1,3,2,4],'three':[5,4,3,2]})
df1.sort_values(by='two') # by argument takes coumnname
df1[['one', 'two', 'three']].sort_values(by=['one','two']) # by list of column names

## Search Sorted
ser = pd.Series([1, 2, 3])
ser.searchsorted([0, 3])
ser.searchsorted([0, 4])
## smallest / largest values for SERIES
s=pd.Series(np.random.permutation(20))
s.sort_values()
s.nsmallest() # 5 smallest like head & tail, pass argument to get other value
s.nsmallest(3)
s.nsmallest(1) # returns smallest pervalue
s.nlargest(3) # similarly for LARGEST value set.

# Smallest & Largest for DATAFRAMES
df2 = pd.DataFrame({'a': [-2, -1, 1, 10, 8, 11, -1],'b': list('abdceff'),'c': [1.0, 2.0, 4.0, 3.2, np.nan, 3.0, 4.0]})
df2.nlargest(3, 'a') # largest 3 values based on column a value
df2.nlargest(5, ['a', 'c'])
df2.nsmallest(3, 'a')
df2.nsmallest(5, ['a', 'c'])

##TIP VALUES: To be clear, no pandas methods have the side effect of modifying your data; almost all methods return new objects,
#leaving the original object untouched. If data is modified,

###DTYPES: ###
#The main types stored in pandas objects are float, int, bool, datetime64[ns] and datetime64[ns, tz]
#(in >= 0.17.0), timedelta[ns], category and object. In addition these dtypes have item sizes, e.g. int64
#and int32.
# assuming we have defined dft as dataframe
dft.dtypes # gives data type of each column 
dft.get_dtype_counts() # gives count of each type object in dataframe.
# By default integer types are int64 and float types are float64, REGARDLESS of platform (32-bit or 64-bit).
## NUMPY, however will choose platform-dependent types when creating arrays.

## UPCASTING: 
#Types can potentially be upcasted when combined with other types, meaning they are promoted from the current type
#(say int to float)
#The values attribute on a DataFrame return the lower-common-denominator of the dtypes, meaning the dtype that
#can accommodate ALL of the types in the resulting homogeneous dtyped numpy array. This can force some upcasting.
#Upcasting is always according to the numpy rules.
#If two different dtypes are involved in an operation, then the more general one will be used as the result of the operation

##CONVERT DATATYPE: astype
#to explicitly convert dtypes from one to another. These will by default return a
#copy, even if the dtype was unchanged (pass copy=False to change this behavior). In addition, they will raise an
#exception if the astype operation is invalid.
df4 = pd.DataFrame({'a':[2,1,1,1],'b':[1,3,2,4],'c':[5,4,3,2]})
df4.dtypes # has int64 as dtype
df4.astype('float32').dtypes # converts to float32 dtype
#convert based on column
df4[['a','b']] = df4[['a','b']].astype(np.uint8) # convert column a,b as unit8 datatype
df4.dtypes

## OBJECT CONVERSION ##
import datetime
df5 = pd.DataFrame([[1, 2],['a', 'b'],[datetime.datetime(2016, 3, 2), datetime. datetime(2016, 3, 2)]])
df5=df5.T # transpose operation DISTRURBS the object definition
df5.dtypes # returns object for all three column
df5.infer_objects().dtypes # to correctly RECOGNIZE the data type

# GOTCHAS: The dtype of theinput data will be preserved in cases where nans are not introduced.
df6 = pd.DataFrame({'a':[-2,1,1,1],'b':[1,3,-2,4],'c':[5,-4,3,2]})
df6.dtypes # has int64 as dtype
casted = df6[df6>0]
casted.dtypes # has float64 as dtype
#if initial is float64, it will remian unchnages after selection casting

##functions are available for one dimensional object arrays or scalars to perform hard conversion of objects
m = ['1.1', 2, 3]
pd.to_numeric(m) #returns numeric array
#refer 581/2215 for other details, ONLY IF NEEDED.

##Selecting columns based on dtype
#assuming "df" having columns of bolean, integer, category,datetime etc as datatype
#select_dtypes() has two parameters "include and exclude".
df.select_dtypes(include=[bool]) # select only boolean column datatype
df.select_dtypes(include=['number', 'bool'], exclude=['unsignedinteger'])
df.select_dtypes(include=['object']) # select string column

##reading files in python with pandas
import pandas as pd
casts = pd.read_csv('path of csv file', index_col=None)
casts = pd.read_excel('path of xls file', index_col=None)
# index_col = None : there is no index i.e. first column is data
casts.head()
casts.tail()
casts.head(3)  # setting desired number to view

#If there is some error while reading the file due to encoding, then try for following option as well
titles = pd.read_csv('titles.csv', index_col=None, encoding='utf-8')

#If we simply type the name of the DataFrame (i.e. cast in below code), then it will show the first 30 and last
#20 rows of the file along with complete list of columns.
# but this can be set using
pd.set_option('max_rows', 10, 'max_columns', 10)

# checks the number of rows presenty in data frame
len(titles)  

## DATA OPERATIONS
#Any row or column of the DataFrame can be selected by passing the name of the column or rows. After selecting
#one from DataFrame, it becomes one-dimensional therefore it is considered as SERIES.
titles['title'] # gets the column wise data for title of titles
titles.ix['index number'] # gives the row data for index number
 
# filter data:  by providing some boolean expression in DataFrame.
after85 = titles[titles['year'] > 1985] # movies after 1985
movies90 = titles[titles['year']>=1990 & titles['year']<2000] # AND,OR operation for movies in 1990-2000 means 90 decade

# sorting data: Sorting can be performed using ‘sort_index’ or ‘sort_values’ keywords.
# by default, sort by index i.e. row header
titles[titles['title'] =='macbeath'].sort_index() # increasing sorting on INDEX 
titles[titles['title'] =='macbeath'].sort_values('year') # sorting increasing on YEAR value
 
 ##NULL VALUES: various columns may contains no values, which are usually filled as NaN.
 #null values can be easily selected, unselected or contents can be replaced by any other values e.g. empty strings or 0 etc.
 titles[titles['column name']].isnull() # return T/F for column name having Null value.
 titles[titles['column name']].isnull().head() # view of above for first five row data
 # filling Nan values
 titles[titles['column name']].isnull().fillna('Value')
 ## try with ffill(forward fill), and bfill(backward fill)
 
 # notnull is opposite of isnull
 titles[titles['column name']].isnull() # return T/F for column name having not null value.
 
 ## STRING OPERATIONS using ’.str.’ option.
 titles[titles['title'].str.startswith("Maa ")].head(3) # movie that start with word Maa
 
 # Counting: Total number of occurrences can be counted using ’value_counts()’ option.
 titles['year'].value_counts().head() # counts frequency of column name YEAR.
 
 ##PLOTS: Pandas supports the matplotlib library and can be used to plot the data as well.
 import matplotlib.pyplot as plt
 t = titles
 p = t['year'].value_counts()
 p.sort_index().plot() # sorted to give better plot visualization
 <matplotlib.axes._subplots.AxesSubplot object at 0xaf18df6c>
 plt.show()
 
 ## GROUP BY: Data can be grouped by columns-headers.
 cg = titles.groupby(['year'])
 cg = titles.groupby(['year']).size()  #The size() option counts the total number for rows for each year;
 
 casts=c
 cf = c[c['name'] == 'Aaron Abrams']
 cf.groupby(['year']).size().head()
 cf.groupby(['year', 'title']).size().head() ##grouping by multiple column headers . first year and then title grouping is done.
 #group the items by year and see the maximum rating in those years,
 c.groupby(['year']).n.max().head()
 c.groupby(['year']).n.min().head() # minimum ratings
 c.groupby(['year']).n.mean().head() # mean rating each year
 
 ## Groupby with custom field
 # we want to group the data based on decades, then we need to create a custom groupby field.
 # decade conversion formula: 1985//10 = 198, 198*10 = 1980
 decade = c['year']//10*10
 c_dec = c.groupby(decade).n.size()
 c_dec.head()
 
 ##UNSTACKING: ’unstack’, which allows to create a new DataFrame based on the grouped Dataframe.
 c.groupby( [c['year']//10*10, 'type'] ).size().head(8)
 #Now we want to compare and plot the total number of actors and actresses in each decade.
 c_decade = c.groupby( ['type', c['year']//10*10] ).size()
 # create new dataframe using unstack command
 c_decade.unstack()
 
 ##plotting as below:
 #plot 1 having grouped by Actor & Actoress
 c_decade.unstack().plot()
 <matplotlib.axes._subplots.AxesSubplot object at 0xb1cec56c>
 plt.show()
 c_decade.unstack().plot(kind='bar')
 <matplotlib.axes._subplots.AxesSubplot object at 0xa8bf778c>
 plt.show()
 
 #To plot the data side by side, use unstack(0) option as shown below (by default unstack(-1) is used)
  #plot 2 having grouped by year
 c_decade.unstack(0)
 c_decade.unstack(0).plot(kind='bar')
 <matplotlib.axes._subplots.AxesSubplot object at 0xb1d218cc>
 plt.show()
 
 ###MERGE OPERATIONS
 #1. Merge with different files
 release = pd.read_csv('release_dates.csv', index_col=None)
 c_amelia = casts[ casts['title'] == 'Amelia'] # for movie Amelia
 release [ release['title'] == 'Amelia' ].head()
 c_amelia.merge(release).head()
 
 #2. Merge with itself
 c = casts[ casts['name']=='Aaron Abrams' ]
 c.merge(casts, on=['title', 'year']).head()
 
 #The problem with above joining is that it displays the ’Aaron Abrams’ as his co-actor as well and can be solved by
 c_costar = c.merge (casts, on=['title', 'year'])
 c_costar = c_costar[c_costar['name_y'] != 'Aaron Abrams']
 c_costar.head()

### INDEXING
 #1. creating Index
 import pandas as pd
 cast = pd.read_excel('C:/Users/akmsrimal/Downloads/SampleSuperstore.xls', index_col=None)
 cast.head()
 %%time
 %%timeit
 c = cast.set_index(['title']) # title is used as INDEX here
 ##Above line will not work for multiple index; POINT TO NOTE
 
 #2. accessing indexed data using 'loc' function
 c.loc['Macbeth']
 
 #3. Sorting & indexing improves performance
 cs = cast.set_index(['title']).sort_index()
 cs.loc['Macbeth']
 
 ## Multiple Index:
  cm = cast.set_index(['title', 'n']).sort_index() # indexed for title & n column
  cm.loc['Macbeth']
  # show Macbeth with ranking 4-18
  cm.loc['Macbeth'].loc[4:18]
  cm.loc['Macbeth'].loc[4] # series returned for one data matching cases
  
  ##Reset Index
  cm = cm.reset_index('n') # removes n index from cm data frame
  
 ### DAta Processing:
 ## Hierarchail Indexing:
 #1. Creating Multiple Index
 import pandas as pd
 data = pd.Series([10, 20, 30, 40, 15, 25, 35, 25], index = [['a', 'a','a', 'a', 'b', 'b', 'b', 'b'], ['obj1', 'obj2', 'obj3', 'obj4', 'obj1','obj2', 'obj3', 'obj4']])
 # gives series having a,b each and both having obj1-2-3-4 element, hence TWO-LEVEL of INDEXING
 data.index # gives view of indexing implemented
 
 #2.Partial Indexing
 data['b'] # will extract data for b index having obj1-2-3-4.
 data[:, 'obj2'] # will extract data for both initial a,b index and obj2 level	
 
 ## Data Unstacking
 #Unstack changes the row header to column header. Since the row index is changed to column index, therefore the Series will become the DataFrame.
 #1. unstack based on first level i.e. a, b
 data.unstack(0)
 #2. unstack based on second level i.e. obj
 data.unstack(1)
 #3. # by default innermost level is used for unstacking
 d= data.unstack()
 
 ## data STACKING: stack()’ operation converts the column index to row index again.
 d.stack() # convert above d dataframe back to row indexing level
 
 ## Column Indexing
 #Remember that, the column indexing is possible for DataFrame only (not for Series), because column-indexing
 #require two dimensional data.
 # lets create data frame with multiple level indeximport numpy as np
    df = pd.DataFrame(np.arange(12).reshape(4, 3),
    index = [['a', 'a', 'b', 'b'], ['one', 'two', 'three', 'four']],
    columns = [['num1', 'num2', 'num3'], ['red', 'green', 'red']] )
	
 # Display row index
   df.index
 # Display row index
   df.columns
   
# Giving name to index
df.index.names=['key1', 'key2'] # gives key1-2 name to row headers index
df.columns.names=['n', 'color']  # gives n,color name to column headers index 

##partial indexing on multilevel data frame
 df['num1'] 
 df.ix[:, 'num1']
 df.ix['a']
 # access row 0 only
 df.ix[0]
 
 ### Swap and sort level
 df.swaplevel('key1', 'key2') #swapping level between key1-2
 df.sort_index(level='key2')  # sorting on key2 alphabetical order.
 
 ##Summary statistics by level
 #1. add all rows with similar key1 name
 df.sum(level = 'key1')
 
 #2. add all the columns based on similar color
 df.sum(level= 'color', axis=1)
 
 ##READING FILES
 # data frame from csv file
 df = pd.DataFrame.from_csv('ex1.csv', index_col=None)
 # read csv
 df = pd.read_csv('ex1.csv')
 #reading table
 df = pd.read_table('ex1.csv', sep=',')
 
 ## around headers in reading files
 # set header as none, default values will be used as header
 pd.read_csv('ex2.csv', header=None)
 # specify the header using 'names'
 pd.read_csv('ex2.csv', names=['a', 'b', 'c', 'd', 'message'])
 # specify the row and column header both
 pd.read_csv('ex2.csv', names=['a', 'b', 'c', 'd', 'message'], index_col='message')
 # removing certain rows in csv,excel etc.
 pd.read_csv('ex4.csv', skiprows=[0,2,3])
 
 ##WRITING DATA FILE : The ’to_csv’ command is used to save the file.
 #1. d.to_csv('d_out.csv')
 #2. d.to_csv('d_out2.csv', header=False, index=False) #saves without headers
 
 ##MERGE OPERATIONS
 #Merge or joins operations combine the data sets by liking rows using one or more keys.
 #1. ’Many to one’ merge joins the Cartesian product of the rows, e.g. lets say df1 and df2 has total 3 and 2 rows of ’b’ respectively, therefore join will result in total 6 rows.
 pd.merge(df1, df2) # or pd.merge(df1, df2, on='key') # assuming df1,df2 has similar common key 
 pd.merge(df1, df2, left_on='key1', right_on='key2')  # if df1,df2 has different keys to merge
 #uncommon entries in DataFrame ’df1’ and ’df2’ are missing from the merge operation. 
 
 ##JOINS
 #By default, pandas perform the INNER join, where only common keys are merged together.
 #To perform OUTER join, we need to use ’how’ keyword which can have 3 different values i.e. ’left’, ’right’ and ’outer’.
 #1. pd.merge(df1, df2, left_on='key1', right_on='key2', how="left") #left join
 #2. pd.merge(df1, df2, left_on='key1', right_on='key2', how="right") # right join
 #3. pd.merge(df1, df2, left_on='key1', right_on='key2', how="outer") # outer join
 
 ## DATA CONCATENATION: 
 #concatenation based on union or intersection of data along with labeling to visualize the grouping.
 s1 = pd.Series([0, 1], index=['a', 'b'])
 s2 = pd.Series([2, 1, 3], index=['c', 'd', 'e'])
 s3 = pd.Series([4, 7], index=['a', 'e'])
 s4 = pd.concat([s1, s2]) # concatenation of s1 &s2
 s5 = pd.concat([s1, s2], axis=1) # joins on axis 1
 
# it is difficult to identify the different pieces of concatenate operation. We can provide ’keys’
#to make the operation identifiable
 s6 = pd.concat([s1, s2, s3], keys=['one', 'two', 'three'])
 
 ##ALL ABOVE Concatenation for UNION based, So to create Intersection basedm follow below method:
 pd.concat([s1, s3], join='inner', axis=1)
 
 #Dataframe concatenation
pd.concat([df1, df2], join='inner', axis=1, keys=['one', 'two'])
# Same as above but by passing data frame as Dictionary
pd.concat({ 'level1':df1, 'level2':df2}, axis=1, join='inner') # dictionary approach

###DATA TRANSFORMATION: means cleaning and filtering the data 
 #1. Removing duplicates: with ’drop_duplicates’ command.
 df = pd.DataFrame({'k1':['one']*3 + ['two']*4,'k2':[1,1,2,3,3,4,4]}) # df with duplicates
 df.duplicated() # gets view of duplicated 
 df.drop_duplicates() # drops the duplicates
 
 #Currently, last entry is removed by drop_duplicates command. If we want to keep the last entry, then ’keep’ keyword can be used,
 df.drop_duplicates(keep="last")
 
 #drop all the duplicate values from based on the specific columns lets say k1
 df.drop_duplicates(['k1']
 # drop if k1 and k2 column matched
 df.drop_duplicates(['k1', 'k2'])
 
 #2. Replacing values
 df.replace('one', 'two') # replaces one with two
 df.replace(['one', 3], ['two', '30']) # replace multiple instance one with two, 3 with 30
 df.replace({'one':'One', 3:30}) # replace multiple with DICTIONARY APPROACH
 
 ##GROUPBY & DATA AGGREGATION
 #1. Groupby
 df = pd.DataFrame({'k1':['a', 'a', 'b', 'b', 'a'],'k2':['one', 'two', 'one', 'two', 'one'],'data1': [2, 3, 3, 2, 4],'data2': [5, 5, 5, 5, 10]})
 gp1 = df['data1'].groupby(df['k1']) # create a group based on ’k1’ and find the mean value
 gp1.mean()
 gp2 = df['data1'].groupby([df['k1'], df['k2']]) # passing multiple keys for grouping
 gp2.mean()
 
 #2. Iterating over group
 #The groupby operation supports iteration which generates the tuple with two values i.e. group-name and data.
#NEED to check th epdf Page 47 0f 64 for details, if needed.
 
 #3. Data aggregation: perform various aggregation operation on the grouped data
 gp1.max()
 gp2.min()
 
 
 ####TIME SERIES #########
 #A series of time can be generated using ’date_range’ command. In below code, ’periods’ is the total number of
#samples; whereas freq = ’M’ represents that series must be generated based on ’Month’.
 #By default, pandas consider ’M’ as end of the month. Use ’MS’ for start of the month. Similarly, other
#options are also available for day (’D’), business days (’B’) and hours (’H’) etc.
 import pandas as pd
 import numpy as np
 rng1 = pd.date_range('2011-03-01 10:15', periods = 10, freq = 'M')
 rng2 = pd.date_range('2015 Jul 2 10:15', periods = 10, freq = 'M')
 # time series using start end 
 rng3 = pd.date_range(start = '2015 Jul 2 10:15', end = '2015 July 12', freq = '12H')
 # with time zone
 rng4 = pd.date_range(start = '2015 Jul 2 10:15', end = '2015 July 12', freq = '12H', tz='Asia/Kolkata')
 # check  type of date 
 type(rng[0]) #types of these dates are Timestamp
 
 ##Convert string to dates using ’to_datetime’ option
 dd = ['07/07/2015', '08/12/2015', '12/04/2015']
 d= list(pd.to_datetime(dd)) # american style
 type(d[0])
 
##Periods: Periods represents the time span e.g. days, years, quarter or month etc. Period class in pandas allows us to convert the frequency easily.
#when we use ’asfreq’ operation with ’start’ operation the date is ’01’ where as it is ’31’ with ’end’ option.
pr = pd.Period('2012', freq='M')
pr.asfreq('D', 'start')
Period('2012-01-01', 'D')
pr.asfreq('D', 'end')
Period('2012-01-31', 'D')

##Periodic Arithmetics: All the operations will be performed based on ’freq’,
pr = pd.Period('2012', freq='A') # Annual
pr + 1
# year to month converison
prMonth = pr.asfreq('M')
prMonth - 1

##Period range: range of periods can be created using ’period_range’ command,
prg = pd.period_range('2010', '2015', freq='A')
#create Series of random nunber with index taken from prg Above.
data = pd.Series(np.random.rand(len(prg)), index=prg)

##Converting string-dates to period:
#Its a two step process, i.e. first we need to convert the string to date format and then convert the dates in periods
dates = ['2013-02-02', '2012-02-02', '2013-02-02'] # dates as string INPUT.
d = pd.to_datetime(dates) # convert string to date format
prd = d.to_period(freq='M') # create period index from datetime index
prd.asfreq('D') # change frequency type with Date format
prd.asfreq('Y') # change frequency type with Year format

##Convert periods to timestamps: package from above
prd.to_timestamp()
prd.to_timestamp(how='end')

##Time Offsets: adding or subtracting time
pd.Timedelta('3 days') # generate time offset in days.
pd.Timedelta('3M')
pd.Timedelta('4 days 3M')
#adding timedelta to time
pd.Timestamp('9 July 2016 12:00') + pd.Timedelta('1 day 3 min')
# adding timedelta to complete range
rng + pd.Timedelta('1 day')

##INDEX data with TIME
dates = pd.date_range('2015-01-12', '2015-06-14', freq = 'M') # create time range
len(dates)
#using abov series as index for temperature
atemp = pd.Series([100.2, 98, 93, 98, 100], index=dates)
#using time index to access temperature
idx = atemp.index[3]
atemp[idx]

#make another temperature series ’stemp’ and create a DataFrame using ’stemp’ and ’atemp’
stemp = pd.Series([89, 98, 100, 88, 89], index=dates)
temps = pd.DataFrame({'Auckland':atemp, 'Delhi':stemp}) # create datra frame for two city temperature against date index
temps['Auckland'] # or temps.Auckland # temperature check for Auckland
temps['Diff'] = temps['Auckland'] - temps['Delhi'] # add new column having difference as value
del temps['Diff'] # delete column from data frame

##BAsic Application: some usage of time series with sample example
  import pandas as pd
  df = pd.read_csv('stocks.csv') # get data set from link shared on pandasgyide.pdf of source file
  d = df.date[0]
  type(d) # gives format of d, which is string & not date time here
 #to import ’date’ as time stamp, ’parse_dates’ option is used.
 df = pd.DataFrame.from_csv('stocks.csv', parse_dates=['date'])
 d = df.date[0]
 type(d)
 # using date column as index
 df = pd.DataFrame.from_csv('stocks.csv', parse_dates=['date'], index_col='date')
 del df['Unnamed: 0'] # removing column 0
 df.index.name # gives the name of index used in dataframe.
 
##SLICE operation works for Date INDEX only if the dates are in sorted order. If dates are not sorted then we need to
##sort them first by using sort_index() command i.e. stocks.sort_index()
 
####ReSampling
#Resampling is the conversion of time series from one frequency to another. If we convert higher frequency data
#to lower frequency, then it is known as DOWN-sampling; whereas if data is converted to low frequency to higher
#frequency, then it is called UP-sampling.

#we want to see the data at the end of each month only (not on daily basis):
stocks.ix[pd.date_range(stocks.index[0], stocks.index[-1], freq='M')]
stocks.ix[pd.date_range(stocks.index[0], stocks.index[-1], freq='BM')] # BM for Business Month
stocks.ix['1990-Mar-30'] ##confirm the entry on mentioned date

stocks.resample('BM').mean().head() # resample & finding the mean of each bin
stocks.resample('BM').size().head(3) # total no of rows in each bin
stocks.resample('BM').count().head(3) # count total number of rows in each bin for each column
ds = stocks.resample('BM').asfreq().head() # # display last resample value from each bin

##Unsampling: When we upsample the data, the values are filled by NaN; therefore we need to use ’fillna’ method to replace
#the NaN value with some other values
rs = ds.resample('B').asfreq() # blank values filled by NaN.
rs = ds.resample('B').asfreq().fillna(method='ffill') # firward fill the NaN.

## Data Plotting:
# plot the data of ’AA’ for complete range:
  import matplotlib.pyplot as plt
  stocks.AA.plot()
  <matplotlib.axes._subplots.AxesSubplot object at 0xa9c3060c>
  plt.show()
  
# plotting multiple usiing IX command:
  stocks.ix['1990':'1995', ['AA', 'IBM', 'MSFT', 'GE']].plot()
<matplotlib.axes._subplots.AxesSubplot object at 0xa9c2d2ac>
  plt.show()
  
## Moving WINDOWS Functions: ways to analyze the data over a sliding window .
# data of ’AA’ is plotted along with the mean value over a window of length 250.
   stocks.AA.plot()
  <matplotlib.axes._subplots.AxesSubplot object at 0xa9c5f4ec>
   stocks.AA.rolling(window=200,center=False).mean().plot()
<matplotlib.axes._subplots.AxesSubplot object at 0xa9c5f4ec>
   plt.show()
   
##### WORKING WITH TEXT DATA ######
# most importantly, these methods exclude missing/NA values automatically.   
s = pd.Series(['A', 'B', 'C', 'Aaba', 'Baca', np.nan, 'CABA', 'dog', 'cat'])  
s.str.lower()
s.str.upper()
s.str.len()

#stripping whitespaces as here
idx = pd.Index([' jack', 'jill ', ' jesse ', 'frank'])
idx.str.strip() # strips extra space of idx
idx.str.lstrip() 
idx.str.rstrip()

df = pd.DataFrame(randn(3, 2), columns=[' Column A ', ' Column B '],index=range(3))
df.columns.str.strip()
df.columns.str.lower()
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_') # remove space and place _
   
## Splitting and Replacing Strings
s2 = pd.Series(['a_b_c', 'c_d_e', np.nan, 'f_g_h'])
s2.str.split('_')   
s2.str.split('_').str.get(1) # accessing split elements with get or [] notation
s2.str.split('_').str[1]
s2.str.split('_', expand=True) # expand to entire dataframe
s2.str.split('_', expand=True, n=1) # limiting number of splits      
# rsplit is similar to split except it works in the reverse direction
s2.str.rsplit('_', expand=True, n=1)

s3 = pd.Series(['A', 'B', 'C', 'Aaba', 'Baca','', np.nan, 'CABA', 'dog', 'cat'])   
s3.str.replace('^.a', 'XX-XX ', case=False)  # pattern ending with a replaced by XX-XX
s3.str.replace('^.a|dog', 'XX-XX ', case=False) #pattern & dog value

# Indexing with .str
s = pd.Series(['A', 'B', 'C', 'Aaba', 'Baca', np.nan,'CABA', 'dog', 'cat'])
s.str[0] # returns oth index element of each row
s.str[1] # returns 1st index element of each row
   
## Extracting Substrings 
#READ 596/2215 if needed more details
#1. Extract first match in each subject (extract)
ps= pd.Series(['a1', 'b2', 'c3'])
ps.str.extract('([ab])(\d)', expand=False)
#2. Extract all match in each subject (extract)   
   
## Testing for Strings that Match or Contain a Pattern
pattern = r'[0-9][a-z]' # pattern with 0-9,a-z values COMBINATION
pd.Series(['1', '2', '3a', '3b', '03c']).str.contains(pattern) # retruns Boolean T/F if combination matched
pd.Series(['1', '2', '3a', '3b', '03c']).str.match(pattern)  
#The distinction between match and contains is strictness: match relies on strict re.match, while contains relies on re.search.
s4 = pd.Series(['A', 'B', 'C', 'Aaba', 'Baca', np.nan, 'CABA', 'dog', 'cat'])
s4.str.contains('A', na=False)

####OPTIONS AND SETTINGS ####
pd.options.display.max_rows # default 60 rows (top 30 & low 30) displayed
pd.options.display.max_rows = 999 # set 999 as max rows to display
pd.set_option("display.max_rows",101) # alternate way 1 of doing ABOVE set operation 
pd.set_option("max_r",102) # alternate way 2 of doing ABOVE
pd.get_option("display.max_rows") # displays max rows

## Frequently Used Options in pandas library
pd.set_option('display.max_rows', 999)
pd.set_option('precision', 5)

df = pd.DataFrame(np.random.randn(7,2))   # 7 rows 2 columns data frame
pd.set_option('max_rows', 5) # contains 7 but displays other mid as DOTS
df   
pd.reset_option('max_rows')   
# representation of dataframes to stretch across pages
df = pd.DataFrame(np.random.randn(5,10))  
pd.set_option('expand_frame_repr', True) #enables WRAPPING operation
pd.set_option('expand_frame_repr', False) # disables WRAPPING operation
pd.reset_option('expand_frame_repr') #resetting the WRAPPING option
# setting precision option
df = pd.DataFrame(np.random.randn(5,5))
pd.set_option('precision',10) # create random with 10+1 digits
# setting threshold 
pd.set_option('chop_threshold', 0)
pd.set_option('chop_threshold', .5) # setting min of 0.5 as threshold
pd.reset_option('chop_threshold')
# Setting header position
df = pd.DataFrame(np.random.randn(5,3)) 
pd.set_option('colheader_justify', 'right')   # makes column header jusified to RIGHT
pd.set_option('colheader_justify', 'left')    # makes column header jusified to RIGHT
pd.reset_option('colheader_justify')

##Number Formatting refer 612/2215 for more details if needed.
   
###INDEXING AND SELECTING DATA #### slice, dice, and generally get and set subsets of pandas objects
#using loc() to label based from 0 to nth value index
#Series: s.loc[indexer]
#Dataframe: df.loc[row_indexer,column_indexer]
dates = pd.date_range('1/1/2000', periods=8)
df = pd.DataFrame(np.random.randn(8, 4), index=dates, columns=['A', 'B', 'C','D'])
df[['B', 'A']] = df[['A', 'B']] # swapping column A & B values
df.loc[:,['B', 'A']] = df[['A', 'B']]
df.loc[:,['B', 'A']] = df[['A', 'B']].values
dfa = df.copy() # getting copy of df as dfa
dfa.A # getting column A of new dfa
dfa.A = list(range(len(dfa.index))) # sets column A with new value
dfa['A'] = list(range(len(dfa.index))) # similar as above

sa = pd.Series([15781,2276,322234],index=list('abc'))
sa.b #gets bth element of series sa
sa.a = 5 # sets a values as 5 in place of previous 15781

#Pandas doesn't allow Series to be assigned into nonexistent columns as shown BELOW
df = pd.DataFrame({'one': [1., 2., 3.]}) # dataframe with ONE as column
df.two = [4, 5, 6] # trying adding TWO as new column

##SLICING
L = range(10)
L[4:] #gives values after 4th Index
L[:4] # gives first 2 value of series sa.If we choose higher value, then display all rows element.
L[::2] # gives value of EVEN index like 0,2,4,6,8 etc
L[::-2] # gives value of EVEN index like 0,2,4,6,8 etc in REVERSE ORDER
L[::-1] # gives series in REVERSE ORDER
L[::N] # values at Nth index and similar logic
#Similarly it works for tuples, arrays, and strings: say s='acbd' etc sort of.

# Selection By Label
dfl = pd.DataFrame(np.random.randn(5,4), columns=list('ABCD'), index=pd.date_range('20130101',periods=5))
dfl.loc[2:3] #if we want to slice, but gives ERROR TypeError
dfl.loc['20130102':'20130104'] # passing date column as row index identifier
# On series
s1 = pd.Series(np.random.randn(6),index=list('abcdef')) 
s1.loc['d':] # gets value after index d
s1.loc[['d'],:] # gives error, works well for DATAFRAME
s1.loc[:'c'] # gets value prior to c
s1.loc['c':] = 0 # sets value at o after cth index
# On dataframe
df1 = pd.DataFrame(np.random.randn(6,4),index=list('abcdef'),columns=list('ABCD'))
df1.loc[['a', 'b', 'd'] :] # gives ibnvalid key. ATTENTION to syntax
df1.loc[['a', 'b', 'd'], :]  # gets the a,b,d index value
df1.loc['d':, 'A':'C'] # after dth element & column range A to C
df1.loc['a'] # gives all column vakue of ath row index
df1.loc['a':] # gives all row after ath and all columns
df1.loc['a'] > 0 # checks for ath row having T/F boolean
df1.loc[:, df1.loc['a'] > 0] # gets B as o/p and then print prior to this value.

# Slicing with labels   
#When using .loc with slices, if both the start and the stop labels are present in the index, then elements located
#between the two (including them) are returned:
s = pd.Series(list('abcde'), index=[0,3,2,5,4])   
s.loc[3:5] # range 3 to 5 is returned
s.sort_index() # sort the series in ORDER
s.sort_index().loc[1:6] # returns between 1 to 6th index in ORDER

# Selection By Position
s2 = pd.Series(np.random.randn(5), index=list(range(0,10,2)))
s2.iloc[:3]
s1.iloc[:3] = 0 # sets all value prior to 3 as ZERO
# for dataframe
df = pd.DataFrame(np.random.randn(6,4),index=list(range(0,12,2)),columns=list(range(0,8,2)))
df.iloc[:3]
df.iloc[[1, 3, 5], [1, 3]]
df.iloc[1:3, :]
df.iloc[:, 1:3]

#Out of range slice indexes are handled gracefully just as in Python/Numpy.
x = list('abcdef')
x[4:10] # gives e,f of list.
x[8:10] # returns null as we dont have value at 8,9,10 index of series.
# Note that using slices that go out of bounds can result in an empty axis
dfl = pd.DataFrame(np.random.randn(5,2), columns=list('AB'))
dfl.iloc[:, 2:3]
dfl.iloc[:, 1:3]
dfl.iloc[2:3]
dfl.iloc[:, 4] # gives indexing error
dfl.iloc[[4, 5, 6]] # gives indexing error

#### REINDEXING ####
s = pd.Series([1, 2, 3])
s.reindex([1, 2, 3]) #oth index is removed. 3rd index is presented with NaN value.
# to preserve the valid keys only
s.loc[s.index.intersection([1, 2, 3, 4])] # 3,4 does not matches & hence will nt apear in list o/p.

##Selecting Random Samples "with the sample() method
s = pd.Series([10,11,22,33,44,555])
s.sample() # without argument, it returns 1 row randomly
s.sample(n=3) # with argument, it returns N as o/p rows
s.sample(n=32) # gives larger sample error when out of index value is passed.
s.sample(frac=0.35) # returns fraction/ percentage of rows
# By default, sample will return each row at most once, but can be changed with replace()
s.sample(n=6, replace=False) # no value is repeated
s.sample(n=5, replace=True) #replace repeat is allowed
# considering the weights for above S series
example_weights = [0, 0, 0.2, 0.2, 0.2, 0.4] # assigning weight to index
s.sample(n=3, weights=example_weights) #giving 3 value accordin to assigned weight

# for DATAFRAME having weight for column
df2 = pd.DataFrame({'col1':[9,8,7,6], 'weight_column':[0.5, 0.4, 0.1, 0]})
df2.sample(n = 3, weights = 'weight_column') # get 3 value based on weight of column
# sample also allows users to sample columns instead of rows using the axis argument.
df3 = pd.DataFrame({'col1':[11,22,33], 'col2':[27,37,47]})
df3.sample(n=1, axis=1) # returns col1 as o/p, axis 1 is for column
df3.sample(n=2, axis=1) # rerurns first 2 column for o/p

## SEEDING concept to select same set of elements during RANDOM selection operation ##
df3.sample(n=2, random_state=2) # gives same set of o/p for every execution for RANDOM_STATE()
df4.sample(n=2) # gives different o/p in every execution

## perform ENLARGEMENT when setting a non-existent key for that axis.
#1. for series
se = pd.Series([11,21,31]) # has 3 elements
se[5] = 51# append 5 at index 5.
#2. for dataframes
dfi = pd.DataFrame(np.arange(6).reshape(3,2),columns=['A','B'])
dfi.loc[:,'C'] = dfi.loc[:,'A'] # appends column A as C in dataframe
dfi.loc[3] = 5 # specifically set5 as 3rd index of dataframe of new column C.

# Accessing using "at and iat methods,"
s.iat[1] # gives 1st index value, i refers to INTEGER index
s.at[1] # same as above

## Boolean Indexing ##
s = pd.Series(range(-3, 4))
s[s > 0] # picks s value greater than 0
s[(s < -1) | (s > 0.5)] # AND operation of two condition
s[~(s > 0)] # NOT condition, against s greater than 0
# llly for dataframe
df3[df3['col1']>30] # in col1, check for value greater than 30 & then publish entire row.

## List comprehensions and map method of Series##
df2 = pd.DataFrame({'a' : ['one', 'one', 'two', 'three', 'two', 'one', 'six'],'b' : ['x', 'y', 'y', 'x', 'y', 'x', 'x'],'c' : np.random.randn(7)})
criterion = df2['a'].map(lambda x: x.endswith('e')) # defining complex criteria using LAMBDA
# specify the target column (a) & condition in lambda selection area (x. startswith())
df2[criterion]
# using MULTIPLE criteria:
df2[criterion & (df2['b'] == 'y')]

## ISIN() method of Indexing ## returns a boolean vector
#1. for Series
s = pd.Series(np.arange(5), index=np.arange(5)[::-1], dtype='int64')
s.isin([2, 4, 6]) # retruns T/F for indexed value
s[s.isin([2, 4, 6])] # returns actual value for indexed value
#2. for dataframes
df = pd.DataFrame({'vals': [1, 2, 3, 4], 'ids': ['a', 'b', 'f', 'n'],'ids2': ['a', 'n', 'c', 'n']})
values = ['a', 'b', 1, 3]
df.isin(values) # retruns a dataframe having boolean True/ false for matches
# match certain values with certain columns as below.
values = {'ids': ['a', 'b'], 'vals': [1, 3]} # key is column & value is list of items
# Combine DataFrame’s isin with the any() and all() methods to quickly select subsets of your data
values = {'ids': ['a', 'b'], 'ids2': ['a', 'c'], 'vals': [1, 3]}
row_mask = df.isin(values).all(1)
df[row_mask]

###WHERE method & MASKING ###
# Selecting values from a Series with a boolean vector generally returns a subset of the data. To guarantee that selection
#output has the same shape as the original data, you can use the where method in Series and DataFrame.
s= np.array(['3','6','8','12'])
s[s > 7]
s.where(s > 0) # using WHERE clause
# llly for dataframe lets say df[df <0]
dfi.where(dfi < 03, -dfi) # NEGATES value having value less than 3 

##By default, where() returns a modified copy of the data. There is an optional parameter inplace() so that the original
#data can be modified without creating a copy:
df_orig = df.copy()
df_orig.where(df > 2 ) # creates new df with > 2 condition
df_orig.where(df > 2, inplace=True); # modifies the dataframe with inplace=true.
df_orig # modified data frame set

## QUERY method that allows selection using an expression.
df = pd.DataFrame(np.random.rand(10, 3), columns=list('abc'))
# with pure pyhton
df[(df.a < df.b) & (df.b < df.c)]
#with query()
df.query('(a < b) & (b < c)') # way of writing 1
df.query('a < b & b < c')  # way of writing 2
df.query('a < b < c') #  # way of writing 3
#Muilti-Index query() for MULTI-INDEX dataframe
df.query('color == "red"') # where column 1 is color & column 2nd tier is red.
# If color index is missing, ilevel_0, which means “index level 0” for the 0th level of the index.
df.query('ilevel_0 == "red"')

## The in and not in operators
df = pd.DataFrame({'col1': list('aabbccddeeff'), 'col2': list('aaaabbbbcccc'),'col3': np.random.randint(5, size=12),'col4': np.random.randint(9, size=12)})
df.query('col3 in col2') # checks col3,2 for common value
df[df.col1.isin(df.col2)] # above query but using pure python
df.query('col1 not in col2')
df[~df.col1.isin(df.col2)] # above query but using pure python
df.query('col1 in col2 and col3 < col4')
df[df.col2.isin(df.col1) & (df.col3 < df.col4)] # pure python

## Special use of the == operator with list objects
df.query('col2 == ["a", "b", "c"]') # using same above df
df.query('col3 == ["0", "1"]') # having 0 & 1 for col3 in df
df[df.col2.isin(["a", "b", "c"])] # pure python
df.query('[1, 2] in col3') # using the in operator
df.query('[1, 2] not in c') # using not in operator

# Boolean opeartors
df1 = pd.DataFrame(np.random.rand(3, 3), columns=list('abc'))
df1['bools'] = np.random.rand(len(df1)) > 0.5
df1.query('~bools')
df.query('not bools') # same as above
## NOTE: DataFrame.query() using numexpr is slightly faster than Python for large frames say 20k+ rows 

## DUPLICATES using duplicated and drop_duplicates() methods
#.By default, the first observed row of a duplicate set is considered unique,
#• keep='first' (default): mark / drop duplicates except for the first occurrence.
#• keep='last': mark / drop duplicates except for the last occurrence.
#• keep=False: mark / drop all duplicates.
df2 = pd.DataFrame({'A': ['one', 'one', 'two', 'two', 'two', 'three', 'four'],
'B': ['x', 'y', 'x', 'y', 'x', 'x', 'x'],'C': np.random.randn(7)})
df2.duplicated('A') # checks duplicate across column A
df2.duplicated('A', keep='last')
df2.duplicated('A', keep=False)
df2.drop_duplicates('A') # drops duplicates from A
df2.drop_duplicates('A', keep='last') # drop duplicates by keeping last
df2.drop_duplicates('A', keep=False)

# pass a list of columns to identify duplications
df2.duplicated(['A', 'B'])
df2.drop_duplicates(['A', 'B'])


## LOOKUP method ####
df3 = pd.DataFrame(np.random.rand(20,4), columns = ['A','B','C','D'])
df3.lookup(list(range(0,10,2)), ['B','C','A','B','D'])

## Index class and its subclasses can be viewed as implementing an ordered multiset.
index = pd.Index(['e', 'd', 'a', 'b'])
'd' in index # returns Boolean True
index = pd.Index(['e', 'd', 'a', 'b'], name='something') # definign name for Index
index.name

#defining df using row & col index
index = pd.Index(list(range(5)), name='rows')
columns = pd.Index(['A', 'B', 'C'], name='cols')
df4 = pd.DataFrame(np.random.randn(5, 3), index=index, columns=columns)
df4['A']

## MISSING VALUES ###
# Index.fillna fills missing values with specified scalar value
idx1 = pd.Index([1, np.nan, 3, 4])
idx1.fillna(2) # sets 2 as value at NaN valued index

## Set / Reset Index ##
# SET index
df2 = pd.DataFrame({'A': ['one', 'one', 'two', 'two', 'two', 'three', 'four'],
'B': ['x', 'y', 'x', 'y', 'x', 'x', 'x'],'C': np.random.randn(7)})
df2.set_index('A') # sets col A as index for df
df2.set_index(['A', 'B']) # sets multi-level index from A&B.

# reset_index which transfers the index values into the DataFrame’s columns and sets a simple integer indexdf2.reset_index()
df2.reset_index()
df2.reset_index(level=0) # level keyword to remove only a portion of the index

### VIEW vs COPY ###
# multi-index dataframe
dfmi = pd.DataFrame([list('abcd'),list('efgh'),list('ijkl'),list('mnop')],columns=pd.MultiIndex.from_product([['one','two'],['first','second']]))
# accessing particlur column
dfmi['one']['second'] # method one: Chained Indexing
dfmi.loc[:,('one','second')] # method two using loc()
# method 2 can be significantly faster, and allows one to index both axes if so desired

####********** ADVANCED INDEXING pg 668 - 702/ 2215 pandas.pdf ************ ##########

## Statistical Functions ##
#Percent Change: pct_change to compute the percent change over a given number of periods
# for series
ser = pd.Series(np.random.randn(8))
ser.pct_change() # percntge change from one to next value in list
# for data frame
df = pd.DataFrame(np.random.randn(10, 4))
df.pct_change(periods=4) # till 4th index row, it will NaN & then % chnage in value down the coulmn

# Covariance: compute covariance between series (excluding NA/null values)
# It is measure of how much two random variables vary together
s1 = pd.Series(np.random.randn(1000))
s2 = pd.Series(np.random.randn(1000))
s1.cov(s2)

frame = pd.DataFrame(np.random.randn(1000, 5), columns=['a', 'b', 'c', 'd', 'e'])
frame.cov() # gives covariane having rows= number of columns of dataframe by default

#using min_period key word
frame = pd.DataFrame(np.random.randn(20, 3), columns=['a', 'b', 'c'])
frame.loc[frame.index[:5], 'a'] = np.nan # setting NaN value
frame.loc[frame.index[5:10], 'b'] = np.nan # setting NaN value
frame.cov()
frame.cov(min_periods=2) # specifies the required minimum number of observations for each column pair in order to have a valid result.


## CORRELATION ### relationship or connection between two or more things.
#1. pearson (default): Standard correlation coefficient
#2. kendall: Kendall Tau correlation coefficient
#3. spearman: Spearman rank correlation coefficient
frame = pd.DataFrame(np.random.randn(1000, 5), columns=['a', 'b', 'c', 'd','e'])
frame.iloc[::2] = np.nan
frame['a'].corr(frame['b']) # series by series correlation
frame.corr() # pairwise corrleation
## NOTE: Note that non-numeric columns will be automatically excluded from the correlation calculation.

frame.loc[frame.index[:5], 'a'] = np.nan
frame.loc[frame.index[5:10], 'b'] = np.nan
frame.corr()
frame.corr(min_periods=12)

# CORRWITH method
index = ['a', 'b', 'c', 'd', 'e']
columns = ['one', 'two', 'three', 'four']
df1 = pd.DataFrame(np.random.randn(5, 4), index=index, columns=columns)
df2 = pd.DataFrame(np.random.randn(4, 4), index=index[:4], columns=columns)
df1.corrwith(df2)
df2.corrwith(df1, axis=1)

## DATA RANKING ##
# rank method produces a data ranking with ties being assigned the mean of the ranks (by default) for the group
s = pd.Series(np.random.np.random.randn(5), index=list('abcde'))
s['d'] = s['b'] # so there's a tie
s.rank() # gives rank/ order of series element.

df = pd.DataFrame(np.random.np.random.randn(10, 6))
df[4] = df[2][:5] # sets column 4 with initial 5 values of d[2][:5] and rest as NaN.
# df[2][:5] gives first 5 row of column 2.
df.rank(1)
# rank optionally takes a parameter ascending which by default is true;

## Window Functions ## 
# count, sum, mean, median, correlation, variance, covariance, standard deviation, skewness, and kurtosis
s = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2000',periods=1000))
s = s.cumsum()
r = s.rolling(window=60)
r.mean()
s.plot(style='k--')
r.mean().plot(style='k')

# refer 707-730 for further details#

###  MISSING DATA #### the isna() and notna() methods
#By “missing” we simply mean NA (“not available”) or “not present for whatever reason”.
df = pd.DataFrame(np.random.randn(5, 3), index=['a', 'c', 'e', 'f', 'h'],columns=['one', 'two', 'three'])
df['four'] = 'bar' # sets all value to bar for column four.
df['five'] = df['one'] > 0 # APPENDS 'five' as new Column having value from ONE.
df2 = df.reindex(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']) # generates df2 basd on df having defined index. 
# puts NaN for cases where indexes are Missing...
# Note: If you want to consider inf and -inf to be “NA” in computations, you can set pandas.options.mode.
#use_inf_as_na = True.
df2['one']
pd.isna(df2['one']) # syntax for IS NA
df2['four'].notna() # syntax for NOT NA
df2.isna() # gets is NA for entire dataframe df2

#Warning: One has to be mindful that in python (and numpy), the nan's don’t compare equal, but None's do
None == None # gives TRUE
np.nan == np.nan # gives FALSE
#scalar equality comparison versus a None/np.nan doesn’t provide usefu
df2['one'] == np.nan # retruns all FALSE only

## Missing Datetimes # NaT represents missing values
df = pd.DataFrame(np.random.randn(5, 3), index=['a', 'c', 'e', 'f', 'h'],columns=['one', 'two', 'three'])
df2 = df.copy()
df2['timestamp'] = pd.Timestamp('20120101')
df2.loc[['a','c','h'],['one','timestamp']] = np.nan # setting certainindex as NaN & NaT respectively
df2.get_dtype_counts()

 ## Inserting MISSING data ###
 #You can insert missing values by simply assigning to containers. The actual missing value used will be chosen based on the dtype.
# For example, numeric containers will always use NaN regardless of the missing value type chosen.
s = pd.Series([1, 2, 3]) # NUMERIC CONTAINER
s.loc[0] = None # will always have NaN in resulting series
# datetime containers will always use NaT.
s1 = pd.Series(["a", "b", "c"]) # OBJECT CONTAINER
s1.loc[0] = None # will be reflected as None
s1.loc[1] = np.nan # will be reflected as NaN

##Calculations with missing data ##
element by element calculation is performed
s2 = pd.Series([3, 5, 31]) 
s+s2 # gives addition index by index
s3 = pd.Series([3, 5, 31,999]) #additional 3rdindex value
s+s3 # 3 value +4th as NaN coz of missing in s series; for Series only
#TIPS: for dataframe ojject When summing data, NA (missing) values will be treated as zero if we explicitly write NAN in S

##Sum/Prod of Empties/Nans
#With sum or prod on an empty or all-NaN Series, or columns of a DataFrame, the result will be all-NaN.
s = pd.Series([np.nan])
s.sum() # sum of nan series
pd.Series([]).sum() # sum of empty series
## Warning: These behaviors differ from the default in numpy where an empty sum returns zero.
np.nansum(np.array([]))
np.nansum(np.array([np.nan]))
##NA groups in GroupBy are automatically excluded

####### CLEANING & FILLING MISSING DATA ######
## Filling missing values: fillna
df = pd.DataFrame(np.random.randn(5, 3), index=['a', 'c', 'e', 'f', 'h'],columns=['one', 'two', 'three'])
df2 = df.copy()
df2['timestamp'] = pd.Timestamp('20120101')
df2.loc[['a','c','h'],['one','timestamp']] = np.nan
#df2 has missing data along with time stamp
df2.fillna(0) # fills NaN with O, or other value we choose say 89999 etc.
df2['one'].fillna('missing') # replaces 'NaN' with 'missing' in column one of df2
df2['timestamp'].fillna('missing') 

##Fill gaps forward or backward with last accessed value
# pad,ffill: fills values in forward direction
#bfill, backfill: fills values in backward direction
# Again using df2 as generated above for NaN.
df2.fillna(method='pad') # fills NaN,NaT using previous value
df2.fillna(method='pad', limit=1) # fills value upto 1st occurnce. 2nd ocurence is left as it is.

## Filling with a PandasObject that in ALIGNABLE
dff = pd.DataFrame(np.random.randn(10,3), columns=list('ABC'))
dff.iloc[3:5,0] = np.nan # filss 3,5 rows of column index 0 with NaN.
dff.iloc[4:6,1] = np.nan
dff.iloc[5:8,2] = np.nan
dff.mean() # gives mean Column wise for ABC
dff.mean()['B':'C'] # gives means of column B & C only
dff.fillna(dff.mean()) # fill nan with mean for respective column
dff.where(pd.notna(dff), dff.mean(), axis='columns') # same result set as ABOVE
dff.fillna(dff.mean()['B':'C']) #fils nan for B,C column

## Dropping axis labels with missing data: dropna
# using above df2 having NaN & NaT 
df2.dropna(axis=0) # removes any row having NAN,NAT coz axis is 0 ROWS
df2.dropna(axis=1) # removes any column having NAN,NAT coz axis is 1 COLUMNS
df2['one'].dropna() # for column One drops Nan,NaT & gives remaining values as o/p
df2.dropna() # for every column drops Nan,NaT & gives remaining values as o/p for FULL ROW&COLUMN

### Interpolation ####
# Both Series and DataFrame objects have an interpolate method.
#By default, both performs linear interpolation at missing datapoints.
idf = pd.DataFrame(np.random.randn(60, 2), index=pd.date_range('1-1-2000', periods=60), columns=list('XY'))
idf.iloc[2:7,1]=np.nan # setting few from column 1 as nan for interpolation purpose
idf.count() # gives count of non-zero along both the columns X,Y
idf.interpolate().count() # X,Y can be interpolated to 60 each here
idf.interpolate() # fills appx value for missing cases
idf.interpolate(limit=2) # fills only 2 value on interpolation in default FORWARD direction
idf.interpolate(limit=2,limit_direction='backward') # BACKWARD interpolate with limit as 2
idf.interpolate(limit=2, limit_direction='both') # both direction
idf.interpolate().plot() # plots with appx interpolate value for missing timeseries values
# for dataframe interpolation
df = pd.DataFrame({'A': [1, 2.1, np.nan, 4.7, 5.6, 6.8],'B': [.25, np.nan, np.nan, 4, 12.2, 14.4]})
df.interpolate() # inserts some appx value for missing 

## Replacing Generic Values ##
#1. for series
ser = pd.Series([0., 4., 22., 33., 44.])
ser.replace(2, 5) # finds 2 & replace with 5, if not found then no action is expected
ser.replace([0, 1, 2, 3, 4], [4, 3, 2, 1, 0]) # replaceing with ARRAY find & replace
ser.replace([0,1,2,3,4], method='pad') # whichever it finds, pad them with filling methods
ser.replace([0,4, 22], method='pad') # replace 0 with PAD method forward.
#basically whatever value if finds, it replaces with coresponding element frm second array
#2. for dataframes
df = pd.DataFrame({'a': [0, 1, 2, 3, 4], 'b': [5, 6, 7, 8, 9]})
df.replace({'a': 0, 'b': 5}, 100) # will replace 0,5 as 100 from a,b column of df
df.replace({'a': [0, 1, 2, 3, 4]}, 45678)
df.replace({'a': [0, 1, 2, 3, 4]}, method='pad') # will GIVE ERROR HERE & cant be used.

####String/Regular Expression Replacement
#Python strings prefixed with the r character such as r'hello world' are so-called “raw” strings.
d = {'A': list(range(4)), 'B': list('ab..'), 'C': ['a', 'b', np.nan, 'd']} # tuple create
df = pd.DataFrame(d) # creating df from tuple
df.replace('.', np.nan)
df.replace(r'\s*\.\s*', np.nan, regex=True) #regular expression that removes surrounding whitespace
df.replace(['a', '.'], ['pp', 'dot']) # finds a & . then replace with pp & dot respectively
df.replace({'B': ['.','a']}, {'B': ['np.nan','AA']})# dictionary appraoch to REPLACE multiple values

##Numeric Replacements
df = pd.DataFrame(np.random.randn(10, 2))
df[np.random.rand(df.shape[0]) > 0.5] = 1.5 # finds value gerater 0.5 & set value as 1.5
df[df[0]> 0.5]=233 # sets value as 233 for condition mentioned
df.replace(233, np.nan) #alternatively for above case
#replacing using LIST
df00 = df.values[0, 0]
df.replace([233, df00], ['322', 'a'])
df[1].dtype #checking datatype post replacement operation

######    GROUP BY: SPLIT-APPLY-COMBINE    #######################
#By “group by” we are referring to a process involving one or more of the following steps:
#• Splitting the data into groups based on some criteria
#• Applying a function to each group independently
#• Combining the results into a data structure

# Additionally the Applying may need to use functions such as   
# Aggregation: computing a summary statistic (or statistics) about each group. eg. Compute group sums or means, group sizes / counts.
# Transformation: perform some group-specific computations and return a like-indexed.e.gStandardizing data (zscore) within group,Filling NAs within groups etc.
# Filtration: discard some groups, according to a group-wise computation eg. Filling NAs within groups, filtering data based on xyz condition.

# ****** Sequence of operation from SQL [""WGHOL"" to remember] ***************
#1. FROM & JOINs determine & filter rows
#2. WHERE more filters on the rows
#3. GROUP BY combines those rows into groups
#4. HAVING filters groups
#5. ORDER BY arranges the remaining rows/groups
#6. LIMIT filters on the remaining rows/groups

## Splitting an object into groups, DEFAULT axis is row  axis=0
#>>> grouped = obj.groupby(key)
#>>> grouped = obj.groupby(key, axis=1)
#>>> grouped = obj.groupby([key1, key2])

#A string passed to groupby may refer to either a column or an index level. 
#If a string matches both a column name and an index level name then a warning is issued and the column takes precedence.
df = pd.DataFrame({'A' : ['foo', 'bar', 'foo','bar','foo', 'bar', 'foo','foo'],'B' : ['one', 'one', 'two','three','two', 'two', 'one','three'],'C' : np.random.randn(8),'D' : np.random.randn(8)})
grouped = df.groupby('A') # grouped by column
grouped = df.groupby(['A', 'B'])
grouped.groups #returns index & data types
#using user defined function
def get_letter_type(letter):
    if letter.lower() in 'aeiou':
        return 'vowel'
    else:
        return 'consonant'
grouped = df.groupby(get_letter_type, axis=1)
grouped.groups
len(grouped)
## Non-unique value handling. If a non-unique index is used as the group key in a groupby operation,
#all values for the same index value will be considered to be in one group and thus the output of aggregation functions
#will only contain unique index values:
lst = [1, 2, 3, 1, 2, 3]
s = pd.Series([1, 2, 3, 10, 20, 30], lst)
grouped = s.groupby(level=0)
grouped.first()
grouped.last()
grouped.sum()
# Note that no splitting occurs until it’s needed.

## GroupBy sorting: By default the group keys are sorted during the groupby operation., else pass sort=False
dfgs = pd.DataFrame({'X' : ['B', 'B', 'A', 'A','A'], 'Y' : [2, 2, 32, 41, 22]})
dfgs.groupby(['X']).sum() # grouped along X column and SUm agggregated for row element
dfgs.groupby(['X'], sort=False).sum()
dfgs.groupby(['X']).get_group('A') # group along X column & then get list by its element
dfgs.groupby(['Y']).get_group(2) # group along Y coulmn & then get list by its element
dfgs.groupby('X').groups

## GroupBy with MultiIndex
arrays = [['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']]
index = pd.MultiIndex.from_arrays(arrays, names=['first', 'second'])
# Series using above array
sarray = pd.Series(np.random.randn(8), index=index)

#grouping operation
sarray.groupby(level='first').sum() # can use level=0 for same result: Dont confuse with AXIS
sarray.groupby(['first']).sum() # new in 0.20 , has same result as ABOVE. kEy passing
sarray.groupby(level='second').sum()# can use level=1 for same result: Dont confuse with AXIS
sarray.groupby(['second']).sum() # new in 0.20 , has same result as ABOVE. kEy passing
sarray.groupby(['first', 'second']).sum()
sarray.sum(level='second')
## ceate further level : sm.groupby(level=['first', 'second']).sum()

# data frame
dfarray = pd.DataFrame({'A': [1, 1, 1, 1, 2, 2, 3, 3],'B': np.arange(8)},index=index)
## DataFrame column selection in GroupBy ##
grouped = dfarray.groupby(['A'])
grouped_C = grouped['A'].groups

# Iterating through groups
grouped = dfarray.groupby(['A'])
for name, group in grouped:
    print(name)
    print(group)
    
## Selecting a group from Grouped section
grouped.get_group(1)    
dfarray.groupby(['A', 'B']).get_group((1,2))

### Transformation #### returns an object that is indexed the same (same size) as the one being grouped
#Refer 770-780 0f 2215 documentation pandas.pdf

## NA and NaT group handling: 
#If there are any NaN or NaT values in the grouping key, these will be automatically excluded. So there will never be
#an “NA group” or “NaT group”.

## Grouping with ordered factors
data = pd.Series(np.random.randn(100))
factor = pd.qcut(data, [0, .25, .5, .75, 1.])
data.groupby(factor).mean()

##Taking the first rows of each group
df = pd.DataFrame([[1, 2], [1, 4], [5, 6]], columns=['A', 'B'])
g = df.groupby('A')
g.head(1)
g.tail(1)
##Taking the nth row of each group
g.nth(0) # nth(0) is the same as g.first()
g.nth(-1) # nth(-1) is the same as g.last()
g.nth(1)
#passing as_index=False, will achieve a filtration, which returns the grouped row.
g = df.groupby('A',as_index=False)
g.nth(0)

## Enumerate group items
# To see the order in which each row appears within its group, use the cumcount method:
dfg = pd.DataFrame(list('aaabba'), columns=['A'])
dfg.groupby('A').cumcount()
dfg.groupby('A').cumcount(ascending=False)
# To see the ordering of the groups
dfg.groupby('A').ngroup()
dfg.groupby('A').ngroup(ascending=False)

##### MERGE, JOIN, AND CONCATENATE  #########################
df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2'],'B': ['B0', 'B1', 'B2'],'C': ['C0', 'C1', 'C2'],'D': ['D0', 'D1', 'D2']},index=[0, 1, 2])
df2 = pd.DataFrame({'A': ['A4', 'A5', 'A6'],'B': ['B4', 'B5', 'B6'],'C': ['C4', 'C5', 'C6'],'D': ['D4', 'D5', 'D6']},index=[33 ,43, 55])
frames = [df1, df2]
result = pd.concat(frames)
result = pd.concat(frames, keys=['df1key', 'df2key']) # assigning keys x,y corresponding to df1, df2
result.loc['df2key'] # accessing keyed value against df2

# Concatenate along other axes: say Column Joining . Earlier it was along ROW
df3 = pd.DataFrame({'B': ['B2', 'B3', 'B6'],'D': ['D2', 'D3', 'D6'],'F': ['F2', 'F3', 'F6']},index=[2, 3, 66])
result = pd.concat([df1, df3], axis=1)
#check concetnete on different column
pd.concat([df1,df3]) # preserves all column from both data frames, but doesnot DUPLICATE
# inner join on top of df's UNION.
resultj = pd.concat([df1, df3], axis=1, join='inner') # column gets duplicated
# join based on index of any dataframe
resultk = pd.concat([df1, df3], axis=1, join_axes=[df1.index]) # column gets duplicated

### Concatenating using append ##
#shortcut to concat are the APPEND instance methods on Series and DataFrame.
# They concatenate along axis=0
# Append here does not modify df1 and returns its copy with df2 appended.
resultl = df1.append(df3) # duplicate index posible along side multiple columns
# appending multiple df
resultm = df1.append([df2, df3])
## Ignoring indexes on the concatenation axis
resulti = pd.concat([df1, df2], ignore_index=True) # original index are removed; 0 to n index created
resultia = pd.concat([df1, df2, df3], ignore_index=True) # original index are removed; 0 to n index created
resultiaa = df1.append(df3, ignore_index=True)

## concatenate a mix of Series and DataFrames
s1 = pd.Series(['X0', 'X1', 'X2', 'X3'], name='X') # names series used for appending
resultr = pd.concat([df1, s1], axis=0) # append each element as new ROW to df1 
resulta = pd.concat([df1, s1], axis=1) # append as new column to df1
# passing Unnames series
s2 = pd.Series(['_0', '_1', '_2', '_3'])
resultru = pd.concat([df3, s2], axis=0) # append each element as new ROW to df1 
resultau = pd.concat([df3, s2], axis=1)
# Passing ignore_index=True will drop all name references.
resultrui = pd.concat([df3, s2], axis=0, ignore_index=True) # append each element as new ROW to df1 & drop Index name
resultaui = pd.concat([df3, s2], axis=1, ignore_index=True) # append & drop column name

## concatenating with group keys, override the column names when creating a new DataFrame based on existing Series
s3 = pd.Series([0, 1, 2, 3], name='foo')
s4 = pd.Series([0, 1, 2, 3])
s5 = pd.Series([0, 1, 4, 5], name='bar')
pd.concat([s3, s4, s5], axis=1) # no overideing
pd.concat([s3, s4, s5], axis=1, keys=['red','blue','yellow']) # overriding column name
### identfiying parent df after concat/ appending operation 
#1. trying group keys on df frames
pd.concat(frames, keys=['x', 'y']) # identifies explicityly the df1,df2 in concat operation
#2. using dictionary approach to concat
pieces = {'x': df1, 'y': df2, 'z': df3}
pd.concat(pieces) # identifies explicityly the df1,df2 in concat operation

## Appending ROWS to a DataFrame ##
s6 = pd.Series(['X0', 'X1', 'X2', 'X3'], index=['A', 'B', 'C', 'D'])
df1.append(s6, ignore_index=True) # cant use axix argument like we used for Concat
df1.append(s6,ignore_index=False ) #an only append a Series if ignore_index=True or if the Series has a name
# With list of dicts or Series:
dicts = [{'A': 1, 'B': 2, 'C': 3, 'X': 4},{'A': 5, 'B': 6, 'C': 7, 'Y': 8}]
df1.append(dicts, ignore_index=True)

### Database-style DataFrame joining/merging ###
#pandas provides a single function, MERGE, as the entry point for all standard database join operations between DataFrame objects:
#1 • one-to-one joins: for example when joining two DataFrame objects on their indexes (which must contain unique values)
#2 • many-to-one joins: for example when joining an index (unique) to one or more columns in a DataFrame
#3 • many-to-many joins: joining columns on columns.
## if a key combination appears more than once in both tables, the resulting table will have the Cartesian product of the associated data.
left = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],'A': ['A0', 'A1', 'A2', 'A3'],'B': ['B0', 'B1', 'B2', 'B3']})
right = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],'C': ['C0', 'C1', 'C2', 'C3'],'D': ['D0', 'D1', 'D2', 'D3']})
result = pd.merge(left, right, on='key') # common key
right1 = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3','K4'],'C': ['C0', 'C1', 'C2', 'C3','C4'],'D': ['D0', 'D1', 'D2', 'D3','D4']})
result1 = pd.merge(left, right1, on='key') # only common KEY is indexed, other removed.
# Multiple Key Joins
left2 = pd.DataFrame({'key1': ['K0', 'K0', 'K1', 'K2'],'key2': ['K0', 'K1', 'K0', 'K1'],'A': ['A0', 'A1', 'A2', 'A3'],'B': ['B0', 'B1', 'B2', 'B3']})
right2 = pd.DataFrame({'key1': ['K0', 'K0', 'K1', 'K2'],'key2': ['K0', 'K1', 'K0', 'K1'],'C': ['C0', 'C1', 'C2', 'C3'],'D': ['D0', 'D1', 'D2', 'D3']})
result2 = pd.merge(left2, right2, on=['key1', 'key2'])
# If a key combination does not appear in either the left or right tables, the values in the joined table will be NA
#1.left: LEFT OUTER JOIN
#2.right: RIGHT OUTER JOIN
#3.outer: FULL OUTER JOIN
#4. inner: INNER JOIN
resultleft = pd.merge(left2, right2, how='left', on=['key1', 'key2'])
resultright = pd.merge(left2, right2, how='right', on=['key1', 'key2'])
resultouter = pd.merge(left2, right2, how='outer', on=['key1', 'key2'])
resultinner = pd.merge(left2, right2, how='inner', on=['key1', 'key2'])
resultinner.dtypes # returns the datatypes which is PRESERVED.

# Checking for DUPLICATE keys
#Key uniqueness is checked before merge operations and so should protect against memory overflows
leftu = pd.DataFrame({'A' : [1,2], 'B' : [1, 2]})
rightu = pd.DataFrame({'A' : [4,5,6], 'B': [2, 2, 2]})
resultu = pd.merge(leftu, rightu, on='B', how='outer', validate="one_to_one") # not a one-to-one merge

# If the user is aware of the duplicates in the right DataFrame but wants to ensure there are no duplicates in the left
#DataFrame, one can use the validate=’one_to_many’ argument instead, which will not raise an exception.
pd.merge(leftu, rightu, on='B', how='outer', validate="one_to_many")

## MERGE INDICATOR: Categorical-type column called _merge will be added to the output object that takes on values:
pd.merge(leftu, rightu, on='B', how='outer', indicator=True)
pd.merge(leftu, rightu, on='B', how='outer', indicator='AmtSrimal') # same as above but merge coulmn named as AmtSrimal

## Joining on index ## 
left = pd.DataFrame({'A': ['A0', 'A1', 'A2'],'B': ['B0', 'B1', 'B2']},index=['K0', 'K1', 'K2'])
right = pd.DataFrame({'C': ['C0', 'C2', 'C3'],'D': ['D0', 'D2', 'D3']},index=['K0', 'K2', 'K3'])
result1 = left.join(right) # has index of left dataframe only
result2 = left.join(right, how='outer') # all index of both dataframes
result3 = left.join(right, how='inner') # only common index of both dataframes

## Joining key columns on an index ## on argument which may be a column or multiple column names
left1 = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],'B': ['B0', 'B1', 'B2', 'B3'],'key': ['K0', 'K1', 'K0', 'K1']})
right1 = pd.DataFrame({'C': ['C0', 'C1'],'D': ['D0', 'D1']},index=['K0', 'K1'])
result1 = left1.join(right1, on='key')

## Joining a single Index to a Multi-index 817 /2215 EXPLORE if needed.

## Overlapping value columns ##
left = pd.DataFrame({'k': ['K0', 'K1', 'K2'], 'v': [1, 2, 3]})
right = pd.DataFrame({'k': ['K0', 'K0', 'K3'], 'v': [4, 5, 6]})
pd.merge(left, right, on='k')
pd.merge(left, right, on='k', suffixes=['_l##', '_r##'])
left.join(right, lsuffix='_l**', rsuffix='_r**')                                       

## Merging Ordered Data: allows combining time series and other ordered data
# optional fill_method keyword to fill/interpolate missing data
lefto = pd.DataFrame({'k': ['K0', 'K1', 'K1', 'K2'],'lv': [1, 2, 3, 4],'s': ['a', 'b', 'c', 'd']})
righto = pd.DataFrame({'k': ['K1', 'K2', 'K4'],'rv': [1, 2, 3]})
pd.merge_ordered(lefto, righto, fill_method='ffill', left_by='s')

###   RESHAPING   ####### 
# Reshaping by stacking and unstacking
# stack: “pivot” a level of the (possibly hierarchical) column labels, returning a DataFrame with an index with a new inner-most level of row labels.
# unstack: inverse operation from stack: “pivot” a level of the (possibly hierarchical) row index to the column axis, producing a reshaped DataFrame with a new inner-most level of column labels.
tuples = list(zip(*[['bar','bar','baz','baz',
                     'foo','foo','qux','qux'],
                    ['one','two','one','two',
                     'one','two','one','two']]))
index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])
df = pd.DataFrame(np.random.randn(8, 2), index=index, columns=['A', 'B'])
df2 = df[:4]
df2
# The stack function “compresses” a level in the DataFrame’s columns to produce either:
    #• A Series, in the case of a simple column Index
    #• A DataFrame, in the case of a MultiIndex in the columns
stacked = df2.stack() # o/p type is series here
# unstacking the stacked data
unstacked = stacked.unstack() # o/p is dataframe
stacked.unstack(0)
stacked.unstack(1)
# refer 828-831 for further readings detail

## Reshaping by Melt ##
# melt() functions are useful to massage a DataFrame into a format where one or more
#columns are identifier variables, while all other columns, considered measured variables, are “unpivoted” to the row
#axis, leaving just two non-identifier columns, “variable” and “value”.
cheese = pd.DataFrame({'first' : ['John', 'Mary'],'last' : ['Doe', 'Bo'], 'height' : [5.5, 6.0],'weight' : [130, 150]})
cheese.melt(id_vars=['first', 'last']) # 2 row converted to 4, with variable & value set
cheese.melt(id_vars=['first', 'last'], var_name='quantity') # rename the variable to quantity

### TIME SERIES / DATE FUNCTIONALITY ###
rng = pd.date_range('1/1/2011', periods=72, freq='H')
rng[:5]
# index pandas with date
ts = pd.Series(np.random.randn(len(rng)), index=rng)
ts.head()
# Change frequency and fill gaps:
converted = ts.asfreq('45Min', method='pad') # frward fill with 45 mins

## Timestamps vs. Time Spans ##
# Timestamped data is the most basic type of time series data that associates values with points in time.
pd.Timestamp(datetime(2012, 5, 1))
pd.Timestamp('2012-05-01')
pd.Period('2011-01')


## ...........to be continued from 845
#**************************************************************************************************
ss=pd.read_excel('/home/akmsrimal/Downloads/Sample - Superstore.xls', index_col=None)
ss.head()

























