# CHAPTER1:_____________________________________________________________________
### Why NOT Python:
# Reason1: As Python is an interpreted programming language, in general most Python code will
# run substantially slower than code written in a compiled language like Java or C++.
# As programmer time is often more valuable than CPU time, many are happy to make
# this trade-off.
# Reason2: Python can be a challenging language for building highly concurrent, multithreaded
# applications, particularly applications with many CPU-bound threads. The reason for
# this is that it has what is known as the global interpreter lock (GIL), a mechanism that
# prevents the interpreter from executing more than one Python instruction at a time.

### WHY Python is popular:
# Python has a huge collection of libraries.
# Python is known as the beginner’s level programming language because of it simplicity and easiness.
# From developing to deploying and maintaining Python wants their developers to be more productive.
# Portability is another reason for huge popularity of Python.
# Python’s programming syntax is simple to learn and is of high level compared to C, Java, and C++.
# new applications can be developed by writing fewer lines of codes, the simplicity of Python has attracted many developers.

###TOP few python Libraries:
# TensorFlow
# Scikit-Learn
# Numpy
# Keras
# PyTorch
# LightGBM
# Eli5
# SciPy
# Theano
# Pandas

## Important packages to learn:
# 1.Numpy
# 2.Pandas
# 3.Matplotlib
# 4.Scipy
# 5.Scikit-learn: which includes
# Classification: SVM, nearest neighbors, random forest, logistic regression, etc.
# Regression: Lasso, ridge regression, etc.
# Clustering: k-means, spectral clustering, etc.
# Dimensionality reduction: PCA, feature selection, matrix factorization, etc.
# Model selection: Grid search, cross-validation, metrics
# Preprocessing: Feature extraction, normalization
# 6.Statsmodel: which includes
# Regression models: Linear regression, generalized linear models, robust linear models, linear mixed effects models, etc.
# Analysis of variance (ANOVA)
# Time series analysis: AR, ARMA, ARIMA, VAR, and other models
# Nonparametric methods: Kernel density estimation, kernel regression
# Visualization of statistical model results.
## ** NOTE: ** statsmodels is more focused on statistical inference, providing uncertainty estimates
# and p-values for parameters. scikit-learn, by contrast, is more prediction-focused.

# CHAPTER2:_____________________________________________________________________
### Python Basics: Refer pg 15 onwards for details

# CHAPTER3:______________________________________________________________________
### Built-in Data Structures, Functions, and Files ###

## TUPLE:
# tuple is a fixed-length, IMMUTABLE sequence of Python objects. While the objects stored in a tuple may be mutable themselves, once the tuple is
# created it’s not possible to modify which object is stored in each slot
tuple = 41, 15, 16  # simple tuple. o/p is (4,5,6)
nested_tuple = (4, 5, 6), (7, 8)  # o/p is ((4, 5, 6), (7, 8))
# convert any sequence or iterator to a tuple by invoking tuple :
tuple([4, 0, 2])  # [4, 0, 2] converted to tuple
tuple('Srimal')  # o/p is ('S', 'r', 'i', 'm', 'a', 'l')
# accessing element inside tuple
nested_tuple[0]  # o/p is (4, 5, 6)
# concatenate tuples using the + operator to produce longer tuples
conc_tuple = tuple + nested_tuple  # o/p is (41, 15, 16, (4, 5, 6), (7, 8))
# Multiplying a tuple by an integer
tuple * 2  # o/p is (41, 15, 16, 41, 15, 16). NA for add,sub,multiply etc.

## Unpacking tuples: assign to a tuple-like expression of variables
# Note: here size, coveringetc needs to be SAME for assigment, else error is thrown
a, b, c = tuple
b  # 15 which is in line with same sized tuple created earlier
(x, y, z), (m, n) = nested_tuple
z  # o/p is 6
# unpacking over sequence:
seq = [(11, 22, 33), (41, 52, 63), (27, 58, 69)]
for a, b, c in seq:
    print('a={2}, b={0}, c={1}'.format(a, b, c))  # based on index value
# unpacking with *rest to capture an arbitrarily long list of positional arguments:
values = 1, 2, 3, 4, 5
a, b, *rest = values  ## Availablein python 3, clubs all values other than a,b
a, b, *_ = values  ## Availablein python 3, discard all values other than a,b
a, b  # o/p is (1,2)
rest  # o/p is (3,4,5)
# count on tuple element
a = (1, 2, 2, 2, 3, 4, 2)
a.count(2)  # o/p is 4, frequency of 2 in a

## LIST:In contrast with tuples, lists are variable-length and their contents can be modified in-place
# define them using 1)square brackets [] or 2)using the list type function:
a_list = [2, 3, 7, None]  # list using sq. brackets
b_list = list(tup)  # where tup=(3,4,5) # list using LIST function
a_list[0]  # o/p is 2... accessing index based elements
# Adding/insert elements
a_list.append(99)  # o/p is [2, 3, 7, None, 99]. 99 is added to list
a_list.insert(1, 111)  # inserts 111 at 1stindex. o/p is [2, 111, 3, 7, None, 99]
# Removing elements: inverse operation to insert is pop
a_list.pop(1)  # removes 1st index 111. o/p is [2, 3, 7, None, 99]
a_list.remove(7)  # removes 7 fromlist. '' to pass string value
# Note : These operations takes only ONE ARGUMENT. we cant pass multiple values
a_list.append([99, 999])  # will add [99,999] to a_list here
## Check if a list contains a value using the in keyword
7 in a_list  # true as 7 removed coz of remove operation
77 not in a_list  # true as 77not in list
## Concatenating and combining lists
a_list + b_list  # o/p is [2, 3, None, 99, [99, 999], 3, 4, 5]
# extending list
a_list.extend([4, 'test', ['srimal']])  # o/p is [2, 3, None, 99, [99, 999], 4, 'test', ['srimal']]
## Note: EXTEND is faster & efficient than CONCATENATE +

# SORTING:
a = [7221, 22, 5222, 122, 311344]
a.sort()  # [22, 122, 5222, 7221, 311344]
a.sort(reverse=True)  # [311344, 7221, 5222, 122, 22]
a_list.sort()  # [None, 2, 3, 4, 55, 99, [99, 999], ['srimal'], 'test'] numeric, alphabetically...
## Sorting by key
a.sort(key=len)  # not applicable for INTEGER
b = ['saw', 'small', 'He', 'foxes', 'six']
b.sort(key=len)  # o/p is ['He', 'saw', 'six', 'small', 'foxes']
## Binary search and maintaining a sorted list
import bisect

bisect.bisect(a, 2)  # o/p is 0. INDEX is shown here. we should place 2 at oth index of list a.
bisect.insort(a, 2)  # directly place value in list. o/p is [2, 311344, 7221, 5222, 122, 22]
## Note: bisect module functions do not check whether the list is sorted, as doing so would be computationally expensive. Thus, using
# them with an unsorted list will succeed without error but may lead to incorrect results.
# TIP: Better to SORT first & then apply BISECT function.

## SLICING:  basic form consists of start:stop command.
# number of elements in the result is stop - start .
a[:3]  # o/p is [2, 311344, 7221]
# Slices can also be assigned to with a sequence:
a[2:4] = [45, 54, 78]  # o/p is [7221, 22, 45, 54, 78, 311344]. The [5222, 122] is REMOVED
# negative Indices
a[:-3]  # [7221, 22, 45]
a[-5:-3]  # [22, 45]...ie -3-(-5)
# A step can also be used after a second colon to, say, take every other element:
a[::2]  # o/p is [7221, 45, 78]. Start, End & Stride is 2. Start to End with stride of 2.
a[::-2]  # o/p is [311344, 54, 22], with Opposite direction of above

## Built-in Sequence Functions
# ENUMERATE:
# when iterating over a sequence to want to keep track of the index of the current item
# returns a sequence of (i, value) tuples
some_list = ['foo', 'bar', 'baz']
mapping = {}
for i, v in enumerate(some_list):
    mapping[v] = i  # concept in python3
# SORTED: retruns new sorted list from the elements of any sequence:
sorted('Amu Anu')  # o/p is [' ', 'A', 'A', 'm', 'n', 'u', 'u']
# ZIP:“pairs” up the elements of a number of lists, tuples, or other sequences to create a list of tuples
# structure needs to be similar for creating list.
zipped = zip(['Amit', 'Annu', 'Ashu'], [32, 27, 25], ['baroda', 'tarsali', 'Srimal'])
list(zipped)  # o/p is [('Amit', 32, 'baroda'), ('Annu', 27, 'tarsali'), ('Ashu', 25, 'Srimal')]
# Unzip: from ZIPPED block
pitchers = [('Nolan', 'Ryan'), ('Roger', 'Clemens'), ('Schilling', 'Curt')]
first_names, last_names = zip(*pitchers)
first_names  # ('Nolan', 'Roger', 'Schilling')
last_names  # ('Ryan', 'Clemens', 'Curt')
# REVERSED:
list(reversed(range(10)))
# DICTIONARY: A more common name for it is hash map or associative array
d1 = {'a': 'some value', 'b': [1, 2, 3, 4]}
d1['b']  # o/p is [1, 2, 3, 4]
# access, insert, or set elements
d1[7] = 'Srimal'  # o/p is  {7: 'Srimal', 'a': 'some value', 'b': [1, 2, 3, 4]}
# delete values either using the del keyword or the pop method
del d1[7]  # 7 is key of dict d1. o/p is  {'a': 'some value', 'b': [1, 2, 3, 4]}
d1.pop('a')  # drops key a from dict.o/p is {'b': [1, 2, 3, 4]}
# getting key value pair of dict{}
list(d1.keys())  # gets key only. o/p is ['a', 'b']
list(d1.values())  # gets value only. o/p is  ['some value', [1, 2, 3, 4]]
# merge one dict into another, basically UPDATE
d1.update({'b': 'foo', 'c': 12})  # o/p is {'a': 'some value', 'b': 'foo', 'c': 12}. b is updated & c is added.

## Creating dicts from sequences:
# Since a dict is essentially a collection of 2-tuples, the dict function accepts a list of 2-tuples
tuple_1 = range(5)
tuple_2 = reversed(range(5))
dict_from_2tuples = dict(zip(tuple_1, tuple_2))  # use ZIP to combine two tuples, else throws error.
dict_from_2tuples  # o/p is  {0: 4, 1: 3, 2: 2, 3: 1, 4: 0}
## Valid dict key types:
# In Dictionary, key is immutable & technically its hashability.
# check with has function key
hash('string')  # 6756482857350662598
hash((1, 2, (2, 3)))  # 1097636502276347782
hash((1, 2, [2, 3]))  # fails coz list are MUTABLE.
# To use a list as a key, one option is to convert it to a tuple, which can be hashed as long as its elements
d = {}
d[tuple([1, 2, 3])] = 5
d  # o/p is {(1, 2, 3): 5}

## SET: A set is an unordered collection of unique elements.They are like dict, keys only, no values
# A set can be created in two ways:
# 1) the set function
set1 = set([2, 2, 2, 1, 3, 3])  # o/p is unique {1, 2, 3}
# 2) set literal with curly braces
{2, 2, 2, 1, 3, 3}  # o/p is {1, 2, 3}
# Sets support mathematical set operations like union, intersection, difference, and symmetric difference
a = {1, 2, 3, 4, 5}
b = {3, 4, 5, 6, 7, 8}
# UNION: set of distinct elements occurring in either set
a.union(b)  # o/p is {1, 2, 3, 4, 5, 6, 7, 8}
a | b  # o/p is {1, 2, 3, 4, 5, 6, 7, 8}
# INTERSECTION: contains the elements occurring in both sets
a.intersection(b)  # o/p is {3, 4, 5}
a & b  # o/p is {3, 4, 5}
# ADD
a.add(99)  # adds 9. o/p is {1, 2, 3, 4, 5, 99}
# RESET: Reset the set a to an empty state, discarding all of its elements
a.clear()  # empty set is o/p
# REMOVE
a.remove(99)  # o/p is {1, 2, 3, 4, 5}
# POP: Remove an arbitrary element from the set a , raising KeyError if the set is empty
a.pop()  # removed 1 7 o/p is {2, 3, 4, 5}
# UPDATE: Set the contents of a to be the union of the elements in a and b
a.update(b)  # a is changed as union of a,b. o/p is {1, 2, 3, 4, 5, 6, 7, 8}. b remains UNCHANGED
a |= b  # same as above. alternative syntax
# INTERSECTION UPDATE: Set the contents of a to be the intersection of the elements in a and b
a.intersection_update(b)  # a becomes {3,4,5}, b remains UNCHANGED
a &= b  # same as above. alternative syntax
# DIFFERENCE : The elements in a that are not in b
a.difference(b)  # o/p is {1, 2}
a - b  # same as above. alternative syntax
# DIFFERENCE UPDATE: Set a to the elements in a that are not in b
a.difference_update(b)  # a becomes {1,2}
a -= b  # same as above. alternative syntax
# a.symmetric_difference(b): All of the elements in either a or b but not both
# a.symmetric_difference_update(b): Set a to contain the elements in either a or b but not both
# a.issubset(b): True if the elements of a are all contained in b
# a.issuperset(b): True if the elements of b are all contained in a
# a.isdisjoint(b): True if a and b have no elements in common
# EQUAL: Sets are equal if and only if their contents are equal:
{1, 2, 3} == {3, 2, 1}  # returns TRUE as o/p

## List, Set, and Dict COMPREHENSIONS
## List Comprehension:
# allows to concisely form a new list by filtering the elements of a collection, transforming the elements passing the filter in one concise expression.
# syntax: [expr for val in collection if condition]
# eg: filter out strings with length 2 or less and also convert them to uppercase
strings = ['a', 'as', 'bat', 'car', 'dove', 'python']
[x.upper() for x in strings if len(x) > 2]  # o/p is ['BAT', 'CAR', 'DOVE', 'PYTHON']
## Dict Comprehension:
# syntax: dict_comp = {key-expr : value-expr for value in collection if condition}
## Set Comprehensions:
# syntax like list with CURLY: set_comp = {expr for value in collection if condition}
# eg: create a lookup map of these strings to their locations in the list:
{val: index for index, val in enumerate(strings)}  # {'a': 0, 'as': 1, 'bat': 2, 'car': 3, 'dove': 4, 'python': 5}
## Nested list comprehensions
# eg1: get a single list containing all names with two or more e ’s in them.
name_list = [['John', 'Emily', 'Michael', 'Mary', 'Steven'], ['Maria', 'Juan', 'Javier', 'Natalia', 'Pilar']]
[name for names in name_list for name in names if name.count('e') >= 2]  # o/p is ['Steven']
# eg2:“flatten” a list of tuples of integers into a simple list of integers:
some_tuples = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]
flattened = [x for tup in some_tuples for x in tup]
flattened  # o/p is  [1, 2, 3, 4, 5, 6, 7, 8, 9]

### FUNCTIONS:
# If Python reaches the end of a function without encountering a return statement, None is returned automatically.
# Each function can have positional arguments and keyword arguments.
# keyword argument: specify default values or optional arguments
func(x, y, z=6)  # here x,y are Positional & z is Keyword
# main restriction on function arguments is that the keyword arguments must follow the positional arguments (if any)

## Namespaces, Scope, and Local Functions
## NAMESPACE: an alternative and more descriptive name describing a variable scope in Python is a namespace
# Functions can access variables in two different scopes: global and local.
# The LOCAL namespace is created when the function is called and immedi ately populated by the function’s arguments. After the function is finished, the local namespace is destroyed
# Assigning variables outside of the function’s scope is possible, but those variables must be declared as global via the global keyword:
a = None


def bind_a_variable():
    global a


a = []
bind_a_variable()


# Returning Multiple Values
# functions returns only one o/p. but with python its possible to get multiple value
# eg1.
def f():
    a = 5


b = 6
c = 7
return a, b, c


# eg2.
def f():
    a = 5


b = 6
c = 7
return {'a': a, 'b': b, 'c': c}
# eg3.data cleansing operation for states String
states = ['Alabama ', 'Georgia!', 'Georgia', 'georgia', 'FlOrIda', 'south carolina##', 'West virginia?']


def remove_punctuation(value):
    return re.sub('[!#?]', '', value)


clean_ops = [str.strip, remove_punctuation, str.title]


def clean_strings(strings, ops):
    result = []


for value in strings:
    for function in ops:
        value = function(value)
        result.append(value)
return result

## Anonymous (Lambda) Functions
# writing functions consisting of a single statement, the result of which is the return value.
g = lambda x: x * 2  # retruns twice the value passed in function
g(5)  # returns 5*2 as output here
# eg2.sort a collection of strings by the number of distinct letters in each string
strings = ['foo', 'card', 'bar', 'aaaa', 'abab']
strings.sort(key=lambda x: len(set(list(x))))
strings  # ['aaaa', 'foo', 'abab', 'bar', 'card']


## Currying: Partial Argument Application
# means deriving new functions from existing ones by partial argument application
def add_numbers(x, y):
    return x + y


# derive new function from add_numbers which add 555
add_five = lambda y: add_numbers(555, y)  # add_numbers is said to be curried
# Alternatively in python, The built-in functools module can simplify this process using the partial function:
from functools import partial

add_five = partial(add_numbers, 5)  # either x or y is assigned 5, remaning argument passed with add)five function.
add_five(6)  # gives 11 as o/p here

## Generators:means of the iterator protocol, a generic way to make objects iterable
some_dict = {'a': 1, 'b': 2, 'c': 3}
for key in some_dict:
    print(key)
# same to achieve with ITERATOR
dict_iterator = iter(some_dict)
list(dict_iterator)  # o/p is  ['a', 'b', 'c']


## Difference: Whereas normal functions execute and return a single result at a time, generators return a sequence of
# multiple results lazily, pausing after each one until the next one is requested
# To create a generator, use the YIELD keyword instead of return in a function:
def squares(n=10):
    print('Generating squares from 1 to {0}'.format(n ** 2))


for i in range(1, n + 1):
    yield i ** 2
gen = squares()  # assigning the function to gen variable
# It is not until you request elements from the generator that it begins executing its code:
for x in gen:
    print(x, end=' ')  # o/p is 1 4 9 16 25 36 49 64 81 100

## ITERTOOLs module
import itertools

first_letter = lambda x: x[0]
names = ['Alan', 'Adam', 'Wes', 'Will', 'Albert', 'Steven']
for letter, names in itertools.groupby(names, first_letter):
    print(letter, list(names))  # names is a generator


## Exception handling
# The code in the except part of the block will only be executed if float(x) raises an exception
def attempt_float(x):
    try:
        return float(x)
    except:
        return x


### Files and the Operating System. refer pg 80 onwards of 541


# CHAPTER4:______________________________________________________________________
### NumPy Basics: Arrays and Vectorized Computation ###
# NumPy is a Python package which stands for ‘Numerical Python’.
# It is the core library for scientific computing, which contains a powerful n-dimensional array object, provide tools for integrating C, C++ etc.
# It is also useful in linear algebra, random number capability etc.
# NumPy array can also be used as an efficient multi-dimensional container for generic data.

# -----------------------------------------------------------------------------------------------------------------------
## ARRAY:  is a data structure consisting of a collection of elements, each identified by at least one array index or key.
## An array is stored such that the position of each element can be computed from its index tuple by a mathematical formula.

## LIST: It’s a collection of items (called nodes) ordered in a linear sequence

##List vs Array:
# A list is a different kind of data structure from an array.
# The biggest difference is in the idea of direct access Vs sequential access. Arrays allow both; direct and sequential access, while lists allow only sequential access. And this is because the way that these data structures are stored in memory.
# In addition, the structure of the list doesn’t support numeric index like an array is. And, the elements don’t need to be allocated next to each other in the memory like an array is.
# --------------------------------------------------------------------------------------------------------------------
# NumPy Array: Numpy array is a powerful N-dimensional array object which is in the form of rows and columns.
# We can initialize numpy arrays from nested Python lists and access it elements.

## Python Numpy array instead of a list because of the below three reasons:
# Less Memory
# Fast
# Convenient

# NumPy by itself does not provide modelling or scientific functionality, having an understanding of NumPy arrays and array-oriented computing will help you use tools with array-oriented semantics, like pandas
# NumPy-based algorithms are generally 10 to 100 times faster (or more) than their pure Python counterparts and use significantly less memory.
my_arr = np.arange(1000000)
my_list = list(range(1000000))
# let’s multiply each sequence by 2 & get execution time
% time
for _ in range(10): my_arr2 = my_arr * 2  # time taken is 76.1 ms
% time
for _ in range(10): my_list2 = [x * 2 for x in my_list]  # time taken is 1.34 s

## NumPy ndarray: A Multidimensional Array Object
data = np.random.randn(2, 3)
data.shape  # (2,3) data
data.dtype  # dtype('float64'), which is the default data type
## Creating ndarrays
arr1 = np.array([6, 7.5, 8, 0, 1])  # arr1 is array([ 6. ,  7.5,  8. ,  0. ,  1. ])
arr2 = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
arr2.shape  # shape is (2,4) ie 2 blocks having 4 elements each
arr2.ndim  # o/p is 2
## Creating arrays with ZERo & ONES
np.zeros(10)  # o/p is array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])
np.zeros((3, 6))  # o/p is full of zero, in 3 rows & 6columns
np.empty((2, 3, 2))  # empty creates an array without initializing its values to any particular value.
# Imp: It’s not safe to assume that np.empty will return an array of all zeros. In some cases, it may return uninitialized “garbage” values
# array: Convert input data (list, tuple, array, or other sequence type) to an ndarray either by inferring a dtype or explicitly specifying a dtype; copies the input data by default
# asarray: Convert input to ndarray, but do not copy if the input is already an ndarray
# arange: Like the built-in range but returns an ndarray instead of a list
# ones,ones_like: Produce an array of all 1s with the given shape and dtype; ones_like takes another array and produces a ones array of the same shape and dtype
# zeros,zeros_like: : Like ones and ones_like but producing arrays of 0s instead
# empty,empty_like: Create new arrays by allocating new memory, but do not populate with any values like ones and zeros
# full,full_like: Produce an array of the given shape and dtype with all values set to the indicated “fill value”Produce an array of the given shape and dtype with all values set to the indicated “fill value”full_like takes another array and produces a filled array of the same shape and dtype
# eye, identity: Create a square N × N identity matrix (1s on the diagonal and 0s elsewhere)

## Data Types for ndarrays
# data type or dtype is a special object containing the information (or metadata, data about data)
arr = np.array([1, 2, 3], dtype=np.int32)
arr.dtype  # dtype('int32'). defualt is int64.
## convert or CAST an array from one dtype to another using ndarray’s astype method
# eg1. convert from int32 to float64
float_arr = arr.astype(np.float64)  # converts aa from int32 to float 64.
float_arr.dtype  # dtype('float64')
# eg2. convert from float64 to int32
arr1 = np.array([3.7, -1.2, -2.6, 0.5, 12.9, 10.1])  # data type is dtype('float64')
int_arr = arr1.astype(np.int32)  # (dtype=int32) from float64.
# eg3. convert from string to float
arr2 = np.array(['1.25', '-9.6', '42'], dtype=np.string_)  # its numeric data type
str_arr = arr2.astype(float)
str_arr.dtype  # from string to dtype('float64')
# eg4. use another array’s dtype attribute
int_array = np.arange(10)  # dtype('int64')
calibers = np.array([.22, .270, .357, .380, .44, .50], dtype=np.float64)  # float64
int_array.astype(calibers.dtype)  # convert int_array AS calibers data type. its now float64

## Arithmetic with NumPy Arrays; thanks to VECTORIZATION
# Arrays are important because they enable you to express batch operations on data without writing any for loops. NumPy users call this vectorization.
# Any arithmetic operations between equal-size arrays applies the operation element-wise.
arr = np.array([[1., 2., 3.], [4., 5., 6.]])
arr * arr, arr + arr, arr - arr, arr / arr
# IMP: Operations between differently sized arrays is called BROADCASTING.

## Basic Indexing and Slicing: start & stop, ends with diff of (stop-start)
# eg: 1d arrays:
arr1d = np.arange(10)  # array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
arr1d_slice0 = arr1d[2:8:3]  # o/p is  array([2, 5]), JUMP of 3..
arr1d_slice1 = arr1d[5:8]  # o/p is array([5, 6, 7])
arr1d_slice2 = arr1d[:5]  # o/p is array([0, 1, 2, 3, 4])
arr1d_slice3 = arr1d[3:]  # o/p is array([3, 4, 5, 6, 7, 8, 9])
arr1d_slice4 = arr1d[-3:]  # o/p is array([7, 8, 9])
# array slices are VIEWS on the original array. This means that the data is not copied, and any modifications to the view will be reflected in the source array.
# Slice's DIMENSION remain same as parent array.
# assigning values to sliced part
arr1d_slice[:] = 454  # chnages all value of index 5,6,7 to 454
# eg2. 2d arrays:
arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
arr2d[2]  # o/p is array([7, 8, 9])
arr2d[2, 2]  # o/p is 9, row column basis value is PICKED
arr2d[2][2]  # o/p is 9, Same as above
arr2d[:2]  # o/p is array([[1, 2, 3], [4, 5, 6]])
arr2d[1:2]  # o/p is array([[4, 5, 6]])
arr2d[:-2]  # o/p is array([[1, 2, 3]])
# pass multiple slices just like you can pass multiple indexes
arr2d[:2, 1:]  # first take block & then elements. o/p is array([[2, 3],[5, 6]])
arr2d[:2, 2]  # o/p is array([3, 6])
arr2d[:, :1]  # o/p is array([[1],[4],[7]])
# eg3. 3d arrays:
arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
# if you omit later indices, the returned object will be a lower dimensional ndarray consisting of all the data along the higher dimensions
arr3d[0]  # array([[1, 2, 3], [4, 5, 6]])
arr3d[1]  # array([[ 7,  8,  9], [10, 11, 12]])
arr3d[2]  # index 2 is out of bounds for axis 0 with size 2
arr3d[0, 0]  # array([1, 2, 3])
arr3d[1, 0]  # array([7, 8, 9])

## Boolean Indexing
names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
data = np.random.randn(7, 4)  # matrix of 7 row & 4 columns
# lets say each name corresponds to a row in the data array
names == 'Bob'  # [ True, False, False,  True, False, False, False]
# boolean array can be passed when indexing the array:
data[names == 'Bob']  # retruns data for name = 'Bob' ONLY. for TRUE, it returns values
data[names == 'Bob', 2:]  # same as aboe, with column 2nd index onwards
data[names != 'Bob']  # reruens Non 'Bob' data, 5rows & 4columns
# selecting multiple casing
mask = (names == 'Bob') | (names == 'Will')
data[mask]  # gets 'Bob OR Will' data
data[~mask]  # gets NEGATE of above implementation.
# Note:
# 1. Selecting data from an array by boolean indexing always creates a copy of the data, even if the returned array is unchanged
# 2.The Python keywords and and or do not work with boolean arrays. Use & (and) and | (or) instead.

## Fancy Indexing:
# is a term adopted by NumPy to describe indexing using integer arrays
arr = np.empty((8, 4))
for i in range(8):
    arr[i] = i
# select out a subset of the rows in a particular order: simply pass a list or ndarray of integers specifying the desired order
arr[[4, 3, 0, 6]]
# Reshaping
arr2 = np.arange(32).reshape((8, 4))  # reshaping is possible on MULTIPLE values ONLY.
arr2[[1, 5, 7, 2], [0, 3, 1, 2]]  # indexing values from arr2. o/p is array([ 4, 23, 29, 10])
arr2[[1, 5, 7, 2]][:, [0, 3, 1, 2]]  # 4by4 row,col based on order as set, : says full matrix view
arr2.reshape(4, 8)  # array of 4 rows & 8 columns

## Transposing Arrays and Swapping Axes
# Transposing is a special form of reshaping that similarly returns a view on the under lying data without copying anything
arr = np.arange(15).reshape((3, 5))
arr.T
# computing the inner matrix product using np.dot
np.dot(arr.T, arr)  # rows of 1st & columns of 2nd becomes dimension of .dot product
np.dot(arr, arr.T)
# For higher dimensional arrays, transpose will accept a tuple of axis numbers to permute the axes
arr = np.arange(16).reshape((2, 2, 4))
arr.transpose((1, 0, 2))
# Swapping: swapaxes , which takes a pair of axis numbers and switches the indicated axes to rear‐ range
# swapaxes returns a VIEW on the data without making a copy.
arr = np.arange(16).reshape((2, 2, 4))
arr.swapaxes(1, 2)

## Universal Functions: Fast Element-Wise Array Functions
# universal function, or ufunc, is a function that performs element-wise operations on data in ndarrays.
# eg1. UNARY : taking just one i/p
np.sqrt(np.arange(5))  # array([ 0.,1.,1.41421356,1.73205081,2.])
np.exp(np.arange(5))  # array([1.,2.71828183,7.3890561,20.08553692,54.59815003])
# also called UNARY ufuncs. Others, such as add or maximum , take two arrays (BINARY ufuncs) and return a single array as the result
# abs, fabs: Compute the absolute value element-wise for integer, floating-point, or complex values
# sqrt: Compute the square root of each element (equivalent to arr ** 0.5 )
# square: Compute the square of each element (equivalent to arr ** 2 )
# exp: Compute the exponent e x of each element
# log, log10,log2, log1p: Natural logarithm (base e), log base 10, log base 2, and log(1 + x), respectively
# sign: Compute the sign of each element: 1 (positive), 0 (zero), or –1 (negative)
# ceil: Compute the ceiling of each element (i.e., the smallest integer greater than or equal to that number)
# floor: Compute the floor of each element (i.e., the largest integer less than or equal to each element)
# rint: Round elements to the nearest integer, preserving the dtype
# modf: Return fractional and integral parts of array as a separate array
# isnan: Return boolean array indicating whether each value is NaN (Not a Number)
# isfinite, isinf: Return boolean array indicating whether each element is finite (non- inf , non- NaN ) or infinite,respectively
# cos, cosh, sin,sinh, tan, tanh:Regular and hyperbolic trigonometric functions
# arccos, arccosh,arcsin, arcsinh,arctan, arctanh :Inverse trigonometric functions
# logical_not: Compute truth value of not x element-wise (equivalent to ~arr ).
# eg2. BINARY" taking just 2 i/p
x = np.random.randn(8)
y = np.random.randn(8)
z = np.random.randn(6)
np.maximum(x, y)  # o/p is element wise max
np.minimum(x, y)  # o/p is element wise min
np.minimum(x, z)  # broadcast error coz of x,z shape issue
# add: Add corresponding elements in arrays
# subtract: Subtract elements in second array from first array
# multiply: Multiply array elements
# divide, floor_divide: Divide or floor divide (truncating the remainder)
# power: Raise elements in first array to powers indicated in second array
# maximum, fmax: Element-wise maximum; fmax ignores NaN
# minimum, fmin: Element-wise minimum; fmin ignores NaN
# mod: Element-wise modulus (remainder of division)
# copysign: Copy sign of values in second argument to values in first argument
# greater, greater_equal,less, less_equal,equal, not_equal: Perform element-wise comparison, yielding boolean array (equivalent to infix operators >, >=, <, <=, ==, != )
# logical_and,logical_or, logical_xor: Compute element-wise truth value of logical operation (equivalent to infix operators & |, ^ )
# eg3. POLYNOMIAL: taking multiple i/p
# modf returns the fractional and integral parts of a floating-point array
arr = np.random.randn(3) * 5
remainder, whole_part = np.modf(arr)
remainder  # array([ 0.78231947, -0.74935188,  0.07131087])
whole_part  # array([ 1., -7.,  1.])
# eg4. OPTIONAL
np.sqrt(arr)  # array([ 1.33503538,nan,1.03504148])

## Array-Oriented Programming with Arrays
# practice of replacing explicit loops with array expressions is commonly referred to as vectorization
# vectorized array operations will often be one or two (or more) orders of magnitude faster than their pure Python equivalents
import numpy as np

## Mathematical and Statistical Methods
# can use aggregations (often called reductions) like sum , mean , and std (standard deviation) either by calling the array instance method or using the top-level NumPy function.
arr = np.random.randn(5, 4)
arr.mean()
np.mean(arr)
arr.sum()
# Functions like mean and sum take an optional axis argument that computes the statis tic over the given axis
arr.mean(axis=1)  # along column side
arr.sum(axis=0)  # along row side
# Other methods like cumsum and cumprod do not aggregate, instead producing an array of the intermediate results:
arr1d = np.array([0, 1, 2, 3, 4, 5, 6, 7])
arr1d.cumsum()  # o/p is array([ 0,  1,  3,  6, 10, 15, 21, 28])
# now on multi-dimensional array
arr2d = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
arr2d.cumsum(axis=0)  # o/p is array([[ 0,  1,  2],[ 3,  5,  7],[ 9, 12, 15]])
arr2d.cumprod(axis=1)  # o/p is array([[  0,   0,   0],[  3,  12,  60], [  6,  42, 336]])
# sum: Sum of all the elements in the array or along an axis; zero-length arrays have sum 0
# mean: Arithmetic mean; zero-length arrays have NaN mean
# std, var: Standard deviation and variance, respectively, with optional degrees of freedom adjustment (default denominator n )
# min, max: Minimum and maximum
# argmin, argmax: Indices of minimum and maximum elements, respectively
# cumsum: Cumulative sum of elements starting from 0
# cumprod: Cumulative product of elements starting from 1

## Methods for Boolean Arrays
# Boolean values are coerced to 1 ( True ) and 0 ( False ) in the preceding methods. Thus, sum is often used as a means of counting True values in a boolean array:
arr = np.random.randn(100)
(arr > 0).sum()  # Number of positive values. o/p is 37
# Any & All methods: Check for True instances
bools = np.array([False, False, True, False])
bools.any()  # checks if ANY of the bools is True
bools.all()  # checks if ALL of the bools are True
# Note: methods also work with non-boolean arrays, where non-zero elements evaluate to True .
arr1 = np.array([6, 4, 0, 8, 0, 1, 122, 0, 9, 0])
arr1.any()  # gets true since one non zero element present
arr1.all()  # gets false since non zero elements are present

## Sorting
arr1 = np.random.randn(10)
arr1.sort()  # sorted arr is returned
arr2 = np.random.randn(5, 3)
arr2.sort(1)  # column wise sorting. 5by3 array
arr2.sort(0)  # row wise sorting. 5by3 array
# Note: top-level method np.sort returns a sorted copy of an array instead of modifying the array in-place
np.sort(arr2, axis=0)  # same as above, but COPY is returned

## Unique and Other Set Logic: for one-dimensional ndarrays
names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
np.unique(names)  # ['Bob', 'Joe', 'Will']
# unique(x): Compute the sorted, unique elements in x
# intersect1d(x, y): Compute the sorted, common elements in x and y
# union1d(x, y): Compute the sorted union of elements
# in1d(x, y):Compute a boolean array indicating whether each element of x is contained in y
# setdiff1d(x, y): Set difference, elements in x that are not in y
# setxor1d(x, y): Set symmetric differences; elements that are in either of the arrays, but not both

### File Input and Output with Arrays:
# np.save and np.load are the core functionality
# NumPy is able to save and load data to and from disk either in text or binary format.
# Arrays are saved by default in an uncompressed raw binary format with file extension .npy
arr = np.arange(100)
np.save('some_array_file', arr)  # saving arr in some_array_file, ext is .npy
np.load('some_array_file.npy')  # loading arr from saved file format
# saving multiple arrays
np.savez('array_archive.npz', a=arr, b=arr)
arch = np.load('array_archive.npz')
arch['a']  # retrieving elements of arr a by loading..
# saving file in Compressed format
np.savez_compressed('arrays_compressed.npz', a=arr, b=arr)

## Pseudorandom Number Generation
# these are pseudorandom numbers because they are generated by an algo rithm with
# deterministic behavior based on the seed of the random number generator.
np.random.seed(10)
arr = np.random.rand(5)
# seed: Seed the random number generator
# permutation: Return a random permutation of a sequence, or return a permuted range
# shuffle: Randomly permute a sequence in-place
# rand: Draw samples from a uniform distribution
# randint: Draw random integers from a given low-to-high range
# randn: Draw samples from a normal distribution with mean 0 and standard deviation 1 (MATLAB-like interface)

## Example: Random Walks: Refer pg 119 for details

# CHAPTER5:_____________________________________________________________________
### Getting Started with pandas
# The biggest difference is that pandas is designed for working with tabular or heterogeneous data.
# NumPy, by contrast, is best suited for working with homogeneous numerical array data.

# ----------------------------------------------------------------------------------------------------------------
# Pandas is used for data manipulation, analysis and cleaning. Python pandas is well suited for different kinds of data, such as:
# Tabular data with heterogeneously-typed columns
# Ordered and unordered time series data
# Arbitrary matrix data with row & column labels
# Unlabelled data
# Any other form of observational or statistical data sets

# Python operation: Slicing, Merging and Joining, Concatenation, Index Change, Change Column Header, Data Munging

import pandas as pd
from pandas import Series, DataFrame  # import Series and DataFrame into the local namespace

## Pandas Data Structure
# Series: is a one-dimensional array-like object containing a sequence of values (of similar types to NumPy types) and an associated array of data labels, called its index.
obj = pd.Series([4, 7, -5, 3])
# Index: default one consisting of the integers 0 through N - 1 (where N is the length of the data) is created
obj.values  # o/p is array([ 4,  7, -5,  3])
obj.index  # o/p is  RangeIndex(start=0, stop=4, step=1)
# array with predefined custom index
obj2 = pd.Series([4, 7, -5, 3], index=['d', 'b', 'a', 'c'])
obj2.index  # o/p is Index(['d', 'b', 'a', 'c'], dtype='object')
obj2[['c', 'a', 'd']]  # accessing multiple index element. o/p is 3,-5,4
obj2[obj2 < 0]  # o/p is -5

# Another way to think about a Series is as a fixed-length, ordered dict.
sdata = {'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah': 5000}  # dictionary
obj3 = pd.Series(sdata)  # dictionary to series
obj3.values  # o/p is array([35000, 16000, 71000,  5000])
obj3.index  # o/p is Index(['Ohio', 'Oregon', 'Texas', 'Utah'], dtype='object')
# Series from other index
index_new = ['California', 'Ohio', 'Oregon', 'Texas', 'Delhi']
obj4 = pd.Series(sdata, index=index_new)  # dictionary value ONLY used in series, not keys. California has NaN.
# Only key is maintained in final o/p.
# Handling missing data: missing” or “NA” interchangeably to refer to missing data
# isnull and notnull functions in pandas should be used to detect missing data
pd.isnull(obj4)
pd.notnull(obj4)
obj3 + obj4  # has all KEYS from both Series
# Naming Series object & its index
obj4.name = 'population'
obj4.index.name = 'state'
obj4

## DataFrame
# Ways to construct a DataFrame
# eg1: from a dict of equal-length lists or NumPy arrays
data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada', 'Nevada'], 'year': [2000, 2001, 2002, 2001, 2002, 2003],
        'pop': [1.5, 1.7, 3.6, 2.4, 2.9, 3.2]}
frame = pd.DataFrame(data)
# specify sequence of columns
pd.DataFrame(data, columns=['year', 'state', 'pop'])
pd.DataFrame(data, columns=['year', 'state', 'pop', 'unknown'])  # unknown column appears as NaN.
# retrieving column values
frame['state']
## retreiving rows with LOC/iLOC: Rows can also be retrieved by position or name with the special loc attribute
frame.loc[1, 'state']  # specify row 1 data , State column element
frame.loc[2]  # specify row 1 data , each column element
frame.iloc[3]  # specify all data of row index 2

# first add a new column of boolean values where the state column equals 'Ohio'
frame['eastern'] = frame.state == 'Ohio'
frame['eastern'] = 'NDLS'  # create new column & assigning value NDLS
del frame['eastern']  # will delete eastern column from frame
frame.columns  # returns the column name of data frame
frame.values  # retruns array of frame as o/p

## Index Objects
# pandas’s Index objects are responsible for holding the axis labels and other metadata (like the axis name or names
obj = pd.Series(range(3), index=['a', 'b', 'c'])
obj.index  # Index(['a', 'b', 'c'], dtype='object')
obj.index[1:]  # o/p is Index(['b', 'c'], dtype='object')
# Imp: Index objects are immutable and thus can’t be modified by the user
obj.index['a'] = 'AA'  # Error saying Index does not support mutable operations
# Sharing Index objects operations
labels = pd.Index(np.arange(3))
obj2 = pd.Series([1.5, -2.5, 0], index=labels)
obj2  # returns o/p as obj2 object
# NOTE: Unlike Python sets, a pandas Index can contain duplicate labels
dup_labels = pd.Index(['foo', 'foo', 'bar', 'bar'])  # duplicate index labels
# Selections with duplicate labels will select all occurrences of that label
# append: Concatenate with additional Index objects, producing a new Index
# difference: Compute set difference as an Index
# intersection:Compute set intersection
# union:Compute set union
# isin: Compute boolean array indicating whether each value is contained in the passed collection
# delete: Compute new Index with element at index i deleted
# drop: Compute new Index by deleting passed values
# insert: Compute new Index by inserting element at index i
# is_monotonic: Returns True if each element is greater than or equal to the previous element
# is_unique: Returns True if the Index has no duplicate values
# unique: Compute the array of unique values in the Index

## Essential Functionality
# Re-Indexing: to create a new object with the data conformed to a new index.
obj = pd.Series([4.5, 7.2, -5.3, 3.6], index=['d', 'b', 'a', 'c'])
# Calling reindex on this Series rearranges the data according to the new index, introducing missing values if any index values were not already present
obj2 = obj.reindex(['a', 'b', 'c', 'd', 'e', 'c'])  # missing index e will have NaN.
# Interpolation: interpolation or filling of values when reindexing.
obj3 = pd.Series(['blue', 'purple', 'yellow'], index=[0, 2, 4])
obj3.reindex(range(6), method='ffill')  # obj3 has 6 elements, with values Forward filled
# reindex can alter either the (row) index, columns, or both
frame1 = pd.DataFrame(np.arange(9).reshape((3, 3)), index=['a', 'c', 'd'], columns=['Ohio', 'Texas', 'California'])
frame2 = frame1.reindex(['a', 'b', 'c', 'd'])
# Reindexing with column keywords
states = ['Texas', 'Utah', 'California']
frame1.reindex(columns=states)  # earlier column index renamed to States value.
frame1.loc[['a', 'd'], states]  # reurns a,d row from States column
frame1.loc[['a', 'd'], 'Texas']  # returns a,d row from Texas column
# index:New sequence to use as index. Can be Index instance or any other sequence-like Python data structure. An Index will be used exactly as is without any copying.
# method: Interpolation (fill) method; 'ffill' fills forward, while 'bfill' fills backward.
# fill_value:Substitute value to use when introducing missing data by reindexing.
# limit:When forward- or backfilling, maximum size gap (in number of elements) to fill.
# tolerance: When forward- or backfilling, maximum size gap (in absolute numeric distance) to fill for inexact matches.
# level:Match simple Index on level of MultiIndex; otherwise select subset of.
# copy: If True , always copy underlying data even if new index is equivalent to old index; if False , do not copy the data when the indexes are equivalent.

## Dropping Entries from an Axis
new_obj = frame1.drop('c')  # c row is deleted from frame1
# index values can be deleted from either axis in data frame
frame3 = pd.DataFrame(np.arange(9).reshape((3, 3)), index=['a', 'c', 'd'], columns=['Ohio', 'Texas', 'California'])
frame3.drop(['d', 'c'])  # row indexed c,d is dropped frame3
frame3.drop(['Ohio', 'Texas'], axis=1)  # column Ohio,Texas is dropped from frame3. Default axis is 0
# IMP: Be careful with the inplace , as it destroys any data that is dropped.
frame3.drop('c', inplace=True)  # c is permanently removed from frame3
frame3.drop('California', axis=1, inplace=True)  # California column permanently removed

## Indexing, Selection, and Filtering
# Index
obj = pd.Series(np.arange(4.), index=['a', 'b', 'c', 'd'])
obj[2]  # o/p is 2 . Based on index Integer position
obj[1:3]  # o/p is [1,2] i.e stop-start Values
obj['c']  # o/p is 2 . Based on Index name
obj[['b', 'a', 'd']]  # o/p is [1,0,3]
obj[obj < 2]  # o/p is based on gven condition
# Slicing with labels behaves differently than normal Python slicing
obj['a':'c']  # o/p is [0,1,2] based on indexed passed

# Selction:
# Selection with loc[NAME passed] and iloc[INTEGER passed]
frame4 = pd.DataFrame(np.arange(9).reshape((3, 3)), index=['a', 'c', 'd'], columns=['Ohio', 'Texas', 'California'])
frame4.loc['a', ['Ohio', 'Texas']]  # from a, gets Ohio & Texas value
frame4.loc[:, ['Ohio', 'Texas']]  # from all rows, gets Ohio & Texas
frame4.loc['c':'d', ['Ohio', 'Texas']]  # from c,d, gets Ohio & Texas
frame4.iloc[2, [2, 0, 1]]  # returns o/p based on Indexed integer, Rows by Columns
frame4.iloc[1:4, [2, 0, 1]]  # for rows, not boundary, but YES for column
frame4.iloc[1:2, [88, 0, 1]]  # gives out of bound error for column 88
# df[val]:Select single column or sequence of columns from the DataFrame;
# df.loc[val]:Selects single row or subset of rows from the DataFrame by label
# df.loc[:,val]:Selects single column or subset of columns by label
# df.loc[val1,val2]: Select both rows and columns by label
# df.iloc[where]: Selects single row or subset of rows from the DataFrame by integer position
# df.iloc[:, where]: Selects single column or subset of columns by integer position
# df.iloc[where_i, where_j]: Select both rows and columns by integer position
# df.at[label_i, label_j]: Select a single scalar value by row and column label
# df.iat[i, j]: Select a single scalar value by row and column position (integers)
# reindex method: Select either rows or columns by labels
# get_value, set_value methods: get_value, set_value methods Select single value by row and column label

## Arithmetic and Data Alignmen
# When you are adding together objects, if any index pairs are not the same, the respective index in the result will be the union of the index pairs.
# similar to an automatic outer join on the index labels
s1 = pd.Series([7.3, -2.5, 3.4, 1.5], index=['a', 'c', 'd', 'e'])
s2 = pd.Series([-2.1, 3.6, -1.5, 4, 3.1], index=['a', 'c', 'e', 'f', 'g'])
s1 + s2  # o/p has nan for d,f,g index position. Since no corresponfing value in s2
# Imp: internal data alignment introduces missing values in the label locations that don’t overlap
# Similiar is the case when tried with data drame having matching index etc. NaN at misnatched instances

## Arithmetic methods with fill values
df1.add(df2, fill_value=0)  # will add 4+nan as 4 in o/p is rendered.
# add, radd: Methods for addition (+)
# sub, rsub: Methods for subtraction (-)
# div, rdiv: Methods for division (/)
# floordiv, rfloordiv: Methods for floor division (//)
# mul, rmul:Methods for multiplication (*)
# pow, rpow: Methods for exponentiation (**)

## ***  Operations between DataFrame and Series
arr = np.arange(12.).reshape((3, 4))  # 2d array. considering equivalent to df
arr[0]  # o/p is 0th index row. array([ 0.,  1.,  2.,  3.]). equivalent to series
arr - arr[0]  # retruns 3by4
# Imp: When we subtract arr[0] from arr , the subtraction is performed once for each row.
# This is called broadcasting
#  By default, arithmetic between DataFrame and Series matches the index of the Series on the DataFrame’s columns, broadcasting down the rows:
dataframe.sub(series, axis=0)  # subtarct along each row element
dataframe.sub(series, axis=1)  # subtarct along each column element

## Function Application and Mapping
# NumPy ufuncs (element-wise array methods) also work with pandas objects
frame = pd.DataFrame(np.random.randn(4, 3), columns=list('bde'), index=['Utah', 'Ohio', 'Texas', 'Oregon'])
np.abs(frame)  # reurns absolute value of frame object
# Another frequent operation is applying a function on one-dimensional arrays to each column or row
f = lambda x: x.max() - x.min()
frame.apply(f)  # default axis is 0, for rows
frame.apply(f, axis='columns')  # axis changes to 1, columns

## Sorting and Ranking
# To sort lexicographically by row or column index, use the sort_index method
obj = pd.Series(range(4), index=['d', 'a', 'b', 'c'])
obj.sort_index()  # retuns according to [a,b,c,d] index
obj.sort_index(axis=0)
obj.sort  # 'Series' object has no attribute 'sort'
# Here this object can be series, dataframe, Sorting is performed
frame = pd.DataFrame(np.random.randn(4, 3), columns=list('gde'), index=['Utah', 'Ohio', 'Texas', 'Oregon'])
frame.sort_index(axis=0)  # sorting Ohio, Oregon, texas & Utah based on Rows
frame.sort_index(axis=1)  # sorting based on d,e,g Columns
frame.sort_index(axis=1, ascending=False)  # sorted as d,e,g
# To sort a Series by its values, use its sort_values method:
obj1 = pd.Series([4, 7, -3, 2])
obj1.sort_values()  # o/p is -3,2,4,7
obj2 = pd.Series([4, np.nan, 7, np.nan, -3, 2])
obj2.sort_values()  # nan are sorted at last. o/p is -3,2,4,7,nan,nan
## data frame Sorting
frame = pd.DataFrame({'b': [4, 7, -3, 2], 'a': [0, 1, 0, 1]})
frame.sort_values(by='b')  # entire df sorted based on column b
frame.sort_values(by=['a', 'b'])  # sorting on multiple columns of df

## RANKING:assigns ranks from one through the number of valid data points in an array
obj = pd.Series([7, -5, 7, 4, 2, 0, 4])
obj.rank()  # o/p is [6.5,1,,6.5,4.5,3,2,4.5]. Shared Rank for Duplicate values
# sorting based on when they are observed
obj.rank(method='first')  # o/p is [6,1,7,4,3,2,5]
obj.rank(ascending=False)  # ranking order is reversed. i.e. descending
# DataFrame can compute ranks over the rows or the columns
frame = pd.DataFrame({'b': [4.3, 7, -3, 2], 'a': [0, 1, 0, 1], 'c': [-2, 5, 8, -2.5]})
frame.rank(axis='columns')  # returns column wise rank of elements
# average: Default: assign the average rank to each entry in the equal group
# min: Use the minimum rank for the whole group
# max: Use the maximum rank for the whole group
# first: Assign ranks in the order the values appear in the data
# dense: Like method='min' , but ranks always increase by 1 in between groups rather than the number of equal elements in a group

# Axis Indexes with Duplicate Labels
obj = pd.Series(range(5), index=['a', 'a', 'b', 'b', 'c'])
obj.index.is_unique  # False, indicating non unique index label....
obj['a']  # o/p is [0,1] as we have two instances of a

## Summarizing and Computing Descriptive Statistics
# these fall into the category of reductions or summary statistics
df = pd.DataFrame([[1.4, np.nan], [7.1, -4.5], [np.nan, np.nan], [0.75, -1.3]], index=['a', 'b', 'c', 'd'],
                  columns=['one', 'two'])
df.sum()  # computes sum along row axis
df.sum(axis=1)  # computes sum along column axis
# NA values are excluded unless the entire slice (row or column in this case) is NA.
df.mean(axis='columns', skipna=False)  # skip na by default is TRUE
# axis: Axis to reduce over; 0 for DataFrame’s rows and 1 for columns
# skipna: Exclude missing values; True by default
# level: Reduce grouped by level if the axis is hierarchically indexed (MultiIndex)
## return indirect statistics like the index value where the minimum or maximum values are attained
df.idxmax()
df.cumsum()
# Describe & get statistics
df.describe()
# On non-numeric data, describe produces alternative summary statistics
obj = pd.Series(['a', 'a', 'b', 'c'] * 4)
obj.describe()
# count: no of non NA vales
# describe:Compute set of summary statistics for Series or each DataFrame column
# min, max:Compute minimum and maximum values
# argmin, argmax: Compute index locations (integers) at which minimum or maximum value obtained, respectively
# idxmin, idxmax : Compute index labels at which minimum or maximum value obtained, respectively
# quantile: Compute sample quantile ranging from 0 to 1
# sum: Sum of values
# mean:Mean of values
# median:Arithmetic median (50% quantile) of values
# mad:Mean absolute deviation from mean value
# prod: Product of all values
# var:Sample variance of values
# std:Sample standard deviation of values
# skew:Sample skewness (third moment) of values
# kurt:Sample kurtosis (fourth moment) of values
# cumsum:Cumulative sum of values
# cummin, cummax: Cumulative minimum or maximum of values, respectively
# cumprod:Cumulative product of values
# diff:Compute first arithmetic difference (useful for time series)
# pct_change: Compute percent changes

## Correlation and Covariance: Refer pg 160 for details

## Unique Values, Value Counts, and Membership
obj = pd.Series(['c', 'a', 'd', 'a', 'a', 'b', 'b', 'c', 'c'])
uniques = obj.unique()  # o/p is NON-SORTED array(['c', 'a', 'd', 'b'], dtype=object)
uniques.sort()  # sorting of unique value set
obj.value_counts()  # returns the frequency of each object. c3,a3,b2,d1
pd.value_counts(obj.values, sort=False)  # o/p freq as  c3,a3,b2,d1
# masking:
mask = obj.isin(['b', 'c'])  # returns True/ False
obj[mask]  # retruns indexed psotion of b,c

# CHAPTER6:_____________________________________________________________________
### Data Loading, Storage, and File Formats

## Parsing Functions in Pandas
# read_csv: Load delimited data from a file, URL, or file-like object; use comma as default delimiter
# read_table:Load delimited data from a file, URL, or file-like object; use tab ( '\t' ) as default delimiter
# read_fwf:Read data in fixed-width column format (i.e., no delimiters)
# read_clipboard: Version of read_table that reads data from the clipboard; useful for converting tables from web
# read_excel: Read tabular data from an Excel XLS or XLSX file
# read_hdf:Read HDF5 files written by pandas
# read_html:Read all tables found in the given HTML document
# read_json:Read data from a JSON (JavaScript Object Notation) string representation
# read_msgpack: Read pandas data encoded using the MessagePack binary format
# read_pickle: Read an arbitrary object stored in Python pickle format
# read_sas: Read a SAS dataset stored in one of the SAS system’s custom storage formats
# read_sql: Read the results of a SQL query (using SQLAlchemy) as a pandas DataFrame
# read_stata: Read a dataset from Stata file format
# read_feather: Read the Feather binary file format
pd.read_table('examples/ex1.csv', sep=',')  # o/p df as belwo for Reference
frame = pd.DataFrame(np.random.randn(3, 5), columns=['a', 'b', 'c', 'd', 'message'], header=None)
# defining index column as below
pd.read_csv('examples/ex2.csv', index_col='message')  # assigns message as INDEX in resulting df

# refer pg 170-175 onwards for further reading on reading files in python
## Writing Data to Text Format
data.to_csv('examples/out.csv')  # has comma separated CSV file
# Other delimiters can be used, of course like | etc
import sys

data.to_csv(sys.stdout, sep='|')
# handling missing data & replacing them with value say NULL here
data.to_csv(sys.stdout, na_rep='NULL')
# when no header for row, columns needed'
data.to_csv(sys.stdout, index=False, header=False)

# Working with Delimited Formats: pg 176
# JSON Data: pg 178
# XML and HTML: Web Scraping: pg 180
# Binary Data Formats: pg 183
# Using HDF5 Format: pg 185
# Interacting with Web APIs: pg 187
# Interacting with Databases: pg 188

# CHAPTER7:_____________________________________________________________________
### Data Cleaning and Preparation
# Basically involves Loading, Cleaning, Transforming, and Re-arranging.
# reported to take up 80% or more of an analyst’s time.

## Handling Missing Data
# For numeric data, pandas uses the floating-point value NaN (Not a Number) to represent missing data.
# We call this a sentinel value that can be easily detected
string_data = pd.Series(['aardvark', 'artichoke', np.nan, 'avocado'])
string_data.isnull()  # o/p is false, false, true, false
# IMP: NA is Not Available. In statistics applications, NA data may either be data that does not exist or that exists but was not observed (through problems with data collection
# The built-in Python None value is also treated as NA in object arrays
string_data[0] = None  # makes 0th index as None
string_data.isnull()  # o/p is true, false, true, false
# dropna:Filter axis labels based on whether values for each label have missing data, with varying thresholds for how much missing data to tolerate.
# fillna: Fill in missing data with some value or using an interpolation method such as 'ffill' or 'bfill' .
# isnull: Return boolean values indicating which values are missing/NA.
# notnull: Negation of isnull

## Filtering Out Missing Data
from numpy import nan as NA

data = pd.Series([1, NA, 3.5, NA, 7])  # o/p is [1,NaN,3.5,NaN,7]
data.dropna()  # o/p is [1,3.5,7]
# Equivakent to below:
data[data.notnull()]  # o/p is [1,3.5,7]

# In dataframes now
data = pd.DataFrame([[1., 6.5, 3.], [1., NA, NA], [NA, NA, NA], [NA, 6.5, 3.]])
cleaned = data.dropna()
# Passing how='all' will only drop rows that are all NA
data.dropna(how='all')  # removes 2 index ROW  as its all column has NaN
# to drop columns
data[4] = NA  # adds new column as name 4 and having value as nan
data.dropna(axis=1, how='all')  # drops column having all row as nan
## if you want to keep only rows containing a certain number of observations
df = pd.DataFrame(np.random.randn(7, 3))
df.iloc[:4, 1] = NA
df.iloc[:2, 2] = NA
df.dropna()  # drops all rows having at least one nan element
df.dropna(thresh=2)  # drop rows of nan to Threshold of 2 rows. Starts at 0th index from TOP.

# Filling In Missing Data: using above defined 'df'
df.fillna(0)  # fills all nan of df with 0. Can use other INTEGER as well
df.fillna('srimal')  # fills all nan of df with srimal
# fillna with a dict, you can use a different fill value for each column
df.fillna({1: 0.5, 2: 'srimal'})  # first column as 0.5 & second column as srimal
# fillna basically doesnot creates NEW, however you can do it by
df.fillna('amit', inplace=True)  # original df gets updated with nan as amit.
# Filling direction as here
df.fillna(method='ffill')  # forward fills nan in each column
df.fillna(method='bfill')  # backward fills nan in each column, from lowest to topmost column element
df.fillna(method='bfill', limit=2)  # backward fill with threshold of 2.
# fill nan with mean,median etc
df.fillna(df.mean())  # fills nan column with mean of each row
# value: Scalar value or dict-like object to use to fill missing values
# method:Interpolation; by default 'ffill' if function called with no other arguments
# axis:Axis to fill on; default axis=0
# inplace: Modify the calling object without producing a copy
# limit:For forward and backward filling, maximum number of consecutive periods to fill

## Data Transformation
## Removing Duplicates
data = pd.DataFrame({'k1': ['one', 'two'] * 3 + ['two'], 'k2': [1, 1, 2, 3, 3, 4, 4]})
data.duplicated()  # retruns true/false for duplicated key-value PAIR
# removing duplicates from data
data.drop_duplicates()
# lets add one more column
data['v1'] = range(7)
# drop duplicates based on selected column say k1
data.drop_duplicates(['k1'])
# Atribute the KEEP the last occurence of duplicate., with keep= 'last'
data.drop_duplicates(['k1', 'k2'], keep='last')

## Replacing Values
data = pd.Series([1., -999., 2., -999., -1000., 3.])
# The -999 values might be sentinel values for missing data. To replace these with NA.
data.replace(-999, np.nan)  # -999 is replaced by nan
data.replace([-999, -1000], np.nan)  # replacing multiple values
data.replace([-999, -1000], [np.nan, 1000])  # replacing multple values with each correspondence
data.replace({-999: np.nan, -1000: 0})  # replace with dictionary approach

# Renaming Axis Indexes
data = pd.DataFrame(np.arange(12).reshape((3, 4)), index=['Ohio', 'Colorado', 'Newyork'],
                    columns=['one', 'two', 'three', 'four'])
# using map method for index renaming
transform = lambda x: x[:4].upper()
data.index.map(transform)  # Index gets CAPITALIZED
# assign to index of data
data.index = data.index.map(transform)
# using RENAME: create a transformed version of a dataset without modifying the original
data.rename(index=str.title, columns=str.upper)
data.rename(index=str.title, columns=str.lower)
# renaming with Dictionary approach
data.rename(index={'OHIO': 'INDIANA'}, columns={'three': 'peekaboo'})
data.rename(index={'OHIO': 'INDIANA', 'NEWY': 'NEWDELHI'}, columns={'three': 'peekaboo'})

## Discretization and Binning
ages = [20, 22, 25, 27, 21, 23, 37, 31, 61, 45, 41, 32]
# lets decide bin as divide these into bins of 18 to 25, 26 to 35, 36 to 60, and finally 61 and older.
bins = [18, 25, 35, 60, 100]  # bins value decision
cats = pd.cut(ages, bins)
cats  # gives bin value range for each ages
cats.codes  # o/p is array([0, 0, 0, 1, 0, 0, 2, 1, 3, 2, 2, 1], dtype=int8)
cats.categories  # o/p is [(18, 25], (25, 35], (35, 60], (60, 100]]
pd.value_counts(cats)  # o/p is freq in each bin. [5,3,3,1]
# imp:a parenthesis means that the side is open, while the square bracket means it is closed (inclusive)
# pass your own bin names by passing a list or array to the labels
group_names = ['Youth', 'YoungAdult', 'MiddleAged', 'Senior']
pd.cut(ages, bins, labels=group_names)  # age bins are named now as per group_names
## pass an integer number of bins to cut instead of explicit bin edges
# compute equal-length bins based on the minimum and maximum values in the data.
data = np.random.rand(20)
pd.cut(data, 4, precision=2)  # gets 4 bins having precision upto 2 digits
## bins based on quantiles
# 1.closely related function, qcut , bins the data based on sample quantiles
data = np.random.randn(1000)  # Normally distributed
cats = pd.qcut(data, 4)  # Cut into EQUAL quartiles
pd.value_counts(cats)
# 2. passing your own quantiles, (numbers between 0 and 1, inclusive)
pd.qcut(data, [0, 0.1, 0.5, 0.9, 1.])
pd.value_counts(cats)

## Detecting and Filtering Outliers
# Filtering or transforming outliers is largely a matter of applying array operations
data = pd.DataFrame(np.random.randn(1000, 4))
data.describe()  # returns mean, med, avg,quantile etc.
# find values in one of the columns exceeding 3 in absolute value
col = data[2]  # selecting target column of data, based on col. index
col[np.abs(col) > 3]  # returns mod value greater than 3
# select all rows having a value exceeding 3 or –3
data[(np.abs(data) > 3).any(1)]

## Permutation and Random Sampling
df = pd.DataFrame(np.arange(5 * 4).reshape((5, 4)))
sampler = np.random.permutation(5)
sampler  # o/p is  array([4, 2, 1, 0, 3])
df.take(sampler)  # sampler used as INDEX in df to retrieve row columns df
# random selection of sample
df.sample(n=3)  # randomly selects 3 row df, all columns populated

## Computing Indicator/Dummy Variables
df = pd.DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'b'], 'data1': range(6)})
pd.get_dummies(df['key'])
# Refer 208 for more details..............

## String Manipulation
# String Object Methods
val = 'a,b,guido'
pieces = val.split(',')  # o/p is  ['a', 'b', 'guido']
val.split('u')  # o/p is ['a,b,g', 'ido']
# Imp: split is often combined with strip to trim whitespace (including line breaks)
# concatenate opeartion on above opeartion using '::'
'::'.join(pieces)  # o/p is 'a::b::guido'
# count returns the number of occurrences of a particular substring
val.count(',')  # o/p is 2 implying 2 occurences
# replacement
val.replace(',', '@')  # replace , with @
# count: Return the number of non-overlapping occurrences of substring in the string.
# endswith: Returns True if string ends with suffix.
# startswith: Returns True if string starts with prefix.
# join: Use string as delimiter for concatenating a sequence of other strings.
# index: Return position of first character in substring if found in the string; raises ValueError if not found.
# find: Return position of first character of first occurrence of substring in the string; like index , but returns –1 if not found.
# rfind: Return position of first character of last occurrence of substring in the string; returns –1 if not found.
# replace: Replace occurrences of string with another string.
# strip,rstrip,lstrip: Trim whitespace, including newlines; equivalent to x.strip() (and rstrip, lstrip , respectively) for each element.
# split:Break string into list of substrings using passed delimiter.
# lower: Convert alphabet characters to lowercase.
# upper: Convert alphabet characters to uppercase.
# casefold: Convert characters to lowercase, and convert any region-specific variable character combinations to a common comparable form.
# ljust,rjust: Left justify or right justify, respectively; pad opposite side of string with spaces (or some other fill character) to return a string with a minimum width.

## Regular Expressions
# Regular expressions provide a flexible way to search or match (often more complex) string patterns in text.
# A single expression, commonly called a REGEX, is a string formed according to the regular expression language.
# Python’s built-in re module is responsible for applying regular expressions to strings
# re module functions fall into three categories: PATTERN MATCHING, SUBSTITUTION, and SPLITTING
# eg:  suppose we wanted to split a string with a variable number of whitespace characters (tabs, spaces, and newlines).
import re

text = "foo   bar\t baz  \tqux"
# The regex describing one or more whitespace characters is \s+
re.split('\s+', text)  # spliting operation. o/p is ['foo', 'bar', 'baz', 'qux']
# alternate to above step
regex = re.compile('\s+')
regex.split(text)  # o/p is ['foo', 'bar', 'baz', 'qux']
# get a list of all patterns matching the regex,then use findall method
regex.findall(text)  # o/p is [' ', '\t ', ' \t']

# Note1: Creating a regex object with re.compile is highly recommended if you intend to apply the same expression to many strings; doing so will save CPU cycles
# Note2: match and search are closely related to findall .
# findall returns all matches in a string, search returns only the first match
# match only matches at the beginning of the string
text = """Dave dave@google.com
Steve steve@gmail.com
Rob rob@gmail.com
Ryan ryan@yahoo.com"""
pattern = r'[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,4}'  # defining expression for email id
import re

regex = re.compile(pattern, flags=re.IGNORECASE)  # re.ignorecase makes it case in-sensitive
# Using findall on the text produces a list of the email addresses:
regex.findall(text)  # finds allpattern in complied regex
# search returns a special match object for the first email address in the text
m = regex.search(text)
< _sre.SRE_Match
object;
span = (5, 20), match = 'dave@google.com' >
text[m.start():m.end()]  # serach text for m matches
# Refer 215 for more details on regex

## Vectorized String Functions in pandas
data = {'Dave': 'dave@google.com', 'Steve': 'steve@gmail.com', 'Rob': 'rob@gmail.com', 'Wes': np.nan}
data = pd.Series(data)
data.isnull()
# check whether each email address has 'gmail' in it with str.contains
data.str.contains('gmail')  # str.contains function to check gmail in elements
# using regex to find gmail string
matches = data.str.findall(pattern, flags=re.IGNORECASE)
# access elements in the embedded lists
matches.str.get(1)  # o/p is all nan
matches.str.get(0)  # o/p is all nan
# string slicing
data.str[:5]  # gets first 5 alphabets from each row element

# cat: Concatenate strings element-wise with optional delimiter
# contains: Return boolean array if each string contains pattern/regex
# count: Count occurrences of pattern
# extract: Use a regular expression with groups to extract one or more strings from a Series of strings; the result will be a DataFrame with one column per group
# endswith:Equivalent to x.endswith(pattern) for each element
# startswith:Equivalent to x.startswith(pattern) for each element
# findall:Compute list of all occurrences of pattern/regex for each string
# get: Index into each element (retrieve i-th element)
# isalnum:Equivalent to built-in str.alnum
# isalpha:Equivalent to built-in str.isalpha
# isdecimal:Equivalent to built-in str.isdecimal
# isdigit:Equivalent to built-in str.isdigit
# islower:Equivalent to built-in str.islower
# isnumeric:Equivalent to built-in str.isnumeric
# isupper:Equivalent to built-in str.isupper
# join:Join strings in each element of the Series with passed separator
# len: Compute length of each string
# lower, upper:Convert cases; equivalent to x.lower() or x.upper() for each element
# match: Use re.match with the passed regular expression on each element, returning matched groups as list
# pad: Add whitespace to left, right, or both sides of strings
# center: Equivalent to pad(side='both')
# repeat: Duplicate values (e.g., s.str.repeat(3) is equivalent to x * 3 for each string)
# replace: Replace occurrences of pattern/regex with some other string
# slice: Slice each string in the Series
# split: Split strings on delimiter or regular expression
# strip: Trim whitespace from both sides, including newlines
# rstrip: Trim whitespace on right side
# lstrip: Trim whitespace on left side

# CHAPTER8:_____________________________________________________________________
### Data Wrangling: Join, Combine, and Reshape
## Hierarchical Indexing : important feature of pandas that enables you to have multiple (two or more) index levels on an axis
# can be used for forming a pivot table
data = pd.Series(np.random.randn(9), index=[['a', 'a', 'a', 'b', 'b', 'c', 'c', 'd', 'd'], [1, 2, 3, 1, 3, 1, 2, 2, 3]])
# “gaps” in the index display mean “use the label directly above
data.index
# With a hierarchically indexed object, so-called partial indexing is possible to select subset of data
data['b']  # gets data of key index b
data['b':'d']  # gets data of key index from b to d inclusive
data.loc[['b', 'd']]  # # gets data of key index b & d ONLY
data.loc[:, 2]  # all FIRST index abcd , with 2nd INDEX element post comma
data.loc[:, 1:2]  # retruns using the same logic as ABOVE
# rearrange the multi-hierarchy data into a DataFrame using its unstack method
data.unstack()  # creates 4by3 df. 4 from 1st level & 3 from 2nd level index of multi-hierarchial df
data.unstack().stack()  # reversing the unstack operation

# With a DataFrame, either axis can have a hierarchical index:
frame = pd.DataFrame(np.arange(12).reshape((4, 3)),
                     index=[['a', 'a', 'b', 'b'], [1, 2, 1, 2]],
                     columns=[['Ohio', 'Ohio', 'Colorado'],
                              ['Green', 'Red', 'Green']])  # has row & coumn level multi-indexing
# assigning name to both level of index
frame.index.names = ['key1', 'key2']  # row level naming
frame.columns.names = ['state', 'color']  # column level naming
frame['Ohio']  # returns Ohio column entire data set
frame['Ohio', 'Red']  # returns Ohio column's Red columns entire data set
frame.loc['a']  # reurtns a row entire data set
frame.loc['a', 2]  # reurtns a row's 2 row's entire data set
frame.loc['a', '2']  # retruns error of labelling

# Reordering and Sorting Levels
# rearrange the order of the levels on an axis or sort the data by the values in one specific level
# The swaplevel takes two level numbers or names and returns a new object with the levels interchanged (but the data is otherwise unaltered)
frame.swaplevel('key1', 'key2')  # swaps key1 &2 positions
# sort_index , on the other hand, sorts the data using only the values in a single level
frame.sort_index(level=0)  # sorting based on 0th index . ie a here in this case
frame.sort_index(level=1)  # sorting based on 1st index . ie 1/2 here in this case

## Summary Statistics by Level
frame.sum(level='key2')
frame.sum(level='color', axis=1)

## Indexing with a DataFrame’s columns
frame = pd.DataFrame({'a': range(7), 'b': range(7, 0, -1), 'c': ['one', 'one', 'one', 'two', 'two', 'two', 'two'],
                      'd': [0, 1, 2, 0, 1, 2, 3]})
# create a new DataFrame [Hierarchial] using one or more of its columns as the index
frame2 = frame.set_index(['c', 'd'])  # c is 1st level & d is 2nd level of INDEXING
# By default the columns are removed from the DataFrame, to keep them pass drop argument as False
frame3 = frame.set_index(['c', 'd'], drop=False)
# reset_index , on the other hand, does the opposite of set_index.
# Keep drop is True, else Already Exists error will be thrown.
frame2.reset_index()  # drop is true & hence no error
frame3.reset_index()  # gives error of already exists

## Combining and Merging Datasets
# Number of ways Pandas offer to combine data as below:
# 1. pandas.merge:connects rows in DataFrames based on one or more keys. Like SQL join
# 2. pandas.concat: concatenates or “stacks” together objects along an axis.
# 3. combine_first: enables splicing together overlapping data to fill in missing values in one object with values from another

# Database-Style DataFrame Joins
df1 = pd.DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'a', 'b'], 'data1': range(7)})
df2 = pd.DataFrame({'key': ['a', 'b', 'd'], 'data2': range(3)})
##.. many to one join, merge uses the overlapping column names as the keys if we dont pass key argument
pd.merge(df1, df2)  # key not passed & hence OVERLAPPING key (a&b) is used.
pd.merge(df1, df2, on='key')  # key passed on merge
# If the column names are different in each object, you can specify them separately:
df3 = pd.DataFrame({'lkey': ['b', 'b', 'a', 'c', 'a', 'a', 'b'], 'data1': range(7)})
df4 = pd.DataFrame({'rkey': ['a', 'b', 'd'], 'data2': range(3)})
pd.merge(df3, df4, left_on='lkey', right_on='rkey')
# IMP Note: y default merge does an 'INNER' join
# outer join takes the union of the keys
pd.merge(df1, df2, how='outer')
# 'inner': Use only the key combinations observed in both tables
# 'left':Use all key combinations found in the left table
# 'right' Use all key combinations found in the right table
# 'output' Use all key combinations observed in both tables together

##.. Many-to-many joins form the Cartesian product of the rows.
pd.merge(df1, df2, on='key', how='left')
pd.merge(df1, df2, how='inner')
# To merge with multiple keys, pass a list of column names:
pd.merge(left, right, on=['key1', 'key2'], how='outer')
# treatment of overlapping column names
pd.merge(left, right, on='key1')
pd.merge(left, right, on='key1', suffixes=('_left', '_right'))  # adding suffix to overlapping columns

## Merging on Index
# pass left_index=True or right_index=True (or both) to indicate that the index should be used as the merge key
left1 = pd.DataFrame({'key': ['a', 'b', 'a', 'a', 'b', 'c'], 'value': range(6)})
right1 = pd.DataFrame({'group_val': [3.5, 7]}, index=['a', 'b'])
pd.merge(left1, right1, left_on='key', right_index=True)
# Since the default merge method is to intersect the join keys, you can instead form the union of them with an outer join
pd.merge(left1, right1, left_on='key', right_index=True, how='outer')
# merge on multi-hierarchial index : refer 233-236

## Concatenating Along an Axis
# IMP: interchangeably known as concatenation, binding, or stacking.
arr = np.arange(12).reshape((3, 4))
np.concatenate([arr, arr], axis=0)  # concatenate along Row, hence 6by4 is created
np.concatenate([arr, arr], axis=1)  # concatenate along Column, hence 3by8 is created
np.concatenate([arr, arr])  # concatenate along Row, hence 6by4 is created
# By default concat works along axis=0 , producing another Series as above.
# suppose 3 series having non-overlapping index
s1 = pd.Series([0, 1], index=['a', 'b'])
s2 = pd.Series([2, 3, 4], index=['c', 'd', 'e'])
s3 = pd.Series([5, 6], index=['f', 'g'])
s4 = pd.concat([s1, s3])
pd.concat([s1, s2, s3]  # produces another list glued along row axis. DEFAULT row axis
pd.concat([s1, s2, s3], axis=1)  # glueing along the colmn axis
# settings type of join
pd.concat([s1, s4], axis=1, join='inner')  # join along column set as INNER
# specify the axes to be used on the other axes with join_axes
pd.concat([s1, s4], axis=1, join_axes=[['a', 'c', 'b', 'e']])

## A potential issue is that the concatenated pieces are not identifiable in the result
result = pd.concat([s1, s1, s3], keys=['one', 'two', 'three'])
result.unstack()
pd.concat([s1, s2, s3], axis=1, keys=['one', 'two', 'three'])  # setting concat axis as COLUMN
# The same logic extends for data frame as well: Refer 239 for details

## Combining Data with Overlap
# case where two datasets whose indexes overlap in full or part.
a = pd.Series([np.nan, 2.5, np.nan, 3.5, 4.5, np.nan], index=['f', 'e', 'd', 'c', 'b', 'a'])
b = pd.Series(np.arange(len(a), dtype=np.float64), index=['f', 'e', 'd', 'c', 'b', 'a'])
b[-1] = np.nan
b[:-2].combine_first(a[2:])  # combines from part1 & part 2 respetively.if index duplcate, takes from first case.
# With DataFrames, combine_first does the same thing column by column
df1 = pd.DataFrame({'a': [1., np.nan, 5., np.nan], 'b': [np.nan, 2., np.nan, 6.], 'c': range(2, 18, 4)})
df2 = pd.DataFrame({'a': [5., 4., np.nan, 3., 7.], 'b': [np.nan, 3., 4., 6., 8.]})
df1.combine_first(df2)  # checks ocurence first & then fills nan with some matched value

## Reshaping and Pivoting
# basic operations for rearranging tabular data.
# Alternatively called "Reshaping and Pivoting".
# Hierarchical indexing provides a consistent way to rearrange data in a DataFrame
# Mainly there are TWO actions as:
# 1. stack: This “rotates” or pivots from the columns in the data to the rows
# 2. unstack: This pivots from the rows into the columns
data = pd.DataFrame(np.arange(6).reshape((2, 3)), index=pd.Index(['Ohio', 'Colorado'], name='state'),
                    columns=pd.Index(['one', 'two', 'three'], name='number'))
# use stack & pivots the columns into the rows, producing a Series
result1 = data.stack()  # columns becomes row
result2 = result.unstack()  # equivalent to original data
# IMP: By default the innermost level is stacked-unstacked. We can choose level as below:
result1.unstack(0)  # along row axis
result.unstack('state')  # along state
# Note: Unstacking might introduce missing data if all of the values in the level aren’t found in each of the subgroups
s1 = pd.Series([0, 1, 2, 3], index=['a', 'b', 'c', 'd'])
s2 = pd.Series([4, 5, 6], index=['c', 'd', 'e'])
data2 = pd.concat([s1, s2], keys=['one', 'two'])
data2.unstack()  # introduces missing values as nan
# Stacking filters out the missing data by DEFAULT
data2.unstack().stack()
data2.unstack().stack(dropna=False)  # passing argument if we want to keep nan
## When you unstack in a DataFrame, the level unstacked becomes the lowest level in the result
df = pd.DataFrame({'left': result, 'right': result + 5}, columns=pd.Index(['left', 'right'], name='side'))
df.unstack('state')
# When calling stack , we can indicate the name of the axis to stack
df.unstack('state').stack('side')

## Pivoting long to wide :refer 246
## Pivoting wide to long :refer 249

# CHAPTER9:_____________________________________________________________________
### Plotting and Visualization
# refer 253 - 287 for details..........


# CHAPTER10:_____________________________________________________________________
### Data Aggregation and Group Operations
# GroupBy Mechanics
# Hadley Wickham, coined the term split-apply-combine(SAC) for describing group operations.
# Stage1: data is split into groups based on one or more keys
# Stage2: a function is applied to each group, producing a new value.
# Stage3 :results of all those function applications are combined into a result object
df = pd.DataFrame(
    {'key1': ['a', 'a', 'b', 'b', 'a'], 'key2': ['one', 'two', 'one', 'two', 'one'], 'data1': np.random.randn(5),
     'data2': np.random.randn(5)})
# compute the mean of the data1 column using the labels from key1
grouped = df['data1'].groupby(df['key1'])
# Note: This groupby has not actually computed anything yet except for some intermediate data about the group key df['key1']
# to compute group means we can call the GroupBy’s mean method
grouped.mean()  # o/p is [a/b : -0.429/-1.324] for each key's mean
# Hence here data (a Series) has been aggregated according to the group key
# Now passing multiple arrays as a list
means = df['data1'].groupby([df['key1'], df['key2']]).mean()  # compute level wise mean
means.unstack()  # will unstack the above hierarchial frame
# grouping based on passing correct length of array, instead of keys
states = np.array(['Ohio', 'California', 'California', 'Ohio', 'Ohio'])
years = np.array([2005, 2005, 2006, 2005, 2006])
df['data1'].groupby([states, years]).mean()  # no of elelemts needs to EQUAL here, else indexing ERROR.

# Passing column name as group by key
df.groupby('key1').mean()  # takes unique element as key & perform grouping
df.groupby(['key1', 'key2']).mean()  # on multiple keys column
## GroupBy method SIZE, which returns a Series containing group sizes
df.groupby(['key1', 'key2']).size()  # returns size
# Imp: any missing values in a group key will be excluded from the result

## Iterating Over Groups
# GroupBy object supports iteration, generating a sequence of 2-tuples containing the group name along with the chunk of data
for name, group in df.groupby('key1'):
    print(name)
print(group)
# case of multiple keys, the first element in the tuple will be a tuple of key values
for (k1, k2), group in df.groupby(['key1', 'key2']):
    print((k1, k2))
print(group)
# handling pieces of data generated from above action
pieces = dict(list(df.groupby('key1')))
pieces['a']
pieces['b']
# IMP: By default groupby groups on axis=0 , but you can group on any of the other axes.
# Lets group the columns of our df here by dtype
df.dtypes  # float, float, object, object
grouped = df.groupby(df.dtypes, axis=1)  # gives description & needs printing as below
for dtype, group in grouped:
    print(dtype)
print(group)

## Selecting a Column or Subset of Columns
# Indexing a GroupBy object created from a DataFrame with a column name or array
# of column names has the effect of column subsetting for aggregation
df.groupby('key1')['data1']  # quick approach of grouping when compared to below method
df['data1'].groupby(df['key1'])  # same as above
df.groupby('key1')[['data2']]
df[['data2']].groupby(df['key1'])  # same as above
# for large datasets, it may be desirable to aggregate only a few columns.
df.groupby(['key1', 'key2'])[['data2']].mean()

## Grouping with Dicts and Series
# Grouping information may exist in a form other than an array
people = pd.DataFrame(np.random.randn(5, 5),
                      columns=['a', 'b', 'c', 'd', 'e'],
                      index=['Joe', 'Steve', 'Wes', 'Jim', 'Travis'])  # dataframe people created
people.iloc[2:3, [1, 2]] = np.nan  # Add a few NA values
# suppose I have a group correspondence for the columns and want to sum together the columns by group
mapping = {'a': 'red', 'b': 'red', 'c': 'blue', 'd': 'blue', 'e': 'red', 'f': 'orange'}
by_column = people.groupby(mapping, axis=1)
by_column.sum()
## The same functionality holds for Series, which can be viewed as a fixed-size mapping
map_series = pd.Series(mapping)
map_series
people.groupby(map_series, axis=1).count()
people.groupby(map_series, axis=1).sum()

## Grouping with Functions
# Any function passed as a group key will be called once per index value, with the return values being used as the group name
# Suppose you wanted to group by the length of the names in people dataframe
people.groupby(len).sum()
# Note: Mixing functions with arrays, dicts, or Series is not a problem as everything gets converted to arrays internally
key_list = ['one', 'one', 'one', 'two', 'two']
people.groupby([len, key_list]).min()

## Grouping by Index Levels
# aggregate using one of the levels of an axis index
columns = pd.MultiIndex.from_arrays([['US', 'US', 'US', 'JP', 'JP'], [1, 3, 5, 1, 3]], names=['cty', 'tenor'])
hier_df = pd.DataFrame(np.random.randn(4, 5), columns=columns)
# To group by level, pass the level number or name using the level keyword:
hier_df.groupby(level='cty', axis=1).count()

## Data Aggregation
# Aggregations refer to any data transformation that produces scalar values from arrays.
# count: Number of non-NA values in the group
# sum: Sum of non-NA values
# mean:Mean of non-NA values
# median: Arithmetic median of non-NA values
# std, var: Unbiased (n – 1 denominator) standard deviation and variance
# min, max: Minimum and maximum of non-NA values
# prod:Product of non-NA values
# first, last: First and last non-NA values
df = pd.DataFrame(
    {'key1': ['a', 'a', 'b', 'b', 'a'], 'key2': ['one', 'two', 'one', 'two', 'one'], 'data1': np.random.randn(5),
     'data2': np.random.randn(5)})
grouped = df.groupby('key1')
grouped['data1'].quantile(0.9)
## To use your own aggregation functions, pass any function that aggregates an array to the aggregate or agg method


def peak_to_peak(arr):
    return arr.max() - arr.min()


grouped.agg(peak_to_peak)
grouped.describe()
# Imp: Custom aggregation functions are generally much slower than the optimized functions
# becoz there is some extra overhead (function calls, data rearrangement) in con structing the intermediate group data chunks.

## Column-Wise and Multiple Function Application
tips = pd.read_csv('examples/tips.csv')  # reading csv file
tips['tip_pct'] = tips['tip'] / tips['total_bill']  # Add tip percentage of total bill
tips[:6]  # returning 6 rows
grouped = tips.groupby(['day', 'smoker'])  # grouping by day & smoker
# for descriptive stats, pass the name of the function as a string
grouped_pct = grouped['tip_pct']
grouped_pct.agg('mean')
# If you pass a list of functions or function names instead, you get back a DataFrame with column names taken from the functions
grouped_pct.agg(['mean', 'std', peak_to_peak])
# tuples in ordered marching
grouped_pct.agg([('foo', 'mean'), ('bar', np.std)])

# ************Refer from 299 - 317 for further readings

# CHAPTER11:_____________________________________________________________________
### Time Series- 1)Fixed & 2)Irregular frequency time Series
#  The simplest and most widely used kind of time series are those indexed by timestamp

## Date and Time Data Types and Tools
# The datetime, time, and calendar modules are the main places to start.
# The datetime.datetime type, or simply datetime, is widely used.
from datetime import datetime

now = datetime.now()  # o/p is datetime.datetime(2018, 7, 8, 8, 54, 2)
now.year, now.month, now.day  # o/p is (2018, 7, 8)
# timedelta represents the temporal difference between two datetime objects
delta = datetime(2011, 1, 7) - datetime(2008, 6, 24, 8, 15)
delta  # o/p is datetime.timedelta(926, 56700), first is days, second is seconds time
delta.days  # 926 days is o/p
delta.seconds  # 56700 seconds is o/p
# add (or subtract) a timedelta or multiple thereof to a datetime object
from datetime import timedelta

start = datetime(2011, 1, 7)
start + timedelta(12)  # o/p is datetime.datetime(2011, 1, 19, 0, 0)
start - 2 * timedelta(12)  # o/p is datetime.datetime(2010, 12, 14, 0, 0)
# date: Store calendar date (year, month, day) using the Gregorian calendar
# time: Store time of day as hours, minutes, seconds, and microseconds
# datetime: Stores both date and time
# timedelta: Represents the diﬀerence between two datetime values (as days, seconds, and microseconds)
# tzinfo: Base type for storing time zone information

## Converting Between String and Datetime
stamp = datetime(2011, 1, 3)
str(stamp)  # o/p is string  '2011-01-03 00:00:00'
stamp.strftime('%Y-%m-%d')  # o/p is '2011-01-03'

## Format Specification
# %Y Four-digit year
# %y Two-digit year
# %m Two-digit month [01, 12]
# %d Two-digit day [01, 31]
# %H Hour (24-hour clock) [00, 23]
# %I Hour (12-hour clock) [01, 12]
# %M Two-digit minute [00, 59]
# %S Second [00, 61] (seconds 60, 61 account for leap seconds)
# %w Weekday as integer [0 (Sunday), 6]
# %U Week number of the year [00, 53]; Sunday is considered the frst day of the week, and days before the frst Sunday of the year are “week 0”
# %W Week number of the year [00, 53]; Monday is considered the frst day of the week, and days before the frst Monday of the year are “week 0”
# %z UTC time zone oﬀset as +HHMM or -HHMM; empty if time zone naive
# %F Shortcut for %Y-%m-%d (e.g., 2012-4-18)
# %D Shortcut for %m/%d/%y (e.g., 04/18/12)

320

# CHAPTER1:_____________________________________________________________________
### Why NOT Python:
# Reason1: As Python is an interpreted programming language, in general most Python code will
# run substantially slower than code written in a compiled language like Java or C++.
# As programmer time is often more valuable than CPU time, many are happy to make
# this trade-off.
# Reason2: Python can be a challenging language for building highly concurrent, multithreaded
# applications, particularly applications with many CPU-bound threads. The reason for
# this is that it has what is known as the global interpreter lock (GIL), a mechanism that
# prevents the interpreter from executing more than one Python instruction at a time.

### WHY Python is popular:
# Python has a huge collection of libraries.
# Python is known as the beginner’s level programming language because of it simplicity and easiness.
# From developing to deploying and maintaining Python wants their developers to be more productive.
# Portability is another reason for huge popularity of Python.
# Python’s programming syntax is simple to learn and is of high level compared to C, Java, and C++.
# new applications can be developed by writing fewer lines of codes, the simplicity of Python has attracted many developers.

###TOP few python Libraries:
# TensorFlow
# Scikit-Learn
# Numpy
# Keras
# PyTorch
# LightGBM
# Eli5
# SciPy
# Theano
# Pandas

## Important packages to learn:
# 1.Numpy
# 2.Pandas
# 3.Matplotlib
# 4.Scipy
# 5.Scikit-learn: which includes
# Classification: SVM, nearest neighbors, random forest, logistic regression, etc.
# Regression: Lasso, ridge regression, etc.
# Clustering: k-means, spectral clustering, etc.
# Dimensionality reduction: PCA, feature selection, matrix factorization, etc.
# Model selection: Grid search, cross-validation, metrics
# Preprocessing: Feature extraction, normalization
# 6.Statsmodel: which includes
# Regression models: Linear regression, generalized linear models, robust linear models, linear mixed effects models, etc.
# Analysis of variance (ANOVA)
# Time series analysis: AR, ARMA, ARIMA, VAR, and other models
# Nonparametric methods: Kernel density estimation, kernel regression
# Visualization of statistical model results.
## ** NOTE: ** statsmodels is more focused on statistical inference, providing uncertainty estimates
# and p-values for parameters. scikit-learn, by contrast, is more prediction-focused.

# CHAPTER2:_____________________________________________________________________
### Python Basics: Refer pg 15 onwards for details

# CHAPTER3:______________________________________________________________________
### Built-in Data Structures, Functions, and Files ###

## TUPLE:
# tuple is a fixed-length, IMMUTABLE sequence of Python objects. While the objects stored in a tuple may be mutable themselves, once the tuple is
# created it’s not possible to modify which object is stored in each slot
tuple = 41, 15, 16  # simple tuple. o/p is (4,5,6)
nested_tuple = (4, 5, 6), (7, 8)  # o/p is ((4, 5, 6), (7, 8))
# convert any sequence or iterator to a tuple by invoking tuple :
tuple([4, 0, 2])  # [4, 0, 2] converted to tuple
tuple('Srimal')  # o/p is ('S', 'r', 'i', 'm', 'a', 'l')
# accessing element inside tuple
nested_tuple[0]  # o/p is (4, 5, 6)
# concatenate tuples using the + operator to produce longer tuples
conc_tuple = tuple + nested_tuple  # o/p is (41, 15, 16, (4, 5, 6), (7, 8))
# Multiplying a tuple by an integer
tuple * 2  # o/p is (41, 15, 16, 41, 15, 16). NA for add,sub,multiply etc.

## Unpacking tuples: assign to a tuple-like expression of variables
# Note: here size, coveringetc needs to be SAME for assigment, else error is thrown
a, b, c = tuple
b  # 15 which is in line with same sized tuple created earlier
(x, y, z), (m, n) = nested_tuple
z  # o/p is 6
# unpacking over sequence:
seq = [(11, 22, 33), (41, 52, 63), (27, 58, 69)]
for a, b, c in seq:
    print('a={2}, b={0}, c={1}'.format(a, b, c))  # based on index value
# unpacking with *rest to capture an arbitrarily long list of positional arguments:
values = 1, 2, 3, 4, 5
a, b, *rest = values  ## Availablein python 3, clubs all values other than a,b
a, b, *_ = values  ## Availablein python 3, discard all values other than a,b
a, b  # o/p is (1,2)
rest  # o/p is (3,4,5)
# count on tuple element
a = (1, 2, 2, 2, 3, 4, 2)
a.count(2)  # o/p is 4, frequency of 2 in a

## LIST:In contrast with tuples, lists are variable-length and their contents can be modified in-place
# define them using 1)square brackets [] or 2)using the list type function:
a_list = [2, 3, 7, None]  # list using sq. brackets
b_list = list(tup)  # where tup=(3,4,5) # list using LIST function
a_list[0]  # o/p is 2... accessing index based elements
# Adding/insert elements
a_list.append(99)  # o/p is [2, 3, 7, None, 99]. 99 is added to list
a_list.insert(1, 111)  # inserts 111 at 1stindex. o/p is [2, 111, 3, 7, None, 99]
# Removing elements: inverse operation to insert is pop
a_list.pop(1)  # removes 1st index 111. o/p is [2, 3, 7, None, 99]
a_list.remove(7)  # removes 7 fromlist. '' to pass string value
# Note : These operations takes only ONE ARGUMENT. we cant pass multiple values
a_list.append([99, 999])  # will add [99,999] to a_list here
## Check if a list contains a value using the in keyword
7 in a_list  # true as 7 removed coz of remove operation
77 not in a_list  # true as 77not in list
## Concatenating and combining lists
a_list + b_list  # o/p is [2, 3, None, 99, [99, 999], 3, 4, 5]
# extending list
a_list.extend([4, 'test', ['srimal']])  # o/p is [2, 3, None, 99, [99, 999], 4, 'test', ['srimal']]
## Note: EXTEND is faster & efficient than CONCATENATE +

# SORTING:
a = [7221, 22, 5222, 122, 311344]
a.sort()  # [22, 122, 5222, 7221, 311344]
a.sort(reverse=True)  # [311344, 7221, 5222, 122, 22]
a_list.sort()  # [None, 2, 3, 4, 55, 99, [99, 999], ['srimal'], 'test'] numeric, alphabetically...
## Sorting by key
a.sort(key=len)  # not applicable for INTEGER
b = ['saw', 'small', 'He', 'foxes', 'six']
b.sort(key=len)  # o/p is ['He', 'saw', 'six', 'small', 'foxes']
## Binary search and maintaining a sorted list
import bisect

bisect.bisect(a, 2)  # o/p is 0. INDEX is shown here. we should place 2 at oth index of list a.
bisect.insort(a, 2)  # directly place value in list. o/p is [2, 311344, 7221, 5222, 122, 22]
## Note: bisect module functions do not check whether the list is sorted, as doing so would be computationally expensive. Thus, using
# them with an unsorted list will succeed without error but may lead to incorrect results.
# TIP: Better to SORT first & then apply BISECT function.

## SLICING:  basic form consists of start:stop command.
# number of elements in the result is stop - start .
a[:3]  # o/p is [2, 311344, 7221]
# Slices can also be assigned to with a sequence:
a[2:4] = [45, 54, 78]  # o/p is [7221, 22, 45, 54, 78, 311344]. The [5222, 122] is REMOVED
# negative Indices
a[:-3]  # [7221, 22, 45]
a[-5:-3]  # [22, 45]...ie -3-(-5)
# A step can also be used after a second colon to, say, take every other element:
a[::2]  # o/p is [7221, 45, 78]. Start, End & Stride is 2. Start to End with stride of 2.
a[::-2]  # o/p is [311344, 54, 22], with Opposite direction of above

## Built-in Sequence Functions
# ENUMERATE:
# when iterating over a sequence to want to keep track of the index of the current item
# returns a sequence of (i, value) tuples
some_list = ['foo', 'bar', 'baz']
mapping = {}
for i, v in enumerate(some_list):
    mapping[v] = i  # concept in python3
# SORTED: retruns new sorted list from the elements of any sequence:
sorted('Amu Anu')  # o/p is [' ', 'A', 'A', 'm', 'n', 'u', 'u']
# ZIP:“pairs” up the elements of a number of lists, tuples, or other sequences to create a list of tuples
# structure needs to be similar for creating list.
zipped = zip(['Amit', 'Annu', 'Ashu'], [32, 27, 25], ['baroda', 'tarsali', 'Srimal'])
list(zipped)  # o/p is [('Amit', 32, 'baroda'), ('Annu', 27, 'tarsali'), ('Ashu', 25, 'Srimal')]
# Unzip: from ZIPPED block
pitchers = [('Nolan', 'Ryan'), ('Roger', 'Clemens'), ('Schilling', 'Curt')]
first_names, last_names = zip(*pitchers)
first_names  # ('Nolan', 'Roger', 'Schilling')
last_names  # ('Ryan', 'Clemens', 'Curt')
# REVERSED:
list(reversed(range(10)))
# DICTIONARY: A more common name for it is hash map or associative array
d1 = {'a': 'some value', 'b': [1, 2, 3, 4]}
d1['b']  # o/p is [1, 2, 3, 4]
# access, insert, or set elements
d1[7] = 'Srimal'  # o/p is  {7: 'Srimal', 'a': 'some value', 'b': [1, 2, 3, 4]}
# delete values either using the del keyword or the pop method
del d1[7]  # 7 is key of dict d1. o/p is  {'a': 'some value', 'b': [1, 2, 3, 4]}
d1.pop('a')  # drops key a from dict.o/p is {'b': [1, 2, 3, 4]}
# getting key value pair of dict{}
list(d1.keys())  # gets key only. o/p is ['a', 'b']
list(d1.values())  # gets value only. o/p is  ['some value', [1, 2, 3, 4]]
# merge one dict into another, basically UPDATE
d1.update({'b': 'foo', 'c': 12})  # o/p is {'a': 'some value', 'b': 'foo', 'c': 12}. b is updated & c is added.

## Creating dicts from sequences:
# Since a dict is essentially a collection of 2-tuples, the dict function accepts a list of 2-tuples
tuple_1 = range(5)
tuple_2 = reversed(range(5))
dict_from_2tuples = dict(zip(tuple_1, tuple_2))  # use ZIP to combine two tuples, else throws error.
dict_from_2tuples  # o/p is  {0: 4, 1: 3, 2: 2, 3: 1, 4: 0}
## Valid dict key types:
# In Dictionary, key is immutable & technically its hashability.
# check with has function key
hash('string')  # 6756482857350662598
hash((1, 2, (2, 3)))  # 1097636502276347782
hash((1, 2, [2, 3]))  # fails coz list are MUTABLE.
# To use a list as a key, one option is to convert it to a tuple, which can be hashed as long as its elements
d = {}
d[tuple([1, 2, 3])] = 5
d  # o/p is {(1, 2, 3): 5}

## SET: A set is an unordered collection of unique elements.They are like dict, keys only, no values
# A set can be created in two ways:
# 1) the set function
set1 = set([2, 2, 2, 1, 3, 3])  # o/p is unique {1, 2, 3}
# 2) set literal with curly braces
{2, 2, 2, 1, 3, 3}  # o/p is {1, 2, 3}
# Sets support mathematical set operations like union, intersection, difference, and symmetric difference
a = {1, 2, 3, 4, 5}
b = {3, 4, 5, 6, 7, 8}
# UNION: set of distinct elements occurring in either set
a.union(b)  # o/p is {1, 2, 3, 4, 5, 6, 7, 8}
a | b  # o/p is {1, 2, 3, 4, 5, 6, 7, 8}
# INTERSECTION: contains the elements occurring in both sets
a.intersection(b)  # o/p is {3, 4, 5}
a & b  # o/p is {3, 4, 5}
# ADD
a.add(99)  # adds 9. o/p is {1, 2, 3, 4, 5, 99}
# RESET: Reset the set a to an empty state, discarding all of its elements
a.clear()  # empty set is o/p
# REMOVE
a.remove(99)  # o/p is {1, 2, 3, 4, 5}
# POP: Remove an arbitrary element from the set a , raising KeyError if the set is empty
a.pop()  # removed 1 7 o/p is {2, 3, 4, 5}
# UPDATE: Set the contents of a to be the union of the elements in a and b
a.update(b)  # a is changed as union of a,b. o/p is {1, 2, 3, 4, 5, 6, 7, 8}. b remains UNCHANGED
a |= b  # same as above. alternative syntax
# INTERSECTION UPDATE: Set the contents of a to be the intersection of the elements in a and b
a.intersection_update(b)  # a becomes {3,4,5}, b remains UNCHANGED
a &= b  # same as above. alternative syntax
# DIFFERENCE : The elements in a that are not in b
a.difference(b)  # o/p is {1, 2}
a - b  # same as above. alternative syntax
# DIFFERENCE UPDATE: Set a to the elements in a that are not in b
a.difference_update(b)  # a becomes {1,2}
a -= b  # same as above. alternative syntax
# a.symmetric_difference(b): All of the elements in either a or b but not both
# a.symmetric_difference_update(b): Set a to contain the elements in either a or b but not both
# a.issubset(b): True if the elements of a are all contained in b
# a.issuperset(b): True if the elements of b are all contained in a
# a.isdisjoint(b): True if a and b have no elements in common
# EQUAL: Sets are equal if and only if their contents are equal:
{1, 2, 3} == {3, 2, 1}  # returns TRUE as o/p

## List, Set, and Dict COMPREHENSIONS
## List Comprehension:
# allows to concisely form a new list by filtering the elements of a collection, transforming the elements passing the filter in one concise expression.
# syntax: [expr for val in collection if condition]
# eg: filter out strings with length 2 or less and also convert them to uppercase
strings = ['a', 'as', 'bat', 'car', 'dove', 'python']
[x.upper() for x in strings if len(x) > 2]  # o/p is ['BAT', 'CAR', 'DOVE', 'PYTHON']
## Dict Comprehension:
# syntax: dict_comp = {key-expr : value-expr for value in collection if condition}
## Set Comprehensions:
# syntax like list with CURLY: set_comp = {expr for value in collection if condition}
# eg: create a lookup map of these strings to their locations in the list:
{val: index for index, val in enumerate(strings)}  # {'a': 0, 'as': 1, 'bat': 2, 'car': 3, 'dove': 4, 'python': 5}
## Nested list comprehensions
# eg1: get a single list containing all names with two or more e ’s in them.
name_list = [['John', 'Emily', 'Michael', 'Mary', 'Steven'], ['Maria', 'Juan', 'Javier', 'Natalia', 'Pilar']]
[name for names in name_list for name in names if name.count('e') >= 2]  # o/p is ['Steven']
# eg2:“flatten” a list of tuples of integers into a simple list of integers:
some_tuples = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]
flattened = [x for tup in some_tuples for x in tup]
flattened  # o/p is  [1, 2, 3, 4, 5, 6, 7, 8, 9]

### FUNCTIONS:
# If Python reaches the end of a function without encountering a return statement, None is returned automatically.
# Each function can have positional arguments and keyword arguments.
# keyword argument: specify default values or optional arguments
func(x, y, z=6)  # here x,y are Positional & z is Keyword
# main restriction on function arguments is that the keyword arguments must follow the positional arguments (if any)

## Namespaces, Scope, and Local Functions
## NAMESPACE: an alternative and more descriptive name describing a variable scope in Python is a namespace
# Functions can access variables in two different scopes: global and local.
# The LOCAL namespace is created when the function is called and immedi ately populated by the function’s arguments. After the function is finished, the local namespace is destroyed
# Assigning variables outside of the function’s scope is possible, but those variables must be declared as global via the global keyword:
a = None


def bind_a_variable():
    global a


a = []
bind_a_variable()


# Returning Multiple Values
# functions returns only one o/p. but with python its possible to get multiple value
# eg1.
def f():
    a = 5


b = 6
c = 7
return a, b, c


# eg2.
def f():
    a = 5


b = 6
c = 7
return {'a': a, 'b': b, 'c': c}
# eg3.data cleansing operation for states String
states = ['Alabama ', 'Georgia!', 'Georgia', 'georgia', 'FlOrIda', 'south carolina##', 'West virginia?']


def remove_punctuation(value):
    return re.sub('[!#?]', '', value)


clean_ops = [str.strip, remove_punctuation, str.title]


def clean_strings(strings, ops):
    result = []


for value in strings:
    for function in ops:
        value = function(value)
        result.append(value)
return result

## Anonymous (Lambda) Functions
# writing functions consisting of a single statement, the result of which is the return value.
g = lambda x: x * 2  # retruns twice the value passed in function
g(5)  # returns 5*2 as output here
# eg2.sort a collection of strings by the number of distinct letters in each string
strings = ['foo', 'card', 'bar', 'aaaa', 'abab']
strings.sort(key=lambda x: len(set(list(x))))
strings  # ['aaaa', 'foo', 'abab', 'bar', 'card']


## Currying: Partial Argument Application
# means deriving new functions from existing ones by partial argument application
def add_numbers(x, y):
    return x + y


# derive new function from add_numbers which add 555
add_five = lambda y: add_numbers(555, y)  # add_numbers is said to be curried
# Alternatively in python, The built-in functools module can simplify this process using the partial function:
from functools import partial

add_five = partial(add_numbers, 5)  # either x or y is assigned 5, remaning argument passed with add)five function.
add_five(6)  # gives 11 as o/p here

## Generators:means of the iterator protocol, a generic way to make objects iterable
some_dict = {'a': 1, 'b': 2, 'c': 3}
for key in some_dict:
    print(key)
# same to achieve with ITERATOR
dict_iterator = iter(some_dict)
list(dict_iterator)  # o/p is  ['a', 'b', 'c']


## Difference: Whereas normal functions execute and return a single result at a time, generators return a sequence of
# multiple results lazily, pausing after each one until the next one is requested
# To create a generator, use the YIELD keyword instead of return in a function:
def squares(n=10):
    print('Generating squares from 1 to {0}'.format(n ** 2))


for i in range(1, n + 1):
    yield i ** 2
gen = squares()  # assigning the function to gen variable
# It is not until you request elements from the generator that it begins executing its code:
for x in gen:
    print(x, end=' ')  # o/p is 1 4 9 16 25 36 49 64 81 100

## ITERTOOLs module
import itertools

first_letter = lambda x: x[0]
names = ['Alan', 'Adam', 'Wes', 'Will', 'Albert', 'Steven']
for letter, names in itertools.groupby(names, first_letter):
    print(letter, list(names))  # names is a generator


## Exception handling
# The code in the except part of the block will only be executed if float(x) raises an exception
def attempt_float(x):
    try:
        return float(x)
    except:
        return x


### Files and the Operating System. refer pg 80 onwards of 541


# CHAPTER4:______________________________________________________________________
### NumPy Basics: Arrays and Vectorized Computation ###
# NumPy is a Python package which stands for ‘Numerical Python’.
# It is the core library for scientific computing, which contains a powerful n-dimensional array object, provide tools for integrating C, C++ etc.
# It is also useful in linear algebra, random number capability etc.
# NumPy array can also be used as an efficient multi-dimensional container for generic data.

# -----------------------------------------------------------------------------------------------------------------------
## ARRAY:  is a data structure consisting of a collection of elements, each identified by at least one array index or key.
## An array is stored such that the position of each element can be computed from its index tuple by a mathematical formula.

## LIST: It’s a collection of items (called nodes) ordered in a linear sequence

##List vs Array:
# A list is a different kind of data structure from an array.
# The biggest difference is in the idea of direct access Vs sequential access. Arrays allow both; direct and sequential access, while lists allow only sequential access. And this is because the way that these data structures are stored in memory.
# In addition, the structure of the list doesn’t support numeric index like an array is. And, the elements don’t need to be allocated next to each other in the memory like an array is.
# --------------------------------------------------------------------------------------------------------------------
# NumPy Array: Numpy array is a powerful N-dimensional array object which is in the form of rows and columns.
# We can initialize numpy arrays from nested Python lists and access it elements.

## Python Numpy array instead of a list because of the below three reasons:
# Less Memory
# Fast
# Convenient

# NumPy by itself does not provide modelling or scientific functionality, having an understanding of NumPy arrays and array-oriented computing will help you use tools with array-oriented semantics, like pandas
# NumPy-based algorithms are generally 10 to 100 times faster (or more) than their pure Python counterparts and use significantly less memory.
my_arr = np.arange(1000000)
my_list = list(range(1000000))
# let’s multiply each sequence by 2 & get execution time
% time
for _ in range(10): my_arr2 = my_arr * 2  # time taken is 76.1 ms
% time
for _ in range(10): my_list2 = [x * 2 for x in my_list]  # time taken is 1.34 s

## NumPy ndarray: A Multidimensional Array Object
data = np.random.randn(2, 3)
data.shape  # (2,3) data
data.dtype  # dtype('float64'), which is the default data type
## Creating ndarrays
arr1 = np.array([6, 7.5, 8, 0, 1])  # arr1 is array([ 6. ,  7.5,  8. ,  0. ,  1. ])
arr2 = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
arr2.shape  # shape is (2,4) ie 2 blocks having 4 elements each
arr2.ndim  # o/p is 2
## Creating arrays with ZERo & ONES
np.zeros(10)  # o/p is array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])
np.zeros((3, 6))  # o/p is full of zero, in 3 rows & 6columns
np.empty((2, 3, 2))  # empty creates an array without initializing its values to any particular value.
# Imp: It’s not safe to assume that np.empty will return an array of all zeros. In some cases, it may return uninitialized “garbage” values
# array: Convert input data (list, tuple, array, or other sequence type) to an ndarray either by inferring a dtype or explicitly specifying a dtype; copies the input data by default
# asarray: Convert input to ndarray, but do not copy if the input is already an ndarray
# arange: Like the built-in range but returns an ndarray instead of a list
# ones,ones_like: Produce an array of all 1s with the given shape and dtype; ones_like takes another array and produces a ones array of the same shape and dtype
# zeros,zeros_like: : Like ones and ones_like but producing arrays of 0s instead
# empty,empty_like: Create new arrays by allocating new memory, but do not populate with any values like ones and zeros
# full,full_like: Produce an array of the given shape and dtype with all values set to the indicated “fill value”Produce an array of the given shape and dtype with all values set to the indicated “fill value”full_like takes another array and produces a filled array of the same shape and dtype
# eye, identity: Create a square N × N identity matrix (1s on the diagonal and 0s elsewhere)

## Data Types for ndarrays
# data type or dtype is a special object containing the information (or metadata, data about data)
arr = np.array([1, 2, 3], dtype=np.int32)
arr.dtype  # dtype('int32'). defualt is int64.
## convert or CAST an array from one dtype to another using ndarray’s astype method
# eg1. convert from int32 to float64
float_arr = arr.astype(np.float64)  # converts aa from int32 to float 64.
float_arr.dtype  # dtype('float64')
# eg2. convert from float64 to int32
arr1 = np.array([3.7, -1.2, -2.6, 0.5, 12.9, 10.1])  # data type is dtype('float64')
int_arr = arr1.astype(np.int32)  # (dtype=int32) from float64.
# eg3. convert from string to float
arr2 = np.array(['1.25', '-9.6', '42'], dtype=np.string_)  # its numeric data type
str_arr = arr2.astype(float)
str_arr.dtype  # from string to dtype('float64')
# eg4. use another array’s dtype attribute
int_array = np.arange(10)  # dtype('int64')
calibers = np.array([.22, .270, .357, .380, .44, .50], dtype=np.float64)  # float64
int_array.astype(calibers.dtype)  # convert int_array AS calibers data type. its now float64

## Arithmetic with NumPy Arrays; thanks to VECTORIZATION
# Arrays are important because they enable you to express batch operations on data without writing any for loops. NumPy users call this vectorization.
# Any arithmetic operations between equal-size arrays applies the operation element-wise.
arr = np.array([[1., 2., 3.], [4., 5., 6.]])
arr * arr, arr + arr, arr - arr, arr / arr
# IMP: Operations between differently sized arrays is called BROADCASTING.

## Basic Indexing and Slicing: start & stop, ends with diff of (stop-start)
# eg: 1d arrays:
arr1d = np.arange(10)  # array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
arr1d_slice0 = arr1d[2:8:3]  # o/p is  array([2, 5]), JUMP of 3..
arr1d_slice1 = arr1d[5:8]  # o/p is array([5, 6, 7])
arr1d_slice2 = arr1d[:5]  # o/p is array([0, 1, 2, 3, 4])
arr1d_slice3 = arr1d[3:]  # o/p is array([3, 4, 5, 6, 7, 8, 9])
arr1d_slice4 = arr1d[-3:]  # o/p is array([7, 8, 9])
# array slices are VIEWS on the original array. This means that the data is not copied, and any modifications to the view will be reflected in the source array.
# Slice's DIMENSION remain same as parent array.
# assigning values to sliced part
arr1d_slice[:] = 454  # chnages all value of index 5,6,7 to 454
# eg2. 2d arrays:
arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
arr2d[2]  # o/p is array([7, 8, 9])
arr2d[2, 2]  # o/p is 9, row column basis value is PICKED
arr2d[2][2]  # o/p is 9, Same as above
arr2d[:2]  # o/p is array([[1, 2, 3], [4, 5, 6]])
arr2d[1:2]  # o/p is array([[4, 5, 6]])
arr2d[:-2]  # o/p is array([[1, 2, 3]])
# pass multiple slices just like you can pass multiple indexes
arr2d[:2, 1:]  # first take block & then elements. o/p is array([[2, 3],[5, 6]])
arr2d[:2, 2]  # o/p is array([3, 6])
arr2d[:, :1]  # o/p is array([[1],[4],[7]])
# eg3. 3d arrays:
arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
# if you omit later indices, the returned object will be a lower dimensional ndarray consisting of all the data along the higher dimensions
arr3d[0]  # array([[1, 2, 3], [4, 5, 6]])
arr3d[1]  # array([[ 7,  8,  9], [10, 11, 12]])
arr3d[2]  # index 2 is out of bounds for axis 0 with size 2
arr3d[0, 0]  # array([1, 2, 3])
arr3d[1, 0]  # array([7, 8, 9])

## Boolean Indexing
names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
data = np.random.randn(7, 4)  # matrix of 7 row & 4 columns
# lets say each name corresponds to a row in the data array
names == 'Bob'  # [ True, False, False,  True, False, False, False]
# boolean array can be passed when indexing the array:
data[names == 'Bob']  # retruns data for name = 'Bob' ONLY. for TRUE, it returns values
data[names == 'Bob', 2:]  # same as aboe, with column 2nd index onwards
data[names != 'Bob']  # reruens Non 'Bob' data, 5rows & 4columns
# selecting multiple casing
mask = (names == 'Bob') | (names == 'Will')
data[mask]  # gets 'Bob OR Will' data
data[~mask]  # gets NEGATE of above implementation.
# Note:
# 1. Selecting data from an array by boolean indexing always creates a copy of the data, even if the returned array is unchanged
# 2.The Python keywords and and or do not work with boolean arrays. Use & (and) and | (or) instead.

## Fancy Indexing:
# is a term adopted by NumPy to describe indexing using integer arrays
arr = np.empty((8, 4))
for i in range(8):
    arr[i] = i
# select out a subset of the rows in a particular order: simply pass a list or ndarray of integers specifying the desired order
arr[[4, 3, 0, 6]]
# Reshaping
arr2 = np.arange(32).reshape((8, 4))  # reshaping is possible on MULTIPLE values ONLY.
arr2[[1, 5, 7, 2], [0, 3, 1, 2]]  # indexing values from arr2. o/p is array([ 4, 23, 29, 10])
arr2[[1, 5, 7, 2]][:, [0, 3, 1, 2]]  # 4by4 row,col based on order as set, : says full matrix view
arr2.reshape(4, 8)  # array of 4 rows & 8 columns

## Transposing Arrays and Swapping Axes
# Transposing is a special form of reshaping that similarly returns a view on the under lying data without copying anything
arr = np.arange(15).reshape((3, 5))
arr.T
# computing the inner matrix product using np.dot
np.dot(arr.T, arr)  # rows of 1st & columns of 2nd becomes dimension of .dot product
np.dot(arr, arr.T)
# For higher dimensional arrays, transpose will accept a tuple of axis numbers to permute the axes
arr = np.arange(16).reshape((2, 2, 4))
arr.transpose((1, 0, 2))
# Swapping: swapaxes , which takes a pair of axis numbers and switches the indicated axes to rear‐ range
# swapaxes returns a VIEW on the data without making a copy.
arr = np.arange(16).reshape((2, 2, 4))
arr.swapaxes(1, 2)

## Universal Functions: Fast Element-Wise Array Functions
# universal function, or ufunc, is a function that performs element-wise operations on data in ndarrays.
# eg1. UNARY : taking just one i/p
np.sqrt(np.arange(5))  # array([ 0.,1.,1.41421356,1.73205081,2.])
np.exp(np.arange(5))  # array([1.,2.71828183,7.3890561,20.08553692,54.59815003])
# also called UNARY ufuncs. Others, such as add or maximum , take two arrays (BINARY ufuncs) and return a single array as the result
# abs, fabs: Compute the absolute value element-wise for integer, floating-point, or complex values
# sqrt: Compute the square root of each element (equivalent to arr ** 0.5 )
# square: Compute the square of each element (equivalent to arr ** 2 )
# exp: Compute the exponent e x of each element
# log, log10,log2, log1p: Natural logarithm (base e), log base 10, log base 2, and log(1 + x), respectively
# sign: Compute the sign of each element: 1 (positive), 0 (zero), or –1 (negative)
# ceil: Compute the ceiling of each element (i.e., the smallest integer greater than or equal to that number)
# floor: Compute the floor of each element (i.e., the largest integer less than or equal to each element)
# rint: Round elements to the nearest integer, preserving the dtype
# modf: Return fractional and integral parts of array as a separate array
# isnan: Return boolean array indicating whether each value is NaN (Not a Number)
# isfinite, isinf: Return boolean array indicating whether each element is finite (non- inf , non- NaN ) or infinite,respectively
# cos, cosh, sin,sinh, tan, tanh:Regular and hyperbolic trigonometric functions
# arccos, arccosh,arcsin, arcsinh,arctan, arctanh :Inverse trigonometric functions
# logical_not: Compute truth value of not x element-wise (equivalent to ~arr ).
# eg2. BINARY" taking just 2 i/p
x = np.random.randn(8)
y = np.random.randn(8)
z = np.random.randn(6)
np.maximum(x, y)  # o/p is element wise max
np.minimum(x, y)  # o/p is element wise min
np.minimum(x, z)  # broadcast error coz of x,z shape issue
# add: Add corresponding elements in arrays
# subtract: Subtract elements in second array from first array
# multiply: Multiply array elements
# divide, floor_divide: Divide or floor divide (truncating the remainder)
# power: Raise elements in first array to powers indicated in second array
# maximum, fmax: Element-wise maximum; fmax ignores NaN
# minimum, fmin: Element-wise minimum; fmin ignores NaN
# mod: Element-wise modulus (remainder of division)
# copysign: Copy sign of values in second argument to values in first argument
# greater, greater_equal,less, less_equal,equal, not_equal: Perform element-wise comparison, yielding boolean array (equivalent to infix operators >, >=, <, <=, ==, != )
# logical_and,logical_or, logical_xor: Compute element-wise truth value of logical operation (equivalent to infix operators & |, ^ )
# eg3. POLYNOMIAL: taking multiple i/p
# modf returns the fractional and integral parts of a floating-point array
arr = np.random.randn(3) * 5
remainder, whole_part = np.modf(arr)
remainder  # array([ 0.78231947, -0.74935188,  0.07131087])
whole_part  # array([ 1., -7.,  1.])
# eg4. OPTIONAL
np.sqrt(arr)  # array([ 1.33503538,nan,1.03504148])

## Array-Oriented Programming with Arrays
# practice of replacing explicit loops with array expressions is commonly referred to as vectorization
# vectorized array operations will often be one or two (or more) orders of magnitude faster than their pure Python equivalents
import numpy as np

## Mathematical and Statistical Methods
# can use aggregations (often called reductions) like sum , mean , and std (standard deviation) either by calling the array instance method or using the top-level NumPy function.
arr = np.random.randn(5, 4)
arr.mean()
np.mean(arr)
arr.sum()
# Functions like mean and sum take an optional axis argument that computes the statis tic over the given axis
arr.mean(axis=1)  # along column side
arr.sum(axis=0)  # along row side
# Other methods like cumsum and cumprod do not aggregate, instead producing an array of the intermediate results:
arr1d = np.array([0, 1, 2, 3, 4, 5, 6, 7])
arr1d.cumsum()  # o/p is array([ 0,  1,  3,  6, 10, 15, 21, 28])
# now on multi-dimensional array
arr2d = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
arr2d.cumsum(axis=0)  # o/p is array([[ 0,  1,  2],[ 3,  5,  7],[ 9, 12, 15]])
arr2d.cumprod(axis=1)  # o/p is array([[  0,   0,   0],[  3,  12,  60], [  6,  42, 336]])
# sum: Sum of all the elements in the array or along an axis; zero-length arrays have sum 0
# mean: Arithmetic mean; zero-length arrays have NaN mean
# std, var: Standard deviation and variance, respectively, with optional degrees of freedom adjustment (default denominator n )
# min, max: Minimum and maximum
# argmin, argmax: Indices of minimum and maximum elements, respectively
# cumsum: Cumulative sum of elements starting from 0
# cumprod: Cumulative product of elements starting from 1

## Methods for Boolean Arrays
# Boolean values are coerced to 1 ( True ) and 0 ( False ) in the preceding methods. Thus, sum is often used as a means of counting True values in a boolean array:
arr = np.random.randn(100)
(arr > 0).sum()  # Number of positive values. o/p is 37
# Any & All methods: Check for True instances
bools = np.array([False, False, True, False])
bools.any()  # checks if ANY of the bools is True
bools.all()  # checks if ALL of the bools are True
# Note: methods also work with non-boolean arrays, where non-zero elements evaluate to True .
arr1 = np.array([6, 4, 0, 8, 0, 1, 122, 0, 9, 0])
arr1.any()  # gets true since one non zero element present
arr1.all()  # gets false since non zero elements are present

## Sorting
arr1 = np.random.randn(10)
arr1.sort()  # sorted arr is returned
arr2 = np.random.randn(5, 3)
arr2.sort(1)  # column wise sorting. 5by3 array
arr2.sort(0)  # row wise sorting. 5by3 array
# Note: top-level method np.sort returns a sorted copy of an array instead of modifying the array in-place
np.sort(arr2, axis=0)  # same as above, but COPY is returned

## Unique and Other Set Logic: for one-dimensional ndarrays
names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
np.unique(names)  # ['Bob', 'Joe', 'Will']
# unique(x): Compute the sorted, unique elements in x
# intersect1d(x, y): Compute the sorted, common elements in x and y
# union1d(x, y): Compute the sorted union of elements
# in1d(x, y):Compute a boolean array indicating whether each element of x is contained in y
# setdiff1d(x, y): Set difference, elements in x that are not in y
# setxor1d(x, y): Set symmetric differences; elements that are in either of the arrays, but not both

### File Input and Output with Arrays:
# np.save and np.load are the core functionality
# NumPy is able to save and load data to and from disk either in text or binary format.
# Arrays are saved by default in an uncompressed raw binary format with file extension .npy
arr = np.arange(100)
np.save('some_array_file', arr)  # saving arr in some_array_file, ext is .npy
np.load('some_array_file.npy')  # loading arr from saved file format
# saving multiple arrays
np.savez('array_archive.npz', a=arr, b=arr)
arch = np.load('array_archive.npz')
arch['a']  # retrieving elements of arr a by loading..
# saving file in Compressed format
np.savez_compressed('arrays_compressed.npz', a=arr, b=arr)

## Pseudorandom Number Generation
# these are pseudorandom numbers because they are generated by an algo rithm with
# deterministic behavior based on the seed of the random number generator.
np.random.seed(10)
arr = np.random.rand(5)
# seed: Seed the random number generator
# permutation: Return a random permutation of a sequence, or return a permuted range
# shuffle: Randomly permute a sequence in-place
# rand: Draw samples from a uniform distribution
# randint: Draw random integers from a given low-to-high range
# randn: Draw samples from a normal distribution with mean 0 and standard deviation 1 (MATLAB-like interface)

## Example: Random Walks: Refer pg 119 for details

# CHAPTER5:_____________________________________________________________________
### Getting Started with pandas
# The biggest difference is that pandas is designed for working with tabular or heterogeneous data.
# NumPy, by contrast, is best suited for working with homogeneous numerical array data.

# ----------------------------------------------------------------------------------------------------------------
# Pandas is used for data manipulation, analysis and cleaning. Python pandas is well suited for different kinds of data, such as:
# Tabular data with heterogeneously-typed columns
# Ordered and unordered time series data
# Arbitrary matrix data with row & column labels
# Unlabelled data
# Any other form of observational or statistical data sets

# Python operation: Slicing, Merging and Joining, Concatenation, Index Change, Change Column Header, Data Munging

import pandas as pd
from pandas import Series, DataFrame  # import Series and DataFrame into the local namespace

## Pandas Data Structure
# Series: is a one-dimensional array-like object containing a sequence of values (of similar types to NumPy types) and an associated array of data labels, called its index.
obj = pd.Series([4, 7, -5, 3])
# Index: default one consisting of the integers 0 through N - 1 (where N is the length of the data) is created
obj.values  # o/p is array([ 4,  7, -5,  3])
obj.index  # o/p is  RangeIndex(start=0, stop=4, step=1)
# array with predefined custom index
obj2 = pd.Series([4, 7, -5, 3], index=['d', 'b', 'a', 'c'])
obj2.index  # o/p is Index(['d', 'b', 'a', 'c'], dtype='object')
obj2[['c', 'a', 'd']]  # accessing multiple index element. o/p is 3,-5,4
obj2[obj2 < 0]  # o/p is -5

# Another way to think about a Series is as a fixed-length, ordered dict.
sdata = {'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah': 5000}  # dictionary
obj3 = pd.Series(sdata)  # dictionary to series
obj3.values  # o/p is array([35000, 16000, 71000,  5000])
obj3.index  # o/p is Index(['Ohio', 'Oregon', 'Texas', 'Utah'], dtype='object')
# Series from other index
index_new = ['California', 'Ohio', 'Oregon', 'Texas', 'Delhi']
obj4 = pd.Series(sdata, index=index_new)  # dictionary value ONLY used in series, not keys. California has NaN.
# Only key is maintained in final o/p.
# Handling missing data: missing” or “NA” interchangeably to refer to missing data
# isnull and notnull functions in pandas should be used to detect missing data
pd.isnull(obj4)
pd.notnull(obj4)
obj3 + obj4  # has all KEYS from both Series
# Naming Series object & its index
obj4.name = 'population'
obj4.index.name = 'state'
obj4

## DataFrame
# Ways to construct a DataFrame
# eg1: from a dict of equal-length lists or NumPy arrays
data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada', 'Nevada'], 'year': [2000, 2001, 2002, 2001, 2002, 2003],
        'pop': [1.5, 1.7, 3.6, 2.4, 2.9, 3.2]}
frame = pd.DataFrame(data)
# specify sequence of columns
pd.DataFrame(data, columns=['year', 'state', 'pop'])
pd.DataFrame(data, columns=['year', 'state', 'pop', 'unknown'])  # unknown column appears as NaN.
# retrieving column values
frame['state']
## retreiving rows with LOC/iLOC: Rows can also be retrieved by position or name with the special loc attribute
frame.loc[1, 'state']  # specify row 1 data , State column element
frame.loc[2]  # specify row 1 data , each column element
frame.iloc[3]  # specify all data of row index 2

# first add a new column of boolean values where the state column equals 'Ohio'
frame['eastern'] = frame.state == 'Ohio'
frame['eastern'] = 'NDLS'  # create new column & assigning value NDLS
del frame['eastern']  # will delete eastern column from frame
frame.columns  # returns the column name of data frame
frame.values  # retruns array of frame as o/p

## Index Objects
# pandas’s Index objects are responsible for holding the axis labels and other metadata (like the axis name or names
obj = pd.Series(range(3), index=['a', 'b', 'c'])
obj.index  # Index(['a', 'b', 'c'], dtype='object')
obj.index[1:]  # o/p is Index(['b', 'c'], dtype='object')
# Imp: Index objects are immutable and thus can’t be modified by the user
obj.index['a'] = 'AA'  # Error saying Index does not support mutable operations
# Sharing Index objects operations
labels = pd.Index(np.arange(3))
obj2 = pd.Series([1.5, -2.5, 0], index=labels)
obj2  # returns o/p as obj2 object
# NOTE: Unlike Python sets, a pandas Index can contain duplicate labels
dup_labels = pd.Index(['foo', 'foo', 'bar', 'bar'])  # duplicate index labels
# Selections with duplicate labels will select all occurrences of that label
# append: Concatenate with additional Index objects, producing a new Index
# difference: Compute set difference as an Index
# intersection:Compute set intersection
# union:Compute set union
# isin: Compute boolean array indicating whether each value is contained in the passed collection
# delete: Compute new Index with element at index i deleted
# drop: Compute new Index by deleting passed values
# insert: Compute new Index by inserting element at index i
# is_monotonic: Returns True if each element is greater than or equal to the previous element
# is_unique: Returns True if the Index has no duplicate values
# unique: Compute the array of unique values in the Index

## Essential Functionality
# Re-Indexing: to create a new object with the data conformed to a new index.
obj = pd.Series([4.5, 7.2, -5.3, 3.6], index=['d', 'b', 'a', 'c'])
# Calling reindex on this Series rearranges the data according to the new index, introducing missing values if any index values were not already present
obj2 = obj.reindex(['a', 'b', 'c', 'd', 'e', 'c'])  # missing index e will have NaN.
# Interpolation: interpolation or filling of values when reindexing.
obj3 = pd.Series(['blue', 'purple', 'yellow'], index=[0, 2, 4])
obj3.reindex(range(6), method='ffill')  # obj3 has 6 elements, with values Forward filled
# reindex can alter either the (row) index, columns, or both
frame1 = pd.DataFrame(np.arange(9).reshape((3, 3)), index=['a', 'c', 'd'], columns=['Ohio', 'Texas', 'California'])
frame2 = frame1.reindex(['a', 'b', 'c', 'd'])
# Reindexing with column keywords
states = ['Texas', 'Utah', 'California']
frame1.reindex(columns=states)  # earlier column index renamed to States value.
frame1.loc[['a', 'd'], states]  # reurns a,d row from States column
frame1.loc[['a', 'd'], 'Texas']  # returns a,d row from Texas column
# index:New sequence to use as index. Can be Index instance or any other sequence-like Python data structure. An Index will be used exactly as is without any copying.
# method: Interpolation (fill) method; 'ffill' fills forward, while 'bfill' fills backward.
# fill_value:Substitute value to use when introducing missing data by reindexing.
# limit:When forward- or backfilling, maximum size gap (in number of elements) to fill.
# tolerance: When forward- or backfilling, maximum size gap (in absolute numeric distance) to fill for inexact matches.
# level:Match simple Index on level of MultiIndex; otherwise select subset of.
# copy: If True , always copy underlying data even if new index is equivalent to old index; if False , do not copy the data when the indexes are equivalent.

## Dropping Entries from an Axis
new_obj = frame1.drop('c')  # c row is deleted from frame1
# index values can be deleted from either axis in data frame
frame3 = pd.DataFrame(np.arange(9).reshape((3, 3)), index=['a', 'c', 'd'], columns=['Ohio', 'Texas', 'California'])
frame3.drop(['d', 'c'])  # row indexed c,d is dropped frame3
frame3.drop(['Ohio', 'Texas'], axis=1)  # column Ohio,Texas is dropped from frame3. Default axis is 0
# IMP: Be careful with the inplace , as it destroys any data that is dropped.
frame3.drop('c', inplace=True)  # c is permanently removed from frame3
frame3.drop('California', axis=1, inplace=True)  # California column permanently removed

## Indexing, Selection, and Filtering
# Index
obj = pd.Series(np.arange(4.), index=['a', 'b', 'c', 'd'])
obj[2]  # o/p is 2 . Based on index Integer position
obj[1:3]  # o/p is [1,2] i.e stop-start Values
obj['c']  # o/p is 2 . Based on Index name
obj[['b', 'a', 'd']]  # o/p is [1,0,3]
obj[obj < 2]  # o/p is based on gven condition
# Slicing with labels behaves differently than normal Python slicing
obj['a':'c']  # o/p is [0,1,2] based on indexed passed

# Selction:
# Selection with loc[NAME passed] and iloc[INTEGER passed]
frame4 = pd.DataFrame(np.arange(9).reshape((3, 3)), index=['a', 'c', 'd'], columns=['Ohio', 'Texas', 'California'])
frame4.loc['a', ['Ohio', 'Texas']]  # from a, gets Ohio & Texas value
frame4.loc[:, ['Ohio', 'Texas']]  # from all rows, gets Ohio & Texas
frame4.loc['c':'d', ['Ohio', 'Texas']]  # from c,d, gets Ohio & Texas
frame4.iloc[2, [2, 0, 1]]  # returns o/p based on Indexed integer, Rows by Columns
frame4.iloc[1:4, [2, 0, 1]]  # for rows, not boundary, but YES for column
frame4.iloc[1:2, [88, 0, 1]]  # gives out of bound error for column 88
# df[val]:Select single column or sequence of columns from the DataFrame;
# df.loc[val]:Selects single row or subset of rows from the DataFrame by label
# df.loc[:,val]:Selects single column or subset of columns by label
# df.loc[val1,val2]: Select both rows and columns by label
# df.iloc[where]: Selects single row or subset of rows from the DataFrame by integer position
# df.iloc[:, where]: Selects single column or subset of columns by integer position
# df.iloc[where_i, where_j]: Select both rows and columns by integer position
# df.at[label_i, label_j]: Select a single scalar value by row and column label
# df.iat[i, j]: Select a single scalar value by row and column position (integers)
# reindex method: Select either rows or columns by labels
# get_value, set_value methods: get_value, set_value methods Select single value by row and column label

## Arithmetic and Data Alignmen
# When you are adding together objects, if any index pairs are not the same, the respective index in the result will be the union of the index pairs.
# similar to an automatic outer join on the index labels
s1 = pd.Series([7.3, -2.5, 3.4, 1.5], index=['a', 'c', 'd', 'e'])
s2 = pd.Series([-2.1, 3.6, -1.5, 4, 3.1], index=['a', 'c', 'e', 'f', 'g'])
s1 + s2  # o/p has nan for d,f,g index position. Since no corresponfing value in s2
# Imp: internal data alignment introduces missing values in the label locations that don’t overlap
# Similiar is the case when tried with data drame having matching index etc. NaN at misnatched instances

## Arithmetic methods with fill values
df1.add(df2, fill_value=0)  # will add 4+nan as 4 in o/p is rendered.
# add, radd: Methods for addition (+)
# sub, rsub: Methods for subtraction (-)
# div, rdiv: Methods for division (/)
# floordiv, rfloordiv: Methods for floor division (//)
# mul, rmul:Methods for multiplication (*)
# pow, rpow: Methods for exponentiation (**)

## ***  Operations between DataFrame and Series
arr = np.arange(12.).reshape((3, 4))  # 2d array. considering equivalent to df
arr[0]  # o/p is 0th index row. array([ 0.,  1.,  2.,  3.]). equivalent to series
arr - arr[0]  # retruns 3by4
# Imp: When we subtract arr[0] from arr , the subtraction is performed once for each row.
# This is called broadcasting
#  By default, arithmetic between DataFrame and Series matches the index of the Series on the DataFrame’s columns, broadcasting down the rows:
dataframe.sub(series, axis=0)  # subtarct along each row element
dataframe.sub(series, axis=1)  # subtarct along each column element

## Function Application and Mapping
# NumPy ufuncs (element-wise array methods) also work with pandas objects
frame = pd.DataFrame(np.random.randn(4, 3), columns=list('bde'), index=['Utah', 'Ohio', 'Texas', 'Oregon'])
np.abs(frame)  # reurns absolute value of frame object
# Another frequent operation is applying a function on one-dimensional arrays to each column or row
f = lambda x: x.max() - x.min()
frame.apply(f)  # default axis is 0, for rows
frame.apply(f, axis='columns')  # axis changes to 1, columns

## Sorting and Ranking
# To sort lexicographically by row or column index, use the sort_index method
obj = pd.Series(range(4), index=['d', 'a', 'b', 'c'])
obj.sort_index()  # retuns according to [a,b,c,d] index
obj.sort_index(axis=0)
obj.sort  # 'Series' object has no attribute 'sort'
# Here this object can be series, dataframe, Sorting is performed
frame = pd.DataFrame(np.random.randn(4, 3), columns=list('gde'), index=['Utah', 'Ohio', 'Texas', 'Oregon'])
frame.sort_index(axis=0)  # sorting Ohio, Oregon, texas & Utah based on Rows
frame.sort_index(axis=1)  # sorting based on d,e,g Columns
frame.sort_index(axis=1, ascending=False)  # sorted as d,e,g
# To sort a Series by its values, use its sort_values method:
obj1 = pd.Series([4, 7, -3, 2])
obj1.sort_values()  # o/p is -3,2,4,7
obj2 = pd.Series([4, np.nan, 7, np.nan, -3, 2])
obj2.sort_values()  # nan are sorted at last. o/p is -3,2,4,7,nan,nan
## data frame Sorting
frame = pd.DataFrame({'b': [4, 7, -3, 2], 'a': [0, 1, 0, 1]})
frame.sort_values(by='b')  # entire df sorted based on column b
frame.sort_values(by=['a', 'b'])  # sorting on multiple columns of df

## RANKING:assigns ranks from one through the number of valid data points in an array
obj = pd.Series([7, -5, 7, 4, 2, 0, 4])
obj.rank()  # o/p is [6.5,1,,6.5,4.5,3,2,4.5]. Shared Rank for Duplicate values
# sorting based on when they are observed
obj.rank(method='first')  # o/p is [6,1,7,4,3,2,5]
obj.rank(ascending=False)  # ranking order is reversed. i.e. descending
# DataFrame can compute ranks over the rows or the columns
frame = pd.DataFrame({'b': [4.3, 7, -3, 2], 'a': [0, 1, 0, 1], 'c': [-2, 5, 8, -2.5]})
frame.rank(axis='columns')  # returns column wise rank of elements
# average: Default: assign the average rank to each entry in the equal group
# min: Use the minimum rank for the whole group
# max: Use the maximum rank for the whole group
# first: Assign ranks in the order the values appear in the data
# dense: Like method='min' , but ranks always increase by 1 in between groups rather than the number of equal elements in a group

# Axis Indexes with Duplicate Labels
obj = pd.Series(range(5), index=['a', 'a', 'b', 'b', 'c'])
obj.index.is_unique  # False, indicating non unique index label....
obj['a']  # o/p is [0,1] as we have two instances of a

## Summarizing and Computing Descriptive Statistics
# these fall into the category of reductions or summary statistics
df = pd.DataFrame([[1.4, np.nan], [7.1, -4.5], [np.nan, np.nan], [0.75, -1.3]], index=['a', 'b', 'c', 'd'],
                  columns=['one', 'two'])
df.sum()  # computes sum along row axis
df.sum(axis=1)  # computes sum along column axis
# NA values are excluded unless the entire slice (row or column in this case) is NA.
df.mean(axis='columns', skipna=False)  # skip na by default is TRUE
# axis: Axis to reduce over; 0 for DataFrame’s rows and 1 for columns
# skipna: Exclude missing values; True by default
# level: Reduce grouped by level if the axis is hierarchically indexed (MultiIndex)
## return indirect statistics like the index value where the minimum or maximum values are attained
df.idxmax()
df.cumsum()
# Describe & get statistics
df.describe()
# On non-numeric data, describe produces alternative summary statistics
obj = pd.Series(['a', 'a', 'b', 'c'] * 4)
obj.describe()
# count: no of non NA vales
# describe:Compute set of summary statistics for Series or each DataFrame column
# min, max:Compute minimum and maximum values
# argmin, argmax: Compute index locations (integers) at which minimum or maximum value obtained, respectively
# idxmin, idxmax : Compute index labels at which minimum or maximum value obtained, respectively
# quantile: Compute sample quantile ranging from 0 to 1
# sum: Sum of values
# mean:Mean of values
# median:Arithmetic median (50% quantile) of values
# mad:Mean absolute deviation from mean value
# prod: Product of all values
# var:Sample variance of values
# std:Sample standard deviation of values
# skew:Sample skewness (third moment) of values
# kurt:Sample kurtosis (fourth moment) of values
# cumsum:Cumulative sum of values
# cummin, cummax: Cumulative minimum or maximum of values, respectively
# cumprod:Cumulative product of values
# diff:Compute first arithmetic difference (useful for time series)
# pct_change: Compute percent changes

## Correlation and Covariance: Refer pg 160 for details

## Unique Values, Value Counts, and Membership
obj = pd.Series(['c', 'a', 'd', 'a', 'a', 'b', 'b', 'c', 'c'])
uniques = obj.unique()  # o/p is NON-SORTED array(['c', 'a', 'd', 'b'], dtype=object)
uniques.sort()  # sorting of unique value set
obj.value_counts()  # returns the frequency of each object. c3,a3,b2,d1
pd.value_counts(obj.values, sort=False)  # o/p freq as  c3,a3,b2,d1
# masking:
mask = obj.isin(['b', 'c'])  # returns True/ False
obj[mask]  # retruns indexed psotion of b,c

# CHAPTER6:_____________________________________________________________________
### Data Loading, Storage, and File Formats

## Parsing Functions in Pandas
# read_csv: Load delimited data from a file, URL, or file-like object; use comma as default delimiter
# read_table:Load delimited data from a file, URL, or file-like object; use tab ( '\t' ) as default delimiter
# read_fwf:Read data in fixed-width column format (i.e., no delimiters)
# read_clipboard: Version of read_table that reads data from the clipboard; useful for converting tables from web
# read_excel: Read tabular data from an Excel XLS or XLSX file
# read_hdf:Read HDF5 files written by pandas
# read_html:Read all tables found in the given HTML document
# read_json:Read data from a JSON (JavaScript Object Notation) string representation
# read_msgpack: Read pandas data encoded using the MessagePack binary format
# read_pickle: Read an arbitrary object stored in Python pickle format
# read_sas: Read a SAS dataset stored in one of the SAS system’s custom storage formats
# read_sql: Read the results of a SQL query (using SQLAlchemy) as a pandas DataFrame
# read_stata: Read a dataset from Stata file format
# read_feather: Read the Feather binary file format
pd.read_table('examples/ex1.csv', sep=',')  # o/p df as belwo for Reference
frame = pd.DataFrame(np.random.randn(3, 5), columns=['a', 'b', 'c', 'd', 'message'], header=None)
# defining index column as below
pd.read_csv('examples/ex2.csv', index_col='message')  # assigns message as INDEX in resulting df

# refer pg 170-175 onwards for further reading on reading files in python
## Writing Data to Text Format
data.to_csv('examples/out.csv')  # has comma separated CSV file
# Other delimiters can be used, of course like | etc
import sys

data.to_csv(sys.stdout, sep='|')
# handling missing data & replacing them with value say NULL here
data.to_csv(sys.stdout, na_rep='NULL')
# when no header for row, columns needed'
data.to_csv(sys.stdout, index=False, header=False)

# Working with Delimited Formats: pg 176
# JSON Data: pg 178
# XML and HTML: Web Scraping: pg 180
# Binary Data Formats: pg 183
# Using HDF5 Format: pg 185
# Interacting with Web APIs: pg 187
# Interacting with Databases: pg 188

# CHAPTER7:_____________________________________________________________________
### Data Cleaning and Preparation
# Basically involves Loading, Cleaning, Transforming, and Re-arranging.
# reported to take up 80% or more of an analyst’s time.

## Handling Missing Data
# For numeric data, pandas uses the floating-point value NaN (Not a Number) to represent missing data.
# We call this a sentinel value that can be easily detected
string_data = pd.Series(['aardvark', 'artichoke', np.nan, 'avocado'])
string_data.isnull()  # o/p is false, false, true, false
# IMP: NA is Not Available. In statistics applications, NA data may either be data that does not exist or that exists but was not observed (through problems with data collection
# The built-in Python None value is also treated as NA in object arrays
string_data[0] = None  # makes 0th index as None
string_data.isnull()  # o/p is true, false, true, false
# dropna:Filter axis labels based on whether values for each label have missing data, with varying thresholds for how much missing data to tolerate.
# fillna: Fill in missing data with some value or using an interpolation method such as 'ffill' or 'bfill' .
# isnull: Return boolean values indicating which values are missing/NA.
# notnull: Negation of isnull

## Filtering Out Missing Data
from numpy import nan as NA

data = pd.Series([1, NA, 3.5, NA, 7])  # o/p is [1,NaN,3.5,NaN,7]
data.dropna()  # o/p is [1,3.5,7]
# Equivakent to below:
data[data.notnull()]  # o/p is [1,3.5,7]

# In dataframes now
data = pd.DataFrame([[1., 6.5, 3.], [1., NA, NA], [NA, NA, NA], [NA, 6.5, 3.]])
cleaned = data.dropna()
# Passing how='all' will only drop rows that are all NA
data.dropna(how='all')  # removes 2 index ROW  as its all column has NaN
# to drop columns
data[4] = NA  # adds new column as name 4 and having value as nan
data.dropna(axis=1, how='all')  # drops column having all row as nan
## if you want to keep only rows containing a certain number of observations
df = pd.DataFrame(np.random.randn(7, 3))
df.iloc[:4, 1] = NA
df.iloc[:2, 2] = NA
df.dropna()  # drops all rows having at least one nan element
df.dropna(thresh=2)  # drop rows of nan to Threshold of 2 rows. Starts at 0th index from TOP.

# Filling In Missing Data: using above defined 'df'
df.fillna(0)  # fills all nan of df with 0. Can use other INTEGER as well
df.fillna('srimal')  # fills all nan of df with srimal
# fillna with a dict, you can use a different fill value for each column
df.fillna({1: 0.5, 2: 'srimal'})  # first column as 0.5 & second column as srimal
# fillna basically doesnot creates NEW, however you can do it by
df.fillna('amit', inplace=True)  # original df gets updated with nan as amit.
# Filling direction as here
df.fillna(method='ffill')  # forward fills nan in each column
df.fillna(method='bfill')  # backward fills nan in each column, from lowest to topmost column element
df.fillna(method='bfill', limit=2)  # backward fill with threshold of 2.
# fill nan with mean,median etc
df.fillna(df.mean())  # fills nan column with mean of each row
# value: Scalar value or dict-like object to use to fill missing values
# method:Interpolation; by default 'ffill' if function called with no other arguments
# axis:Axis to fill on; default axis=0
# inplace: Modify the calling object without producing a copy
# limit:For forward and backward filling, maximum number of consecutive periods to fill

## Data Transformation
## Removing Duplicates
data = pd.DataFrame({'k1': ['one', 'two'] * 3 + ['two'], 'k2': [1, 1, 2, 3, 3, 4, 4]})
data.duplicated()  # retruns true/false for duplicated key-value PAIR
# removing duplicates from data
data.drop_duplicates()
# lets add one more column
data['v1'] = range(7)
# drop duplicates based on selected column say k1
data.drop_duplicates(['k1'])
# Atribute the KEEP the last occurence of duplicate., with keep= 'last'
data.drop_duplicates(['k1', 'k2'], keep='last')

## Replacing Values
data = pd.Series([1., -999., 2., -999., -1000., 3.])
# The -999 values might be sentinel values for missing data. To replace these with NA.
data.replace(-999, np.nan)  # -999 is replaced by nan
data.replace([-999, -1000], np.nan)  # replacing multiple values
data.replace([-999, -1000], [np.nan, 1000])  # replacing multple values with each correspondence
data.replace({-999: np.nan, -1000: 0})  # replace with dictionary approach

# Renaming Axis Indexes
data = pd.DataFrame(np.arange(12).reshape((3, 4)), index=['Ohio', 'Colorado', 'Newyork'],
                    columns=['one', 'two', 'three', 'four'])
# using map method for index renaming
transform = lambda x: x[:4].upper()
data.index.map(transform)  # Index gets CAPITALIZED
# assign to index of data
data.index = data.index.map(transform)
# using RENAME: create a transformed version of a dataset without modifying the original
data.rename(index=str.title, columns=str.upper)
data.rename(index=str.title, columns=str.lower)
# renaming with Dictionary approach
data.rename(index={'OHIO': 'INDIANA'}, columns={'three': 'peekaboo'})
data.rename(index={'OHIO': 'INDIANA', 'NEWY': 'NEWDELHI'}, columns={'three': 'peekaboo'})

## Discretization and Binning
ages = [20, 22, 25, 27, 21, 23, 37, 31, 61, 45, 41, 32]
# lets decide bin as divide these into bins of 18 to 25, 26 to 35, 36 to 60, and finally 61 and older.
bins = [18, 25, 35, 60, 100]  # bins value decision
cats = pd.cut(ages, bins)
cats  # gives bin value range for each ages
cats.codes  # o/p is array([0, 0, 0, 1, 0, 0, 2, 1, 3, 2, 2, 1], dtype=int8)
cats.categories  # o/p is [(18, 25], (25, 35], (35, 60], (60, 100]]
pd.value_counts(cats)  # o/p is freq in each bin. [5,3,3,1]
# imp:a parenthesis means that the side is open, while the square bracket means it is closed (inclusive)
# pass your own bin names by passing a list or array to the labels
group_names = ['Youth', 'YoungAdult', 'MiddleAged', 'Senior']
pd.cut(ages, bins, labels=group_names)  # age bins are named now as per group_names
## pass an integer number of bins to cut instead of explicit bin edges
# compute equal-length bins based on the minimum and maximum values in the data.
data = np.random.rand(20)
pd.cut(data, 4, precision=2)  # gets 4 bins having precision upto 2 digits
## bins based on quantiles
# 1.closely related function, qcut , bins the data based on sample quantiles
data = np.random.randn(1000)  # Normally distributed
cats = pd.qcut(data, 4)  # Cut into EQUAL quartiles
pd.value_counts(cats)
# 2. passing your own quantiles, (numbers between 0 and 1, inclusive)
pd.qcut(data, [0, 0.1, 0.5, 0.9, 1.])
pd.value_counts(cats)

## Detecting and Filtering Outliers
# Filtering or transforming outliers is largely a matter of applying array operations
data = pd.DataFrame(np.random.randn(1000, 4))
data.describe()  # returns mean, med, avg,quantile etc.
# find values in one of the columns exceeding 3 in absolute value
col = data[2]  # selecting target column of data, based on col. index
col[np.abs(col) > 3]  # returns mod value greater than 3
# select all rows having a value exceeding 3 or –3
data[(np.abs(data) > 3).any(1)]

## Permutation and Random Sampling
df = pd.DataFrame(np.arange(5 * 4).reshape((5, 4)))
sampler = np.random.permutation(5)
sampler  # o/p is  array([4, 2, 1, 0, 3])
df.take(sampler)  # sampler used as INDEX in df to retrieve row columns df
# random selection of sample
df.sample(n=3)  # randomly selects 3 row df, all columns populated

## Computing Indicator/Dummy Variables
df = pd.DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'b'], 'data1': range(6)})
pd.get_dummies(df['key'])
# Refer 208 for more details..............

## String Manipulation
# String Object Methods
val = 'a,b,guido'
pieces = val.split(',')  # o/p is  ['a', 'b', 'guido']
val.split('u')  # o/p is ['a,b,g', 'ido']
# Imp: split is often combined with strip to trim whitespace (including line breaks)
# concatenate opeartion on above opeartion using '::'
'::'.join(pieces)  # o/p is 'a::b::guido'
# count returns the number of occurrences of a particular substring
val.count(',')  # o/p is 2 implying 2 occurences
# replacement
val.replace(',', '@')  # replace , with @
# count: Return the number of non-overlapping occurrences of substring in the string.
# endswith: Returns True if string ends with suffix.
# startswith: Returns True if string starts with prefix.
# join: Use string as delimiter for concatenating a sequence of other strings.
# index: Return position of first character in substring if found in the string; raises ValueError if not found.
# find: Return position of first character of first occurrence of substring in the string; like index , but returns –1 if not found.
# rfind: Return position of first character of last occurrence of substring in the string; returns –1 if not found.
# replace: Replace occurrences of string with another string.
# strip,rstrip,lstrip: Trim whitespace, including newlines; equivalent to x.strip() (and rstrip, lstrip , respectively) for each element.
# split:Break string into list of substrings using passed delimiter.
# lower: Convert alphabet characters to lowercase.
# upper: Convert alphabet characters to uppercase.
# casefold: Convert characters to lowercase, and convert any region-specific variable character combinations to a common comparable form.
# ljust,rjust: Left justify or right justify, respectively; pad opposite side of string with spaces (or some other fill character) to return a string with a minimum width.

## Regular Expressions
# Regular expressions provide a flexible way to search or match (often more complex) string patterns in text.
# A single expression, commonly called a REGEX, is a string formed according to the regular expression language.
# Python’s built-in re module is responsible for applying regular expressions to strings
# re module functions fall into three categories: PATTERN MATCHING, SUBSTITUTION, and SPLITTING
# eg:  suppose we wanted to split a string with a variable number of whitespace characters (tabs, spaces, and newlines).
import re

text = "foo   bar\t baz  \tqux"
# The regex describing one or more whitespace characters is \s+
re.split('\s+', text)  # spliting operation. o/p is ['foo', 'bar', 'baz', 'qux']
# alternate to above step
regex = re.compile('\s+')
regex.split(text)  # o/p is ['foo', 'bar', 'baz', 'qux']
# get a list of all patterns matching the regex,then use findall method
regex.findall(text)  # o/p is [' ', '\t ', ' \t']

# Note1: Creating a regex object with re.compile is highly recommended if you intend to apply the same expression to many strings; doing so will save CPU cycles
# Note2: match and search are closely related to findall .
# findall returns all matches in a string, search returns only the first match
# match only matches at the beginning of the string
text = """Dave dave@google.com
Steve steve@gmail.com
Rob rob@gmail.com
Ryan ryan@yahoo.com"""
pattern = r'[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,4}'  # defining expression for email id
import re

regex = re.compile(pattern, flags=re.IGNORECASE)  # re.ignorecase makes it case in-sensitive
# Using findall on the text produces a list of the email addresses:
regex.findall(text)  # finds allpattern in complied regex
# search returns a special match object for the first email address in the text
m = regex.search(text)
< _sre.SRE_Match
object;
span = (5, 20), match = 'dave@google.com' >
text[m.start():m.end()]  # serach text for m matches
# Refer 215 for more details on regex

## Vectorized String Functions in pandas
data = {'Dave': 'dave@google.com', 'Steve': 'steve@gmail.com', 'Rob': 'rob@gmail.com', 'Wes': np.nan}
data = pd.Series(data)
data.isnull()
# check whether each email address has 'gmail' in it with str.contains
data.str.contains('gmail')  # str.contains function to check gmail in elements
# using regex to find gmail string
matches = data.str.findall(pattern, flags=re.IGNORECASE)
# access elements in the embedded lists
matches.str.get(1)  # o/p is all nan
matches.str.get(0)  # o/p is all nan
# string slicing
data.str[:5]  # gets first 5 alphabets from each row element

# cat: Concatenate strings element-wise with optional delimiter
# contains: Return boolean array if each string contains pattern/regex
# count: Count occurrences of pattern
# extract: Use a regular expression with groups to extract one or more strings from a Series of strings; the result will be a DataFrame with one column per group
# endswith:Equivalent to x.endswith(pattern) for each element
# startswith:Equivalent to x.startswith(pattern) for each element
# findall:Compute list of all occurrences of pattern/regex for each string
# get: Index into each element (retrieve i-th element)
# isalnum:Equivalent to built-in str.alnum
# isalpha:Equivalent to built-in str.isalpha
# isdecimal:Equivalent to built-in str.isdecimal
# isdigit:Equivalent to built-in str.isdigit
# islower:Equivalent to built-in str.islower
# isnumeric:Equivalent to built-in str.isnumeric
# isupper:Equivalent to built-in str.isupper
# join:Join strings in each element of the Series with passed separator
# len: Compute length of each string
# lower, upper:Convert cases; equivalent to x.lower() or x.upper() for each element
# match: Use re.match with the passed regular expression on each element, returning matched groups as list
# pad: Add whitespace to left, right, or both sides of strings
# center: Equivalent to pad(side='both')
# repeat: Duplicate values (e.g., s.str.repeat(3) is equivalent to x * 3 for each string)
# replace: Replace occurrences of pattern/regex with some other string
# slice: Slice each string in the Series
# split: Split strings on delimiter or regular expression
# strip: Trim whitespace from both sides, including newlines
# rstrip: Trim whitespace on right side
# lstrip: Trim whitespace on left side

# CHAPTER8:_____________________________________________________________________
### Data Wrangling: Join, Combine, and Reshape
## Hierarchical Indexing : important feature of pandas that enables you to have multiple (two or more) index levels on an axis
# can be used for forming a pivot table
data = pd.Series(np.random.randn(9), index=[['a', 'a', 'a', 'b', 'b', 'c', 'c', 'd', 'd'], [1, 2, 3, 1, 3, 1, 2, 2, 3]])
# “gaps” in the index display mean “use the label directly above
data.index
# With a hierarchically indexed object, so-called partial indexing is possible to select subset of data
data['b']  # gets data of key index b
data['b':'d']  # gets data of key index from b to d inclusive
data.loc[['b', 'd']]  # # gets data of key index b & d ONLY
data.loc[:, 2]  # all FIRST index abcd , with 2nd INDEX element post comma
data.loc[:, 1:2]  # retruns using the same logic as ABOVE
# rearrange the multi-hierarchy data into a DataFrame using its unstack method
data.unstack()  # creates 4by3 df. 4 from 1st level & 3 from 2nd level index of multi-hierarchial df
data.unstack().stack()  # reversing the unstack operation

# With a DataFrame, either axis can have a hierarchical index:
frame = pd.DataFrame(np.arange(12).reshape((4, 3)),
                     index=[['a', 'a', 'b', 'b'], [1, 2, 1, 2]],
                     columns=[['Ohio', 'Ohio', 'Colorado'],
                              ['Green', 'Red', 'Green']])  # has row & coumn level multi-indexing
# assigning name to both level of index
frame.index.names = ['key1', 'key2']  # row level naming
frame.columns.names = ['state', 'color']  # column level naming
frame['Ohio']  # returns Ohio column entire data set
frame['Ohio', 'Red']  # returns Ohio column's Red columns entire data set
frame.loc['a']  # reurtns a row entire data set
frame.loc['a', 2]  # reurtns a row's 2 row's entire data set
frame.loc['a', '2']  # retruns error of labelling

# Reordering and Sorting Levels
# rearrange the order of the levels on an axis or sort the data by the values in one specific level
# The swaplevel takes two level numbers or names and returns a new object with the levels interchanged (but the data is otherwise unaltered)
frame.swaplevel('key1', 'key2')  # swaps key1 &2 positions
# sort_index , on the other hand, sorts the data using only the values in a single level
frame.sort_index(level=0)  # sorting based on 0th index . ie a here in this case
frame.sort_index(level=1)  # sorting based on 1st index . ie 1/2 here in this case

## Summary Statistics by Level
frame.sum(level='key2')
frame.sum(level='color', axis=1)

## Indexing with a DataFrame’s columns
frame = pd.DataFrame({'a': range(7), 'b': range(7, 0, -1), 'c': ['one', 'one', 'one', 'two', 'two', 'two', 'two'],
                      'd': [0, 1, 2, 0, 1, 2, 3]})
# create a new DataFrame [Hierarchial] using one or more of its columns as the index
frame2 = frame.set_index(['c', 'd'])  # c is 1st level & d is 2nd level of INDEXING
# By default the columns are removed from the DataFrame, to keep them pass drop argument as False
frame3 = frame.set_index(['c', 'd'], drop=False)
# reset_index , on the other hand, does the opposite of set_index.
# Keep drop is True, else Already Exists error will be thrown.
frame2.reset_index()  # drop is true & hence no error
frame3.reset_index()  # gives error of already exists

## Combining and Merging Datasets
# Number of ways Pandas offer to combine data as below:
# 1. pandas.merge:connects rows in DataFrames based on one or more keys. Like SQL join
# 2. pandas.concat: concatenates or “stacks” together objects along an axis.
# 3. combine_first: enables splicing together overlapping data to fill in missing values in one object with values from another

# Database-Style DataFrame Joins
df1 = pd.DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'a', 'b'], 'data1': range(7)})
df2 = pd.DataFrame({'key': ['a', 'b', 'd'], 'data2': range(3)})
##.. many to one join, merge uses the overlapping column names as the keys if we dont pass key argument
pd.merge(df1, df2)  # key not passed & hence OVERLAPPING key (a&b) is used.
pd.merge(df1, df2, on='key')  # key passed on merge
# If the column names are different in each object, you can specify them separately:
df3 = pd.DataFrame({'lkey': ['b', 'b', 'a', 'c', 'a', 'a', 'b'], 'data1': range(7)})
df4 = pd.DataFrame({'rkey': ['a', 'b', 'd'], 'data2': range(3)})
pd.merge(df3, df4, left_on='lkey', right_on='rkey')
# IMP Note: y default merge does an 'INNER' join
# outer join takes the union of the keys
pd.merge(df1, df2, how='outer')
# 'inner': Use only the key combinations observed in both tables
# 'left':Use all key combinations found in the left table
# 'right' Use all key combinations found in the right table
# 'output' Use all key combinations observed in both tables together

##.. Many-to-many joins form the Cartesian product of the rows.
pd.merge(df1, df2, on='key', how='left')
pd.merge(df1, df2, how='inner')
# To merge with multiple keys, pass a list of column names:
pd.merge(left, right, on=['key1', 'key2'], how='outer')
# treatment of overlapping column names
pd.merge(left, right, on='key1')
pd.merge(left, right, on='key1', suffixes=('_left', '_right'))  # adding suffix to overlapping columns

## Merging on Index
# pass left_index=True or right_index=True (or both) to indicate that the index should be used as the merge key
left1 = pd.DataFrame({'key': ['a', 'b', 'a', 'a', 'b', 'c'], 'value': range(6)})
right1 = pd.DataFrame({'group_val': [3.5, 7]}, index=['a', 'b'])
pd.merge(left1, right1, left_on='key', right_index=True)
# Since the default merge method is to intersect the join keys, you can instead form the union of them with an outer join
pd.merge(left1, right1, left_on='key', right_index=True, how='outer')
# merge on multi-hierarchial index : refer 233-236

## Concatenating Along an Axis
# IMP: interchangeably known as concatenation, binding, or stacking.
arr = np.arange(12).reshape((3, 4))
np.concatenate([arr, arr], axis=0)  # concatenate along Row, hence 6by4 is created
np.concatenate([arr, arr], axis=1)  # concatenate along Column, hence 3by8 is created
np.concatenate([arr, arr])  # concatenate along Row, hence 6by4 is created
# By default concat works along axis=0 , producing another Series as above.
# suppose 3 series having non-overlapping index
s1 = pd.Series([0, 1], index=['a', 'b'])
s2 = pd.Series([2, 3, 4], index=['c', 'd', 'e'])
s3 = pd.Series([5, 6], index=['f', 'g'])
s4 = pd.concat([s1, s3])
pd.concat([s1, s2, s3]  # produces another list glued along row axis. DEFAULT row axis
pd.concat([s1, s2, s3], axis=1)  # glueing along the colmn axis
# settings type of join
pd.concat([s1, s4], axis=1, join='inner')  # join along column set as INNER
# specify the axes to be used on the other axes with join_axes
pd.concat([s1, s4], axis=1, join_axes=[['a', 'c', 'b', 'e']])

## A potential issue is that the concatenated pieces are not identifiable in the result
result = pd.concat([s1, s1, s3], keys=['one', 'two', 'three'])
result.unstack()
pd.concat([s1, s2, s3], axis=1, keys=['one', 'two', 'three'])  # setting concat axis as COLUMN
# The same logic extends for data frame as well: Refer 239 for details

## Combining Data with Overlap
# case where two datasets whose indexes overlap in full or part.
a = pd.Series([np.nan, 2.5, np.nan, 3.5, 4.5, np.nan], index=['f', 'e', 'd', 'c', 'b', 'a'])
b = pd.Series(np.arange(len(a), dtype=np.float64), index=['f', 'e', 'd', 'c', 'b', 'a'])
b[-1] = np.nan
b[:-2].combine_first(a[2:])  # combines from part1 & part 2 respetively.if index duplcate, takes from first case.
# With DataFrames, combine_first does the same thing column by column
df1 = pd.DataFrame({'a': [1., np.nan, 5., np.nan], 'b': [np.nan, 2., np.nan, 6.], 'c': range(2, 18, 4)})
df2 = pd.DataFrame({'a': [5., 4., np.nan, 3., 7.], 'b': [np.nan, 3., 4., 6., 8.]})
df1.combine_first(df2)  # checks ocurence first & then fills nan with some matched value

## Reshaping and Pivoting
# basic operations for rearranging tabular data.
# Alternatively called "Reshaping and Pivoting".
# Hierarchical indexing provides a consistent way to rearrange data in a DataFrame
# Mainly there are TWO actions as:
# 1. stack: This “rotates” or pivots from the columns in the data to the rows
# 2. unstack: This pivots from the rows into the columns
data = pd.DataFrame(np.arange(6).reshape((2, 3)), index=pd.Index(['Ohio', 'Colorado'], name='state'),
                    columns=pd.Index(['one', 'two', 'three'], name='number'))
# use stack & pivots the columns into the rows, producing a Series
result1 = data.stack()  # columns becomes row
result2 = result.unstack()  # equivalent to original data
# IMP: By default the innermost level is stacked-unstacked. We can choose level as below:
result1.unstack(0)  # along row axis
result.unstack('state')  # along state
# Note: Unstacking might introduce missing data if all of the values in the level aren’t found in each of the subgroups
s1 = pd.Series([0, 1, 2, 3], index=['a', 'b', 'c', 'd'])
s2 = pd.Series([4, 5, 6], index=['c', 'd', 'e'])
data2 = pd.concat([s1, s2], keys=['one', 'two'])
data2.unstack()  # introduces missing values as nan
# Stacking filters out the missing data by DEFAULT
data2.unstack().stack()
data2.unstack().stack(dropna=False)  # passing argument if we want to keep nan
## When you unstack in a DataFrame, the level unstacked becomes the lowest level in the result
df = pd.DataFrame({'left': result, 'right': result + 5}, columns=pd.Index(['left', 'right'], name='side'))
df.unstack('state')
# When calling stack , we can indicate the name of the axis to stack
df.unstack('state').stack('side')

## Pivoting long to wide :refer 246
## Pivoting wide to long :refer 249

# CHAPTER9:_____________________________________________________________________
### Plotting and Visualization
# refer 253 - 287 for details..........


# CHAPTER10:_____________________________________________________________________
### Data Aggregation and Group Operations
# GroupBy Mechanics
# Hadley Wickham, coined the term split-apply-combine(SAC) for describing group operations.
# Stage1: data is split into groups based on one or more keys
# Stage2: a function is applied to each group, producing a new value.
# Stage3 :results of all those function applications are combined into a result object
df = pd.DataFrame(
    {'key1': ['a', 'a', 'b', 'b', 'a'], 'key2': ['one', 'two', 'one', 'two', 'one'], 'data1': np.random.randn(5),
     'data2': np.random.randn(5)})
# compute the mean of the data1 column using the labels from key1
grouped = df['data1'].groupby(df['key1'])
# Note: This groupby has not actually computed anything yet except for some intermediate data about the group key df['key1']
# to compute group means we can call the GroupBy’s mean method
grouped.mean()  # o/p is [a/b : -0.429/-1.324] for each key's mean
# Hence here data (a Series) has been aggregated according to the group key
# Now passing multiple arrays as a list
means = df['data1'].groupby([df['key1'], df['key2']]).mean()  # compute level wise mean
means.unstack()  # will unstack the above hierarchial frame
# grouping based on passing correct length of array, instead of keys
states = np.array(['Ohio', 'California', 'California', 'Ohio', 'Ohio'])
years = np.array([2005, 2005, 2006, 2005, 2006])
df['data1'].groupby([states, years]).mean()  # no of elelemts needs to EQUAL here, else indexing ERROR.

# Passing column name as group by key
df.groupby('key1').mean()  # takes unique element as key & perform grouping
df.groupby(['key1', 'key2']).mean()  # on multiple keys column
## GroupBy method SIZE, which returns a Series containing group sizes
df.groupby(['key1', 'key2']).size()  # returns size
# Imp: any missing values in a group key will be excluded from the result

## Iterating Over Groups
# GroupBy object supports iteration, generating a sequence of 2-tuples containing the group name along with the chunk of data
for name, group in df.groupby('key1'):
    print(name)
print(group)
# case of multiple keys, the first element in the tuple will be a tuple of key values
for (k1, k2), group in df.groupby(['key1', 'key2']):
    print((k1, k2))
print(group)
# handling pieces of data generated from above action
pieces = dict(list(df.groupby('key1')))
pieces['a']
pieces['b']
# IMP: By default groupby groups on axis=0 , but you can group on any of the other axes.
# Lets group the columns of our df here by dtype
df.dtypes  # float, float, object, object
grouped = df.groupby(df.dtypes, axis=1)  # gives description & needs printing as below
for dtype, group in grouped:
    print(dtype)
print(group)

## Selecting a Column or Subset of Columns
# Indexing a GroupBy object created from a DataFrame with a column name or array
# of column names has the effect of column subsetting for aggregation
df.groupby('key1')['data1']  # quick approach of grouping when compared to below method
df['data1'].groupby(df['key1'])  # same as above
df.groupby('key1')[['data2']]
df[['data2']].groupby(df['key1'])  # same as above
# for large datasets, it may be desirable to aggregate only a few columns.
df.groupby(['key1', 'key2'])[['data2']].mean()

## Grouping with Dicts and Series
# Grouping information may exist in a form other than an array
people = pd.DataFrame(np.random.randn(5, 5),
                      columns=['a', 'b', 'c', 'd', 'e'],
                      index=['Joe', 'Steve', 'Wes', 'Jim', 'Travis'])  # dataframe people created
people.iloc[2:3, [1, 2]] = np.nan  # Add a few NA values
# suppose I have a group correspondence for the columns and want to sum together the columns by group
mapping = {'a': 'red', 'b': 'red', 'c': 'blue', 'd': 'blue', 'e': 'red', 'f': 'orange'}
by_column = people.groupby(mapping, axis=1)
by_column.sum()
## The same functionality holds for Series, which can be viewed as a fixed-size mapping
map_series = pd.Series(mapping)
map_series
people.groupby(map_series, axis=1).count()
people.groupby(map_series, axis=1).sum()

## Grouping with Functions
# Any function passed as a group key will be called once per index value, with the return values being used as the group name
# Suppose you wanted to group by the length of the names in people dataframe
people.groupby(len).sum()
# Note: Mixing functions with arrays, dicts, or Series is not a problem as everything gets converted to arrays internally
key_list = ['one', 'one', 'one', 'two', 'two']
people.groupby([len, key_list]).min()

## Grouping by Index Levels
# aggregate using one of the levels of an axis index
columns = pd.MultiIndex.from_arrays([['US', 'US', 'US', 'JP', 'JP'], [1, 3, 5, 1, 3]], names=['cty', 'tenor'])
hier_df = pd.DataFrame(np.random.randn(4, 5), columns=columns)
# To group by level, pass the level number or name using the level keyword:
hier_df.groupby(level='cty', axis=1).count()

## Data Aggregation
# Aggregations refer to any data transformation that produces scalar values from arrays.
# count: Number of non-NA values in the group
# sum: Sum of non-NA values
# mean:Mean of non-NA values
# median: Arithmetic median of non-NA values
# std, var: Unbiased (n – 1 denominator) standard deviation and variance
# min, max: Minimum and maximum of non-NA values
# prod:Product of non-NA values
# first, last: First and last non-NA values
df = pd.DataFrame(
    {'key1': ['a', 'a', 'b', 'b', 'a'], 'key2': ['one', 'two', 'one', 'two', 'one'], 'data1': np.random.randn(5),
     'data2': np.random.randn(5)})
grouped = df.groupby('key1')
grouped['data1'].quantile(0.9)
## To use your own aggregation functions, pass any function that aggregates an array to the aggregate or agg method


def peak_to_peak(arr):
    return arr.max() - arr.min()


grouped.agg(peak_to_peak)
grouped.describe()
# Imp: Custom aggregation functions are generally much slower than the optimized functions
# becoz there is some extra overhead (function calls, data rearrangement) in con structing the intermediate group data chunks.

## Column-Wise and Multiple Function Application
tips = pd.read_csv('examples/tips.csv')  # reading csv file
tips['tip_pct'] = tips['tip'] / tips['total_bill']  # Add tip percentage of total bill
tips[:6]  # returning 6 rows
grouped = tips.groupby(['day', 'smoker'])  # grouping by day & smoker
# for descriptive stats, pass the name of the function as a string
grouped_pct = grouped['tip_pct']
grouped_pct.agg('mean')
# If you pass a list of functions or function names instead, you get back a DataFrame with column names taken from the functions
grouped_pct.agg(['mean', 'std', peak_to_peak])
# tuples in ordered marching
grouped_pct.agg([('foo', 'mean'), ('bar', np.std)])

# ************Refer from 299 - 317 for further readings

# CHAPTER11:_____________________________________________________________________
### Time Series- 1)Fixed & 2)Irregular frequency time Series
#  The simplest and most widely used kind of time series are those indexed by timestamp

## Date and Time Data Types and Tools
# The datetime, time, and calendar modules are the main places to start.
# The datetime.datetime type, or simply datetime, is widely used.
from datetime import datetime

now = datetime.now()  # o/p is datetime.datetime(2018, 7, 8, 8, 54, 2)
now.year, now.month, now.day  # o/p is (2018, 7, 8)
# timedelta represents the temporal difference between two datetime objects
delta = datetime(2011, 1, 7) - datetime(2008, 6, 24, 8, 15)
delta  # o/p is datetime.timedelta(926, 56700), first is days, second is seconds time
delta.days  # 926 days is o/p
delta.seconds  # 56700 seconds is o/p
# add (or subtract) a timedelta or multiple thereof to a datetime object
from datetime import timedelta

start = datetime(2011, 1, 7)
start + timedelta(12)  # o/p is datetime.datetime(2011, 1, 19, 0, 0)
start - 2 * timedelta(12)  # o/p is datetime.datetime(2010, 12, 14, 0, 0)
# date: Store calendar date (year, month, day) using the Gregorian calendar
# time: Store time of day as hours, minutes, seconds, and microseconds
# datetime: Stores both date and time
# timedelta: Represents the diﬀerence between two datetime values (as days, seconds, and microseconds)
# tzinfo: Base type for storing time zone information

## Converting Between String and Datetime
stamp = datetime(2011, 1, 3)
str(stamp)  # o/p is string  '2011-01-03 00:00:00'
stamp.strftime('%Y-%m-%d')  # o/p is '2011-01-03'

## Format Specification
# %Y Four-digit year
# %y Two-digit year
# %m Two-digit month [01, 12]
# %d Two-digit day [01, 31]
# %H Hour (24-hour clock) [00, 23]
# %I Hour (12-hour clock) [01, 12]
# %M Two-digit minute [00, 59]
# %S Second [00, 61] (seconds 60, 61 account for leap seconds)
# %w Weekday as integer [0 (Sunday), 6]
# %U Week number of the year [00, 53]; Sunday is considered the frst day of the week, and days before the frst Sunday of the year are “week 0”
# %W Week number of the year [00, 53]; Monday is considered the frst day of the week, and days before the frst Monday of the year are “week 0”
# %z UTC time zone oﬀset as +HHMM or -HHMM; empty if time zone naive
# %F Shortcut for %Y-%m-%d (e.g., 2012-4-18)
# %D Shortcut for %m/%d/%y (e.g., 04/18/12)

320






































