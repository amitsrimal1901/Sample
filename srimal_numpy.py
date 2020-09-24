######Numpy Release 1.11.0##### Source: numpy-user-1.11.0.pdf

#NumPy is the fundamental package for scientific computing in Python.
#At the core of the NumPy package, is the ndarray object.
#This encapsulates n-dimensional arrays of homogeneous data types, with many operations being performed in compiled code for performance.

#Differences between NumPy arrays and the standard Python sequences:
#1. NumPy arrays have a fixed size at creation, unlike Python lists (which can grow dynamically). Changing the size of an ndarray will create a new array and delete the original.
#2. The elements in a NumPy array are all required to be of the same data type, and thus will be the same size in
#memory. The exception: one can have arrays of (Python, including NumPy) objects, thereby allowing for arrays
#of different sized elements.
#3. NumPy arrays facilitate advanced mathematical and other types of operations on large numbers of data. Typically,
#such operations are executed more efficiently and with less code than is possible using Python’s built-in
#sequences.

#Two of NumPy’s features which are the basis of much of its power: Vectorization and Broadcasting.
#1. Vectorization describes the absence of any explicit looping, indexing, etc., in the code - these things are taking place,
#of course, just “behind the scenes” in optimized, pre-compiled C code.
#2.Broadcasting is the term used to describe the implicit element-by-element behavior of operations; generally speaking,
#in NumPy all operations, not just arithmetic operations, but logical, bit-wise, functional, etc., behave in this implicit
#element-by-element fashion, i.e., they broadcast.

#NumPy’s main object is the homogeneous multidimensional array. It is a table of elements (usually numbers), all of
#the same type, indexed by a tuple of positive integers. In Numpy dimensions are called axes. The number of axes is rank.

#Numpy’s array class is called ndarray. It is also known by the alias array. Note that numpy.array is not the
#same as the Standard Python Library class array.array, which only handles one-dimensional arrays and offers
#less functionality.

#Array needs to be declared for usage:
import array
arr = array.array('i', [1, 2, 3]) 
print (arr) # array('i', [1, 2, 3])

#The more important attributes of an ndarray object are:
#1. ndarray.ndim: the number of axes (dimensions) of the array
#2. ndarray.shape: the dimensions of the array.
#3. ndarray.size: the total number of elements of the array.
#4. ndarray.dtype: an object describing the type of the elements in the array.
#5. ndarray.itemsize: the size in bytes of each element of the array.
#6. ndarray.data: the buffer containing the actual elements of the array.

import numpy as np
a = np.arange(15).reshape(3, 5)
print(a) # numpy.ndarray
type(a)
a.shape # 3 by 5
a.size # a5 elements in here
a.ndim # o/p is 2
a.dtype.name # int32
a.itemsize

# for a single dimension array
b=np.arange(3)
print (b)
b.ndim # o/p is 1
b.shape # o/p is (3L,)

#Array Creation
a1 = np.array([2,3,4])   ## Wrong np.array(2,3,4)
print (a1)
type(a1) # numpy.ndarray
a2 = np.array([(1.5,2,3), (4,5,7)]) 
print (a2)
a21 = np.array([(1.5,2,3), (4,5)]) # allows array creation even if different count of elements in both set, but size gets impacted
print (a21)
a21.shape # o/p is (2L,)
#array transforms sequences of sequences into two-dimensional arrays, sequences of sequences of sequences into
#three-dimensional arrays, and so on.
a3 = np.array( [ [1,2], [3,4] ], dtype=complex ) #defining data type at run time
a3.shape #o/p is (2L,2L)
##Array creation with PLACEHOLDER
#By default, the dtype of the created array is float64.
np.zeros( (3,4) ) #arrays full of ZERO
np.zeros( (3,4,5) ) #arrays full of ZERO, but different dimension
np.ones( (2,3,4), dtype=np.int16 )  # dtype can also be specified
np.empty( (2,3) )  #uninitialized, output may vary

##Create Sequence
np.arange( 10, 30, 5 )
np.arange( 0, 2, 0.3 ) # it accepts float arguments

##function LINSPACE: that receives as an argument the number of elements that we want, instead of the step:
from numpy import pi
np.linspace( 0, 2, 9 ) # 9 numbers from 0 to 2 (including 0 & 2 both)

a4 = np.arange(6) # 1d array
a5 = np.arange(12).reshape(4,3) # 2d array
a6 = np.arange(24).reshape(2,3,4) # 3d array
print (a6)

#If an array is too large to be printed, NumPy automatically skips the central part of the array and only prints the corners:
print(np.arange(10000))
#To disable this behaviour and force NumPy to print the entire array, you can change the printing options using set_printoptions.
np.set_printoptions(threshold='nan')

##Basic Operatons
#The product operator * operates elementwise in NumPy arrays. The matrix product
#can be performed using the dot function or method:
A = np.array( [[1,1],[0,1]] )
B = np.array( [[2,0],[3,4]] )
A*B # elementwise product
A.dot(B) # matrix product
np.dot(A, B) # another matrix product

#Unary operations
a7 = np.arange(12).reshape(3,4)
a7.sum(axis=0) # sum of each column
a7.min(axis=1) # min of each row
a7.cumsum(axis=1) # cumulative sum along each row

##UNIVERSAL functions
#NumPy provides familiar mathematical functions such as sin, cos, and exp. In NumPy, these are called “universal
#functions”(ufunc).
#Within NumPy, these functions operate elementwise on an array, producing an array as output.
B1 = np.arange(3)
np.exp(B1)
np.sqrt(B1)
B2 = np.array([2., -1., 4.])
np.add(B1, B2)

###Indexing, Slicing and Iterating
a8 = np.arange(10)**3
a8[2:5]
a8[:6:2]
a8[ : :-1] # reversed a

#multidimensinal array
def f(x,y):
     return 10*x+y
a9 = np.fromfunction(f,(5,4),dtype=int)
a9[0:5, 1] # each row in the second column of a9
a9[ : ,1] # equivalent to the previous example
a9[1:3, : ] # each column in the second and third row of a9
a9[-1] # the last row. Equivalent to b[-1,:]

#### SHape Manipulation
a10 = np.floor(10*np.random.random((3,4)))
a10.shape
a10.ravel() # flatten the array
a10.T
## reshape & re-sizing
#The reshape function returns its argument with a modified shape, whereas the ndarray.resize method modifies
#the array itself:
a10.resize((2,6))
a10.reshape(3,-1) #If a dimension is given as -1 in a reshaping operation, the other dimensions are automatically calculated.


## STAcKING operation
#Several arrays can be stacked together along different axes:
a11 = np.floor(10*np.random.random((2,2)))
a12 = np.floor(10*np.random.random((2,2)))
np.vstack((a11,a12))
np.hstack((a11,a12))
#The function column_stack stacks 1D arrays as columns into a 2D array.
from numpy import newaxis
np.column_stack((a11,a12)) # With 2D arrays

a13 = np.array([4.,2.])
a14 = np.array([2.,8.])
a13[:,newaxis] # This allows to have a 2D columns vector
a14[:,newaxis]
np.column_stack((a13[:,newaxis],a14[:,newaxis]))

##r_ & c_
#In complex cases, r_ and c_ are useful for creating arrays by stacking numbers along one axis. They allow the use of
#range literals (”:”) :
np.r_[1:4,0,4]

##SPLITTING one array into several smaller ones
a15 = np.floor(10*np.random.random((2,12)))
np.hsplit(a15,3) # Split a array  into 3 array set HORIZONTALLY
np.hsplit(a15,(3,4)) # Split a after the third and the fourth column

##COPIES & VIEW (three cases to consider)
#1. No Copy at All
a16 = np.arange(12)
b1=a16 # aa and b1 are two names for the same ndarray object
b1.shape = 3,4 # changes the shape of a1
a16.shape

#2. View or Shallow Copy 
#reates a new array object that looks at the same data.
c1 = a16.view()
c.base is a16 # c is a view of the data owned by a1
c.flags.owndata
c.shape = 2,6 # a1's shape doesn't change
a16.shape
c[0,4] = 1234 # a1's data changes as 1234 gets inserted in 0,4 place.
s = a16[ : , 1:3]  ## sclicing operation

#3. Deep copy 
#The copy method makes a complete copy of the array and its data.
d = a16.copy() # a new array object with new data is created
d.base is a16 # d doesn't share anything with a

##FUNCTION & METHODS OVERVIEW
#1. Array Creation:
#arange, array, copy, empty, empty_like, eye, fromfile, fromfunction, identity,
#linspace, logspace, mgrid, ogrid, ones, ones_like, r, zeros, zeros_like
#2. Conversions: ndarray.astype, atleast_1d, atleast_2d, atleast_3d, mat
#3. Manipulations: array_split, column_stack, concatenate, diagonal, dsplit, dstack, hsplit, hstack,
#ndarray.item, newaxis, ravel, repeat, reshape, resize, squeeze, swapaxes, take,
#transpose, vsplit, vstack
#4. Questions: all, any, nonzero, where
#5. Ordering: argmax, argmin, argsort, max, min, ptp, searchsorted, sort
#6. Operations: choose, compress, cumprod, cumsum, inner, ndarray.fill, imag, prod, put, putmask, real,sum
#7. Basic Linear Algebra: cross, dot, outer, linalg.svd, vdot
#8. Basic Statistics: cov, mean, std, var

##BROADCASTING : allows to handle data which doesnot have SAME Shape.
#broadcasting rule 1: if all input arrays do not have the same number of dimensions, a “1” will be
#repeatedly prepended to the shapes of the smaller arrays until all the arrays have the same number of dimensions.
#roadcasting rule 2: ensures that arrays with a size of 1 along a particular dimension act as if they had the
#size of the array with the largest shape along that dimension.
##After application of the broadcasting rules, the sizes of all arrays must match.

##INDEXING
#1. Indexing with Arrays of Indices
a17 = np.arange(12)**2
i = np.array( [ 1,1,3,8,5 ] ) # an array of indices
a17[i] # the elements of a17 at the positions ith position as above

j = np.array( [ [ 3, 4], [ 9, 7 ] ] ) # a bidimensional array of indices
a17[j] # the same shape as j

#assign value with indexing
a18 = np.arange(5)
a18[[1,3,4]] = 0 ## sets value 0 at 1,3,4 th position
#when the list of indices contains repetitions, the assignment is done several times, leaving behind the last value:
a18[[0,0,2]]=[1,2,3]
a18 # return arraywith latest array value reflected
##“a+=1” to be equivalent to “a=a+1”.
a18[[0,0,2]]+=1

#2. Indexing with Boolean Arrays
a19 = np.arange(12).reshape(3,4)
b19 = a19 > 4
a19[b19] # 1d array with the selected elements from b19 true false position


###LINEAR ALGEBRA
import numpy as np
a20 = np.array([[1.0, 2.0], [3.0, 4.0]])
a20.transpose()
u = np.eye(2) # unit 2x2 matrix; "eye" represents "I"
j = np.array([[0.0, -1.0], [1.0, 0.0]])
np.dot (j, j) # matrix product
np.trace(u) # trace

#Automatic Reshaping: Change the dimensions of an array, you can omit one of the sizes which will then be deduced automatically:
a21 = np.arange(30)
a21.shape = 2,-1,3 # -1 means "whatever is needed"
a21.shape

#Sorting array
import numpy as np
a22 = np.array([4,8,5,1,7])
a22.sort()

##########*********  NUMPY 1.11.0 community edition 1538 pages ******#################
# NumPy provides an N-dimensional array type, the ndarray, which describes a collection of “items” of the same type.
# Different ndarrays can share the same data, so that changes made in one ndarray may be visible in another.
# an ndarray can be a “view” to another ndarray, and the data it is referring to is taken care of by the “base” ndarray.

#A 2-dimensional array of size 2 x 3:
import numpy as np
x1 = np.array([[1, 2, 3], [4, 5, 6]]) # default int64 datatype
x2 = np.array([[1, 2, 3], [4, 5, 6]], np.int32) # set as int32 datatype
type(x1) # type of array
x1.dtype # data type
x1.shape
x1[1, 2] #indexing ; first is row & second is column. Indexed from 0 start.
y=x1[:,1]
y[0] = 9 # gets value of 1D array element
print(x1) # 9 gets set as 

##Constructing arrays : Arrays should be constructed using array, zeros or empty.
#Attributes:T,data,dtype,flags, flat,imag,real,size,itemsize,nbytes,ndim, shape,strides,ctypes,base
x3= np.array([[1.,2.],[3.,4.]])
x3.T # gives transpose result
x4 = np.array([1.,2.,3.,4.])
x4.T# transpose has no effect in Singular array
type(x4.dtype)
x5 = np.arange(1, 7).reshape(2, 3) #excess values are flushed out
x6 = np.arange(1, 7).reshape(3, 3) # gives array error & not extended if shape is larger than values provided.
x5.T
x5.flat = 33 # replaces all values of x5 with 33
## IMP below: Indexing of 2D array. as 0 onwards from LEFT to RIGHT & from TOP to BOTTOM
x5.flat[[1,5]] = 11 # replaces the 1st,5th index value as 11 in x5 array

## imaginary part of the array.
x = np.sqrt([1+0j, 0+1j])
x.imag
x.imag.dtype
## Real part
x.real
x.real.dtype
## Size: Number of elements in the array.
x = np.zeros((3, 5, 2), dtype=np.complex128)
x.size
np.prod(x.shape)
## itemsize: Length of one array element in bytes.
x = np.array([1,2,3], dtype=np.float64) # trying using np.complex128
x.itemsize
## byte: Total bytes consumed by the elements of the array.
# note: Does not include memory consumed by non-element attributes of the array object.
x.nbytes
np.prod(x.shape) * x.itemsize
## Dimension: 
x.ndim # one dimension of array
y = np.zeros((2, 3, 4))
y.ndim # 3 dimention of array
## Shape: Tuple of array dimensions.
x = np.array([1, 2, 3, 4])
x.shape
y = np.zeros((2, 3, 4)) # zero array of 3row & 4 cols
y.shape
y.shape = (3, 8) # array reshaping to 3*8 row,col
y
y.shape
y.shape = (3, 6) # cannot reshape array of size 24 into shape (3,6)
y.shape = (4, 6) # reshaped using 24 elements into multiple factors.
## Strides: Tuple of bytes to step in each dimension when traversing an array.
# The strides of an array tell us how many bytes we have to skip in memory to move to the next position along a certain axis.
# refer pg 11/1538 for details

## Scalars ####
# Python defines only one type of a particular data class (there is only one integer type, one floating-point type, etc.).
# In NumPy, there are 24 new fundamental Python types to describe different types of scalars.
dt = np.dtype([('name', np.str_, 16), ('grades', np.float64, (2,))]) # defining datatype of array elements
dt['name'] # get attribute of element name
dt['grades'] # get attribute of element grades
x = np.array([('Sarah', (8.0, 7.0)), ('John', (6.0, 7.0))], dtype=dt)
x[1]
x[1]['grades']
type(x[1])
type(x[1]['grades'])

## Indexing ###
# Three kinds of indexing available: 1. field access 2.basic slicing & 3.advanced indexing.
# Note: In Python, x[(exp1, exp2, ..., expN)] is equivalent to x[exp1, exp2, ..., expN]
# basic Slicing: constructed by start:stop:step notation inside of brackets
x1 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
x1[1:7:2] # from 1 to 7, step in 2nd. hence 1,3,5 is o/p
x1[-2:10] # o/p is 8,9. backward Index starts with 1, instead of 0
x1[-4:10] # -4th is 6 & 10th is 9, henceo/p is 6,7,8,9
x1[-5:10:2] #same as above with step as 2. o/p is 5,7,9
x1[-5:10:-2] ## gives empty as  array([], dtype=int64). DIRECTION is important for slicing
x1[-3:3:-1] # starts with 7,moves toward 3 & step as -1 to BACKWARD
x1[-3:3:2] # gives empty as  array([], dtype=int64). DIRECTION is important for slicing
##If step element is not given, default is 1.
## IMP  :: is the same as :
x1[5::] 
x1[5:] # starts with 5th index & print all
x1[:5] # starts with 0 & print till 5th index
#indexing of multiple array elements
y= np.array([[[1],[2],[3]], [[4],[5],[6]],[[11],[21],[31]]])
y.shape # (3,3,1). first 3 is number of repetition, second is for no of element in each block
y[1:2] # gets 1st index & 2nd deom back, hence 1 block . [4,5,6]
y[1:3] # gets two block based on start & end. [4,5,6] & [11,21,32]
#Ellipsis expand to the number of : objects needed
y[...,0] # 0/p is [[1],[2],[3]], [[4],[5],[6]],[[11],[21],[31]]

## Advanced Indexing ##
#Advanced indexing is triggered when the selection object, obj, is a non-tuple sequence object, an ndarray (of data
#type integer or bool), or a tuple with at least one sequence object or ndarray (of data type integer or bool).
# TYPES of Advacned Indexing: integer and Boolean.
#IMP: Advanced indexing always returns a copy of the data (contrast with basic slicing that returns a view)
z = np.array([[1, 2], [3, 4], [5, 6]])
z[[0, 1, 2], [0, 1, 0]] # first block refers block on z to pick & second soecifies individul element
z[[0, 1], [1, 1]] # from 0th & 1st block , op is [2,4]
z[[0, 1]] # gives all element of 0th & 1st block. Same as z[[0, 1],]
# Below index takes MINUS one from row & coluns
z[1:2,0:1] # row 1& col 0 witho/p as 3
z[1:2,[0,1]] # o/p is 3,4 based on range
z[0:2,0:1] #[1,3] as o/p
z[0:0,0:1] # gives Empty array if index is DUPLICATED. z[0:2,1:1]
z[0:2,] #[1,2,3,4] as o/p. Same as z[0:2]
z[2,1] # o/p 6 i.e 2nd row & 1st column intersection element
# more example on multiple index with ::
z1 = np.array([[ 0, 1, 2],[ 3, 4, 5],[ 6, 7, 8],[ 9, 10, 11]])
z1[1:3,0:1] # 1,2 as row & o as column is given as o/p. based on MINUS funda
z1[1:-2,0:1]# same logic as above
z1[1:-2,[0,2]] #only 0th & 2nd elemnt column as o/p
z1[[1,3],[0,2]] # 3 as o/p

## Boolean array indexing
x= np.array([[1., 2.], [np.nan, 3.], [np.nan, np.nan]])
x[~np.isnan(x)]
x1 = np.array([1., -1., -2., 3])
x1[x1 < 0] += 20 # adding a constant to all negative elements
##PROBLEM: From an array x2, select all rows which sum up to less or equal two:
x2 = np.array([[0, 1], [1, 1], [2, 2]])
rowsum = x2.sum(-1) # finding sum of each array block o/p is 1,2,4
rowsum.shape
x2.sum() # gives 7
x2.sum(1) # gives 1,2,4 ie sum of each block array
x2.sum() # gives 7
x2.sum(2) # gives array out of bound dimension
x2[rowsum <= 2, :] # getting block of array whose sum was less than or equal to 2 
##PROBLEM:  Use boolean indexing to select all rows adding up to an even number.
x3 = np.array([[ 0, 1, 2],[ 3, 4, 5],[ 6, 7, 8],[ 9, 10, 11]])
rowsum3 = x3.sum(-1)%2 # dividion gets o as rem
x3[rowsum3 == 0, :] # getting row whose remainder is 0 as EVEN

####Iterating Over Arrays with nditer()#####
##1. Single Array Iteration: nditer is to visit every element of an array
a=np.arange(6).reshape(2,3)
for i in np.nditer(a):
    print i ## remember toput proper INDENT & SPACE for functions
#Controlling Iteration Order: to visit the elements of an array in a specific order
#default is K, but can be set C Corder or F FortranOrder.
for i in np.nditer(a,order='F'):
    print i # 031425
for i in np.nditer(a,order='C'):
    print i #012345
#Modifying Array Values: By default, the nditer treats the input array as a read-only object.To modify the array elements,enable RW
for i in np.nditer(a, op_flags=['readwrite']): #make op_flags as RW
    i[...] = 2 * i # every value is multiplied by 2 provided its RW enabled
    print i
    
##  Using an External Loop
#better approach is to move the one-dimensional innermost loop into your code, external to the iterator.
for i in np.nditer(a,flags=['external_loop']):
    print i # o/p is [ 0  2  4  6  8 10]
for i in np.nditer(a,flags=['external_loop'],order='F'):
    print i # [o6],[28],[410]
##  Tracking an Index or MultiIndex
a = np.random.randn(6).reshape(2,3)
it = np.nditer(a, flags=['f_index'])
while not it.finished:
    print "%d <%d>" % (it[0], it.index),
    it.iternext()
# refer 118/1538 for more details

#Buffering the Array Elements: By enabling buffering mode, the chunks provided by the iterator to the inner loop can be made larger, significantly
#reducing the overhead of the Python interpreter.
for i in np.nditer(a, flags=['external_loop','buffered'], order='F'):
    print i # o/p is [0 3 1 4 2 5]

##2. Broadcasting Array Iteration; NumPy has a set of rules for dealing with arrays that have differing shapes which are applied whenever functions take
#multiple operands which combine element-wise. This is called broadcasting.
a = np.arange(3)
b = np.arange(6).reshape(2,3)
for x, y in np.nditer([a,b]):
    print "%d:%d" % (2*y,x), # prints 2 times of Y & X respectively. Order can be decided by us.
#IMP: Row count MUST match with column count, else Error is thrown. Like for Matrix case...

## Iterator-Allocated Output Arrays
# creating a function square which squares its input
def square(a):
    it = np.nditer([a, None]) #picker of one by one element. Needs None for 1d array. Cant be left un-assigned.
    for i, j in it:
        j[...] = i*i
    return it.operands[1] # can use -1 value inside operand
square([4,2,-7,5.9])
# Outer Product Iteration: The op_axes parameter needs one list of axes for each operandif multi-dimesniaol array used.
a = np.arange(3)
b = np.arange(8).reshape(2,4)
it = np.nditer([a, b, None], flags=['external_loop'],op_axes=[[0, -1, -1], [-1, 0, 1], None])
for x, y, z in it:
    z[...] = x*y
it.operands[2]

## Reduction Iteration
#Whenever a writeable operand has fewer elements than the full iteration space, that operand is undergoing a reduction
#The nditer object requires that any reduction operand be flagged as read-write, and only allows reductions when
#‘reduce_ok’ is provided as an iterator flag.
a = np.arange(24).reshape(2,3,4)
b = np.array(0)
for x, y in np.nditer([a, b], flags=['reduce_ok', 'external_loop'],
    op_flags=[['readonly'], ['readwrite']]):
    y[...] += x
np.sum(a) # o/p is 276

### Standard array subclasses along with MATRIX object handling  ###
# ***IMP***: refer 122-247 onwards. TO BE COVERED if required

## MASKED ARRAY ### Masked arrays are arrays that may have missing or invalid entries.
# The numpy.ma module provides a convenient way to address this issue, by introducing masked arrays.
# A masked array is the combination of a standard numpy.ndarray and a mask flag set as TRUE.
#The package ensures that masked entries are not used in computations
import numpy as np
import numpy.ma as ma
# numpy.ma module is the MaskedArray class, which is a subclass of numpy.ndarray.
x = np.array([1, 2, 3, -1, 5])
#masking 4th element of array X.
mx = ma.masked_array(x, mask=[0, 0, 0, 1, 0])
# compute the mean of the dataset, without taking the invalid data into account
mx.mean() # 2.75, if not masked value is 2.0
# create an array with the second element invalid
y = ma.array([1, 2, 3], mask = [0, 1, 0]) # 2 gets invalid coz of Mask
#create a masked array where all values close to 1.e20 are invalid,
z = ma.masked_values([1.0, 1.e20, 3.0, 4.0], 1.e20)

## Constructing masked arrays
# Possible ways to construct a masked array as below:
#1. directly invoke the MaskedArray class.
#2. use the two masked array constructors, array and masked_array.
x = np.array([1, 2, 3])
x.view(ma.MaskedArray) # gives set of data
y = np.array([(1, 1.), (3, 2.)], dtype=[('a',int), ('b', float)])
y.view(ma.MaskedArray)

## asanyarray: Similar to asarray, but conserves subclasses. ASARRAY conserved subclasses
x = np.arange(10.).reshape(2, 5)
np.ma.asanyarray(x)
type(np.ma.asanyarray(x))

## Other possibilities as below: 
#asarray(a[, dtype, order]): Convert the input to a masked array of the given data-type.
#asanyarray(a[, dtype]): Convert the input to a masked array, conserving subclasses.
#fix_invalid(a[, mask, copy, fill_value]):Return input with invalid data masked and replaced by a fill value.
#masked_equal(x, value[, copy]): Mask an array where equal to a given value.
#masked_greater(x, value[, copy]): Mask an array where greater than a given value.
#masked_greater_equal(x, value[, copy]):Mask an array where greater than or equal to a given value.
#masked_inside(x, v1, v2[, copy]):Mask an array inside a given interval.
#masked_invalid(a[, copy]): Mask an array where invalid values occur (NaNs or infs).
#masked_less(x, value[, copy]): Mask an array where less than a given value.
#masked_less_equal(x, value[, copy]): Mask an array where less than or equal to a given value.
#masked_not_equal(x, value[, copy]): Mask an array where not equal to a given value.
#masked_object(x, value[, copy, shrink]): Mask the array x where the data are exactly equal to value.
#masked_outside(x, v1, v2[, copy]): Mask an array outside a given interval.
#masked_values(x, value[, rtol, atol, copy, ...]): Mask using floating point equality.
#masked_where(condition, a[, copy]): Mask an array where a condition is met.

## Accessing the mask data: accessible through its mask attribute
# Another possibility is to use the getmask and getmaskarray functions.
#1. Accessing only the valid entries: simply with the ~ operator
x = ma.array([[1, 2], [3, 4]], mask=[[0, 1], [1, 0]])
x[~x.mask] # gives Non-masked data set
x[x.mask] # gives NULL unless we UNMASK it
#2. Another way to retrieve the valid data using Compressed method
x.compressed() # *** Note that the output of compressed is always 1D.

## Masking an entry: by assigning the special value masked to them
x = ma.array([1, 2, 3,4,5,9]) # single D array
x[0] = ma.masked # masks the oth index element
x[0:2] = ma.masked # 2th value is not masked
y = ma.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) # multiple D array
y[(0, 2), (1, 0)] = ma.masked # mask sthe oth & 2nd block from y array
z = ma.array([1, 2, 3, 4])
z[:-2] = ma.masked # 2th value is not masked

## Unmasking an entry
#1. approach 1: we can just assign one or several new valid values to them
x = ma.array([1, 2, 3], mask=[0, 0, 1])
x[-1] = 5 # assigning valid valye 5 to masked value set.
# ** Note: Unmasking an entry by direct assignment will silently fail if the masked array has a hard mask
xh = ma.array([1, 2, 3], mask=[0, 0, 1], hard_mask=True)# masked HARD 
xh[-1] = 5 # doesn't change the mask to 5
xh.soften_mask() # softening the HARD MASK
xh[-1] = 5  # value unmasked & assigned as 5 nOW
# To unmask all masked entries of a masked array (PROVIDED the mask isn’t a hard mask)
xall = ma.array([11, 22, 33], mask=[0, 0, 1], hard_mask=True)
xall.mask = ma.nomask
# using softening technbique will help here: xall.soften_mask()

## Indexing and slicing of masked array
# 1D array
x = ma.array([1, 2, 3], mask=[0, 0, 1])
x[0] # gives 1
x[-1] # gives MAsKED or x[-1] is ma.masked query
# 2D array
y = ma.masked_array([(1,2), (3, 4)],mask=[(0, 0), (0, 1)],dtype=[('a', int), ('b', int)])
y[0] # gives (1,2)
y[-1] # 3 as 4 is masked for -1th block element
x1 = ma.array([1, 2, 3, 4, 5], mask=[0, 1, 0, 0, 1])
mx = x1[:3]
mx[1] = -1
x1.mask
x1.data

## Examples on masked set ##
#eg1.. Data with a given value representing missing data. say -9999 represents missing data
x = [0.,1.,-9999.,3.,4.]
mx = ma.masked_values (x, -9999.) # here -9999 is considered MASK in x
mx.mean() # mean is 2
mx - mx.mean() # masked value from mx is kept as blank, not -2
mx.anom() # maksed value is shown as true
# get mx data, but with the missing values replaced by the average value
mx.filled(mx.mean()) # prints mean 2 for masked -9999 position index
#eg2. Numerical operations: without worrying about missing values, dividing by zero, square roots of negative numbers, etc.
x = ma.array([1., -1., 3., 4., 5., 6.], mask=[0,0,0,0,1,0])
y = ma.array([1., 2., 0., 4., 5., 6.], mask=[0,0,0,0,0,1])
np.sqrt(x/y) # operation performed element wise & error thrown for improper cases.
#eg3. Ignoring extreme values: random array b./w 0 & 1. Lets find mean of values b/w 0.1 & 0.9
d = np.random.uniform(low=0, high=1, size=(50))
ma.masked_outside(d, 0.2, 0.8).mean() # calculate mean of all numbers b/w 0.2 & 0.8 based on array d

## Constants of the numpy.ma module: The masked constant is a special case of MaskedArray, with a float datatype and a null shape. It is used to
#test whether a specific entry of a masked array is masked, or to mask one or several entries of a masked array:
x = ma.array([11, 22, 31], mask=[0, 1, 0])
x[1] is ma.masked
x[-1] = ma.masked
x # having "--" for masked values  

## Attributes and properties of masked arrays
# maskedarray base: Base object if memory is from some other object.
x = np.array([13,23,33,34])
x.base is None # o/p is true as it has no memeory/ view shared.
# Slicing creates a view, whose memory is shared with x:
y = x[2:]
y.base is x # y is shared from x & hence o/p is TRUE
x.dtype
type(x.dtype)
# Itemsize: Length of one array element in bytes.
x.itemsize # 8 bytes size for each element
# nbytes: Total bytes consumed by the elements of the array.
# IMP: Does not include memory consumed by non-element attributes of the array object.
x.nbytes     # 32 as its 4*8 bytes
np.prod(x.shape) * x.itemsize # aliter way of counting total byte size
# ndim: Number of array dimensions.
x.ndim
y = np.zeros((2, 3, 4)) # having 24 elements in array
y.ndim
# shape: Tuple of array dimensions.
## IMP :May be used to “reshape” the array, as long as this would not require a change in the total number of elements
x.shape # o/p is (4,) as it has 4 elements of 1d array 
y.shape
y.shape = (3, 8) # changes the shape to 3*8 =, in accordance with 24 elemenst
y.shape = (3, 6) # cant change to 18 for 24 elements
# size: Number of elements in the array.
x1 = np.zeros((3, 5, 2), dtype=np.complex128)
x1.size
# Compressed: retrurns non masked values
x.compressed()
# fill masked values with random vakeus say -9999
x1 = np.ma.array([1,2,3,4,5], mask=[0,0,1,0,1], fill_value=-999) # cant use #,special symbol to replace
x1.filled() # retruns array filled with -999 for masked cases.
x2 = np.ma.array(np.matrix([[1, 2], [3, 4]]), mask=[[0, 1], [1, 0]])
x2.filled() ## FILLS 999999 as defualt is no value is passed in original array.

## Shape manipulation ##
# flatten(order=’C for row’ or F for Column): Return a copy of the array collapsed into one dimension.
a = np.array([[1,2], [3,4]])
afC = a.flatten()  #get series sort conversion of array a
afF = a.flatten('F')
# ravel(order=’C’): Returns a 1D version of self, as a view.
x = np.ma.array([[1,2,3],[4,5,6],[7,8,9]], mask=[0] + [1,0]*4)
xr = x.ravel() # flattening with masked effect in place
# reshape: Give a new shape to the array without changing its data.The result is a view on the original array
x = np.ma.array([[1,2],[3,4]], mask=[1,0,0,1]) # 2D array
x.shape # original shape of (2,2)
xrs = x.reshape((4,1)) # shape converted to (4,1)
# transpose(*axes): Returns a view of the array with axes transposed.
a = np.array([[1, 2, 3], [3, 4, 5]]) 
a.transpose()
a.transpose((1, 0))
a.transpose(1, 0) # all three gives similar result here

## Item selection and manipulation ##
# argmax(axis=None, fill_value=None, out=None): Returns array of indices of the maximum values along the given axis. Masked values are treated as if they had
#the value fill_value.
a = np.random.randn(6).reshape(2,3)
a.argmax() #gives index of max value
a.argmax()
a1 = np.ma.array([11, 2, 31, 14, 51, 9], mask=[0,0,1,1,1,0])
a1.argmax() # masked value not taken into account
a1.argmin() 
# sort(): 
a1.argsort()
#fill value: fills specific value
a = np.array([1, 2])
a.fill(0)
a = np.empty(2)
a.fill(1)

##        *******286-438 for other properies related to MASKED ARRAYS            ***********

##### **** The Array Interface **** #####
# The array interface (sometimes called array protocol) was created in 2005 as a means for array-like Python objects to re-use each other’s data buffers intelligently whenever possible
# The homogeneous N-dimensional array interface is a default mechanism for objects to share N-dimensional array memory and information
# There are two ways to use the interface: A Python side and a C-side. Both are separate attributes.

#### **********   DATETIME refer 438-453 of 1538 pdf     ###

######## UNIVERSAL FUNCTIONS (UFUNC) ##########################
# A universal function (or ufunc for short) is a function that operates on ndarrays in an element-by-element fashion, supporting array broadcasting, type casting, and several other standard features.
# a ufunc is a “vectorized” wrapper for a function that takes a fixed number of scalar inputs and produces a fixed number of scalar outputs
## A set of arrays is called “broadcastable” to the same shape if the above rules produce a valid result, i.e., one of the following is true:
#1. The arrays all have exactly the same shape.
#2. The arrays all have the same number of dimensions and the length of each dimensions is either a common length or 1.
#3. The arrays that have too few dimensions can have their shapes prepended with a dimension of length 1 to satisfy property 2.

## Use of internal buffers: used for misaligned data, swapped data, and data that has to be converted from one data type to another
# The size of internal buffers is settable on a per-thread basis.
# There can be up to 2(n inputs + n outputs ) buffers of the specified size created to handle the data from all the inputs and outputs of a ufunc
# The default size of a buffer is 10,000 elements
# numpy.setbufsize(size) is thefunctin used.

## Error handling:  is controlled on a per-thread basis, and can be configured using the functions
#1.seterr([all, divide, over, under, invalid]): Set how floating-point errors are handled.
#2.seterrcall(func):Set the floating-point error callback function or log object.

# seterrcall: Set a callback function for the ‘call’ mode.
old_settings = np.seterr(all='ignore') #seterr to known value
np.seterr(over='raise')
np.seterr(**old_settings) # reset to default
# refer 458 onwards for more details...........

### UFUNC behaviour
# nin : The number of inputs.
np.add.nin # i/p is 2
np.multiply.nin # i/p is 2
np.power.nin # i/p is 2
np.exp.nin # i/p is 1
# nout: The number of outputs.
np.add.nout # o/p is 1
np.multiply.nout # o/p is 1
np.power.nout # o/p is 1
np.exp.nout # o/p is 1
# nargs: The number of argument,including optional ones.
np.add.nargs # 3
np.multiply.nargs # 3
np.power.nargs # 2
np.exp.nargs # 1
# ntypes: The number of types: there are 18 total
np.add.ntypes #18
np.multiply.ntypes #18
np.power.ntypes #17
np.exp.ntypes #7
np.remainder.ntypes #14
np.add.types # gives all 18 type of add ufunc
np.multiply.types  # gives all 18 type of multiply ufunc
np.power.types # gives all 17 type of power ufunc
np.exp.types # gives all 7 type of add ufunc
np.remainder.types # gives all 14 type of add ufunc
# identity: The identity value. Data attribute containing the identity element for the ufunc, if it has one
np.add.identity #0
np.multiply.identity #1
np.power.identity#1
np.exp.identity #None

### Methods ####
# All ufuncs have five(reduce+accumulate+reduceat+outer+at) methods.
#These methods only make sense on ufuncs that take two input arguments and return one output argument
#1. reduce(a, axis=0, dtype=None, out=None, keepdims=False):Reduces a‘s dimension by one, by applying ufunc along one axis.
# add.reduce() is equivalent to sum()
X = np.arange(8).reshape((2,2,2))
print X # 0,1,2 etc are the axis
np.add.reduce(X, 0) # removes by ONE Dimension
np.add.reduce(X) # confirm: default axis value is 0
np.add.reduce(X, 1)
np.add.reduce(X, 2)
#2. accumulate(array, axis=0, dtype=None, out=None): Accumulate the result of applying the operator to all elements.
# add.accumulate() is equivalent to np.cumsum()
# for 1D array
np.add.accumulate([2, 3, 5]) # 2,5,10
np.multiply.accumulate([2, 3, 5]) # 2,,6,30
# for 2D array
I = np.eye(2)
np.add.accumulate(I, 0)
np.add.accumulate(I) # no axis specified = axis zero
np.add.accumulate(I, 1) # Accumulate along axis 1 (columns)
#3. reduceat(a, indices, axis=0, dtype=None, out=None): Performs a (local) reduce with specified slices over a single axis.
#4. outer(A, B): Apply the ufunc op to all pairs (a, b) with a in A and b in B.
#5.at(a, indices, b=None): Performs unbuffered in place operation on operand ‘a’ for elements specified by ‘indices’.
# refer 466/1538 pages for further details

### Available ufuncs ##############
# currently more than 60 universal functions defined in numpy on one or more types
# each ufunc operates element-by-element.
# Types as:
#1. Math operations; add, subtract,multiply,divide,exp etc.
#2. Trigonometric functions: sin,cos etc.
#3. Bit-twiddling functions: bitwise_and, invert etc
#4. Comparison functions: greater, less_equal etc
#5. Floating functions: isreal, iscomplex, floor,ceil,trunc etc.

######### ***** ROUTINES ********** #################
## Array creation routines
# Ones and zeros
#1. empty(shape, dtype=float, order=’C’):Return a new array of given shape and type, without initializing entries.
np.empty([2, 2]) # random values inserted
np.empty([2, 2], dtype=int) # random integer values inserted
#2. empty_like(a, dtype=None, order=’K’, subok=True): Return a new array with the same shape and type as a given array.
a = ([11,12,31], [14,51,61])
np.empty_like(a) # array of shape like a.
#3. eye(N, M=None, k=0, dtype=<type ‘float’>):Return a 2-D array with ones on the diagonal and zeros elsewhere.
np.eye(2, dtype=int) # o is the main diagonal, left2right
np.eye(3, k=1) # upper to main 0th diagonal
np.eye(3, k=2) # k is index of diagonal. 
np.eye(3, k=-1) # lower diagonal to 0th index diagonal
#  0 (the default) refers to the main diagonal, a positive value refers to an upper diagonal, and a negative value to a lower diagonal.
#4. identity(n, dtype=None): Return the identity array.The identity array is a square array with ones on the main diagonal.
np.identity(3)
#5.ones(shape, dtype=None, order=’C’): Return a new array of given shape and type, filled with ones.
np.ones(5) # 1D array having elelments as 1
np.ones((5,), dtype=np.int)
np.ones((2, 2)) # 2D array having 2rows & 2 columns
#6. ones_like(a, dtype=None, order=’K’, subok=True): Return an array of ones with the same shape and type as a given array.
b = np.arange(6, dtype=np.float).reshape((2, 3))
np.ones_like(b)
#7. zeros(shape, dtype=float, order=’C’): Return a new array of given shape and type, filled with zeros.
np.zeros((5,), dtype=np.int)
np.zeros((2, 1)) # 2D array having 2 rows & 1 column
np.zeros((2,), dtype=[('x', 'i4'), ('y', 'i4')]) # CUSTOM DATA TYPE
# rows 2, but have column as defined to accomodate dtype definition
#8. zeros_like(a, dtype=None, order=’K’, subok=True): Return an array of zeros with the same shape and type as a given array.
c = np.arange(6, dtype=np.float).reshape((2, 3))
np.zeros_like(c)
#9. full(shape, fill_value, dtype=None, order=’C’): Return a new array of given shape and type, filled with fill_value.
np.full((2, 2), np.inf) # 2x2 array having np.inf as element
np.full((2, 2), 999, dtype=np.int)
#10. full_like(a, fill_value, dtype=None, order=’K’, subok=True):Return a full array with the same shape and type as a given array.
d = np.arange(6, dtype=np.int)
d1 = np.random.randn(6)
np.full_like(d1, 111) # works well for int or other any random data set 
#11. array: Create an array.
np.array([1, 2, 3], dtype=complex) # o/p is complex array set
#12. asarray(a, dtype=None, order=None): Convert the input to an array.
a = [11, 21]
np.asarray(a)
# ndarray subclasses are not passed through when asarray is applied.
##13. asanyarray(a, dtype=None, order=None): Convert the input to an ndarray, but pass ndarray subclasses through.
a = [1, 2]
np.asanyarray(a)
##14. ascontiguousarray(a, dtype=None):Return a contiguous array in memory (C order).
x = np.arange(6).reshape(2,3)
np.ascontiguousarray(x, dtype=np.float32)
## 15.asmatrix(data, dtype=None): Interpret the input as a matrix.
# Unlike matrix, asmatrix does not make a copy if the input is already a matrix or an ndarray.
x = np.array([[13, 23], [33, 34]])
m = np.asmatrix(x)
x[0,0] = 55 # replaces 13 as 55 for matrix 
# 16. copy(a, order=’K’): Return an array copy of the given object.
xcopy= np.copy(x) # where x is any array 
# 17. fromfile(file, dtype=float, count=-1, sep=’‘):Construct an array from data in a text or binary file.
#18. fromfunction(function, shape, **kwargs): Construct an array by executing a function over each coordinate.
#19. :fromiter(iterable, dtype, count=-1): Create a new 1-dimensional array from an iterable object.
iterable = (x*x for x in range(5)) # x squre with range from 0-5
np.fromiter(iterable, np.float) # creating array by iterating the range
#20. .fromstring(string, dtype=float, count=-1, sep=’‘): A new 1-D array initialized from raw binary or text data in a st
np.fromstring('\x01\x02', dtype=np.uint8)
np.fromstring('\x01\x02\x03\x04\x05', dtype=np.uint8, count=3)
#21. loadtxt: load data from text file
from io import StringIO # StringIO behaves like a file object
c = StringIO("0 1\n2 3")
np.loadtxt(c)
d = StringIO("M 21 72\nF 35 58")
np.loadtxt(d, dtype={'names': ('gender', 'age', 'weight'),'formats': ('S1', 'i4', 'f4')})

### Numerical ranges *#######
#1. arange( [ start ] , stop [ , step ] , dtype=None): Return evenly spaced values within a given interval.
np.arange(3,7,2) # 3,5 is o/p. STOP digit not included in o/p array
np.arange(3.0) # 0,1,2
#2.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None):Return evenly spaced numbers over a specified interval.
np.linspace(2.0, 3.0, num=5) # will include start & stop point
np.linspace(2.0, 3.0, num=5, endpoint=False) # will NOT include Stop points
np.linspace(2.0, 3.0, num=5, retstep=True) # gives Multi-Dimensional Array
#3. logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None):Return numbers spaced evenly on a log scale.
np.logspace(2.0, 3.0, num=4)
np.logspace(2.0, 3.0, num=4, endpoint=False) #endpoint Not included in base call
np.logspace(2.0, 3.0, num=4, base=2.0) #base 2 used in calculation, instead of default 10.
#4.mgrid: nd_grid instance which returns a dense multi-dimensional “meshgrid”.
np.mgrid[0:5,0:5] # create 0-5 matrix
np.mgrid[1:5,3:7] # returns 2D array having COMMON count of elements
#5. ogrid: instance which returns an open multi-dimensional “meshgrid”.
from numpy import ogrid
ogrid[-1:1:5j] # array([-1. , -0.5,  0. ,  0.5,  1. ])

#### Matrix building Refer 504/1538 for details

#### The Matrix class
#1. mat(data, dtype=None):Interpret the input as a matrix.
x = np.array([[1, 2], [3, 4]])
m = np.asmatrix(x)
#2. .bmat(obj, ldict=None, gdict=None):Build a matrix object from a string, nested sequence, or array.
A=np.mat('1 1; 1 1')
B=np.mat('2 2; 2 2')
C=np.mat('3 3; 3 3')
D=np.mat('4 4; 4 4')
matrix1= np.bmat([[A, B], [C, D]]) # create matrix from A,B,C,D
matrix2= np.bmat('A,B; C,D') # o/p same as above case
matrix3= np.bmat([[A, B,C, D]]) # check the aligment of blocks
matrix4= np.bmat([A, B,C, D]) # o/p same as above case


#### Array manipulation routines ######

###  Basic operations
#1. .copyto(dst, src, casting=’same_kind’, where=None): Copies values from one array to another, broadcasting as necessary.
### Changing array shape
#1..reshape(a, newshape, order=’C’): Gives a new shape to an array without changing its data.
a = np.arange(6).reshape((3, 2)) # 2D having 3 rows & 2 columns
np.reshape(a, (2, 3)) # reshaping to 3 by 2 D array from previous 3by2 array of a
np.reshape(np.ravel(a), (2, 3)) # same o/p as above 
np.reshape(a, (2, 3), order='F') # Fortran-like index ordering

b = np.array([[1,2,3], [4,5,6]])
np.reshape(b, 6) # conversion from 2D to 1D array
np.reshape(b, 6, order='F')
np.reshape(b, (3,-1)) #t he unspecified value is inferred to be 3 to make it 6
np.reshape(b, (2,-1))  # the unspecified value is inferred to be 3 to make it 6
#2. ravel(a, order=’C’:Return a contiguous flattened array.
# A 1-D array, containing the elements of the input, is returned. A copy is made only if needed.
x1 = np.array([[1, 2, 3], [4, 5, 6]])
xr = np.ravel(x1) # returns flat array
x1.reshape(-1) #o/p same as above
np.ravel(x1, order='F') # Fortran set ravelling opeartion
x2= np.arange(12).reshape(2,3,2)
x2.ravel(order='C')
x2.ravel(order='K')
#3. .flat:A 1-D iterator over the array.
x = np.arange(1, 7).reshape(2, 3)
x.flat[3] # 4 is o/p here
y= x.T # pay attnetin to how transpose  is done & counted
y.flat[3] # 3 is o/p here
x.flat = 3 # produces array with 3
x.flat[[1,4]] = 1 # puts 1 at 1,4 indexed value
#4. flatten(order=’C’):Return a copy of the array collapsed into one dimension.
a = np.array([[1,2], [3,4]])
a.flatten() # o/p is array([1, 2, 3, 4])

## Transpose-like operations
#1. moveaxis(a, source, destination):Move axes of an array to new positions.Other axes remain in their original order.
x = np.zeros((3, 4, 5)) # shape is 3,4,5
np.moveaxis(x, 0, -1).shape # axis moved to 4,5,3
np.moveaxis(x, -1, 0).shape  # axis moved to 5,3,4
#2. rollaxis(a, axis, start=0):Roll the specified axis backwards, until it lies in a given position.
a = np.ones((3,4,5,6))
np.rollaxis(a, 3, 1).shape # o/p is  (3, 6, 4, 5) dimension
#3.swapaxes(a, axis1, axis2): Interchange two axes of an array.
x = np.array([[1,2,3]]) # 1D array
np.swapaxes(x,0,1) # swapping row to column & vice versa
y = np.array([[12,22,32],[14,24,34]]) # 2D array
np.swapaxes(y,0,1) # swapping row to column & vice versa
z = np.array([[[0,1],[2,3]],[[4,5],[6,7]]])
np.swapaxes(z,0,2)
#4. .transpose(a, axes=None):Permute the dimensions of an array.

## Changing number of dimensions: #### pg 518
## Changing kind of array ### pg 525

### Concatenate: 
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6]])
c = np.array([[7], [8]])
np.concatenate((a, b), axis=0) # added as row index
np.concatenate((a, c), axis=1) # added as Column index
np.concatenate((a, b.T), axis=1) # added transposed array to a array
### Stack:
a = np.array([1, 2, 3])
b = np.array([2, 3, 4])
np.stack((a, b)) # default horizontal STACKING
np.stack((a, b), axis=-1) # vertical STACKING by saying axis -1
# column_stack(tup): Stack 1-D arrays as columns into a 2-D array.
a = np.array((1,2,3))
b = np.array((2,3,4))
np.column_stack((a,b)) # 2D array having a,b as columns
# horizotal_stack:  Stack 1-D arrays as rows into a 1-D array.
a = np.array((1,2,3))
b = np.array((2,3,4))
np.hstack((a,b)) # create 1 D array with a, b stacked one after another
# vstack(tup): Stack arrays in sequence vertically (row wise).
a = np.array([1, 2, 3])
b = np.array([2, 3, 4])
np.vstack((a,b))
### Split:
x = np.arange(9.0)
np.split(x, 3) # splits array in 3 equal part. if 4, then gives error.
y = np.arange(8.0)
np.split(y, [3, 5, 6, 10]) # splits into 4 parts. 
# first having 3, then 2 to make it 5, then 1 to make it 6 & last has pending elements.
#array_split(ary, indices_or_sections, axis=0): Split an array into multiple sub-arrays.
x = np.arange(8.0)
np.array_split(x, 3)
# hsplit:
x = np.arange(16.0).reshape(4, 4)
np.hsplit(x, 2) # splits horizonatly into 2 2D aray
np.hsplit(x, np.array([3, 6]))
x2 = np.arange(8.0).reshape(2, 2, 2)
np.hsplit(x2, 2)
# vsplit(ary, indices_or_sections): Split an array into multiple sub-arrays vertically (row-wise).
np.vsplit(x, 2) # using same x array from above case
np.vsplit(x, np.array([3, 6])) # using same x array from above case

### Tiling arrays
#1. tile(A, reps):Construct an array by repeating A the number of times given by reps.
a = np.array([3, 13, 32])
np.tile(a, 2) # new array with a repeated
np.tile(a, (2, 2)) # creates 2by2 2D array with a's element
c = np.array([1,2,3,4])
np.tile(c,(4,1)) # c's repeated 4by1 array type.
#2. repeat(a, repeats, axis=None): Repeat elements of an array.
x = np.array([[1,2],[3,4]])
np.repeat(x, 2) # each element is repeated 2'ice. array([1, 1, 2, 2, 3, 3, 4, 4])
np.repeat(x, 3, axis=1)
np.repeat(x, [1, 2], axis=0) # first block1'ce & second block twice

#### Adding and removing elements
#1. delete(arr, obj, axis=None):Return a new array with sub-arrays along an axis deleted.
# allows further use of mask.
arr = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
np.delete(arr, 1, 0)
np.delete(arr, [1,3,5], None)
#2. insert(arr, obj, values, axis=None):Insert values along the given axis before the given indices.
a = np.array([[1, 1], [2, 2], [3, 3]])
np.insert(a, 4, 57) #inserts 57 at the 4th index value
np.insert(a, 1, 55, axis=1) # fills colum 1st with 55.
#3. append(arr, values, axis=None): Append values to the end of an array.
np.append([1, 2, 3], [[4, 5, 6], [7, 8, 9]]) #we can use a,b,c as appending action to create array
np.append([[1, 2, 3], [4, 5, 6]], [[7, 8, 9]], axis=0) # appending along row axis
#4. resize(a, new_shape): Return a new array with the specified shape.
a=np.array([[0,1],[2,3]])
np.resize(a,(2,3)) # creates 2by3 array with elements from a array
np.resize(a,(3,4))
b = np.array((0, 0, 0, 1, 2, 3, 0, 2, 1, 0))
np.trim_zeros(b) # array([1, 2, 3, 0, 2, 1]). removes start&end zero's from array a
np.trim_zeros(a, 'b')
#5. unique(ar, return_index=False, return_inverse=False, return_counts=False):Find the unique elements of an array.
a=np.array([[0,1,1,3],[2,2,3,3]])
np.unique(a)

### Rearranging elements
#fliplr(m): Flip array in the left/right direction.
#flipud(m): Flip array in the up/down direction.
#reshape(a, newshape[, order]): Gives a new shape to an array without changing its data.
#roll(a, shift[, axis]): Roll array elements along a given axis.
#rot90(m[, k]): Rotate an array by 90 degrees in the counter-clockwise direction.

### String operations #####
## String operations: refer 560
## Comparison
#equal(x1, x2): Return (x1 == x2) element-wise.
#not_equal(x1, x2): Return (x1 != x2) element-wise.
#greater_equal(x1, x2): Return (x1 >= x2) element-wise.
#less_equal(x1, x2): Return (x1 <= x2) element-wise.
#greater(x1, x2): Return (x1 > x2) element-wise.
#less(x1, x2): Return (x1 < x2) element-wise.
### String information
#1. Count: 
c = np.array(['aAaAaA', ' aA ', 'abBAABba'])
np.char.count(c, 'A') # counts instances of A in arry a. o/p is array([3, 1, 2])
np.char.count(c, 'A', start=1, end=4) # o/p is array([2, 1, 1]). CountingIndex from 1 onwards

### Datetime Support Functions ###
## Business Day Functions
#busdaycalendar: A business day calendar object that efficiently stores information defining valid days
#is_busday(dates[, weekmask, holidays, ...]): Calculates which of the given dates are valid days, and which are not.
#busday_offset(dates, offsets[, roll, ...]): First adjusts the date to fall on a valid day according to the roll rule, then applies o
#busday_count(begindates, enddates[, ...]): Counts the number of valid days between begindates and enddates, not including the
#1. busdaycalendar: default valid days are Monday through Friday (“business days”)
bdd = np.busdaycalendar( holidays=['2011-07-01', '2011-07-04', '2011-07-17'])
bdd.weekmask # valid days
bdd.holidays # any holidays on array list

### Data type routines, refer 614 of 1538
### Creating data types, refer 619 of 1538
### Data type information, refer 624 of 1538
### Floating point error handling, refer 634 of 1538
### Discrete Fourier Transform (numpy.fft), refer 640 of 1538
### Real Fourier Transform (numpy.fft), refer 651 of 1538
### Functional programming, refer 674 of 1538

### Financial functions ###
#fv(rate, nper, pmt, pv[, when]): Compute the future value.
#pv(rate, nper, pmt[, fv, when]): Compute the present value.
#npv(rate, values): Returns the NPV (Net Present Value) of a cash flow series.
#pmt(rate, nper, pv[, fv, when]): Compute the payment against loan principal plus interest.
#ppmt(rate, per, nper, pv[, fv, when]): Compute the payment against loan principal.
#ipmt(rate, per, nper, pv[, fv, when]): Compute the interest portion of a payment.
#irr(values): Return the Internal Rate of Return (IRR).
#mirr(values, finance_rate, reinvest_rate): Modified internal rate of return.
#nper(rate, pmt, pv[, fv, when]): Compute the number of periodic payments.
#rate(nper, pmt, pv, fv[, when, guess, tol, ...]): Compute the rate of interest per period.

## Numpy-specific help functions
# lookfor(what, module=None, import_modules=True, regenerate=False, output=None): Do a keyword search on docstrings.
np.lookfor('binary representation')
np.lookfor('hsplit')

### indexing Routines, refer 682 of 1538

### Input and output ###
## Numpy binary files (NPY, NPZ)
#1. load(file, mmap_mode=None, allow_pickle=True, fix_imports=True, encoding=’ASCII’): Load arrays or pickled objects from .npy, .npz or pickled files.
# Store data to disk, and load it again:
np.save('/tmp/filename', np.array([[1, 2, 3], [4, 5, 6]])) # filename is created in tmp folder
np.load('/tmp/filename.npy')  # loading filename from tmp
#Store compressed data to disk, and load it again:
a=np.array([[1, 2, 3], [4, 5, 6]])
b=np.array([1, 2])
np.savez('/tmp/123.npz', a=a, b=b) # saving multiple arrays
data = np.load('/tmp/123.npz')

## Text files ## 
#1.loadtxt: Load data from a text file. 
from io import StringIO
c = StringIO("0 1\n2 3")
np.loadtxt(c)

### Text formatting options ###
#set_printoptions([precision, threshold, ...]): Set printing options.
#get_printoptions(); Return the current print options.
#set_string_function(f[, repr]): Set a Python function to be used when pretty printing arrays.

### Logic Fucntions ###
## Truth value testing
#1. all(a, axis=None, out=None, keepdims=False): Test whether all array elements along a given axis evaluate to True.
np.all([[True,False],[True,True]]) # gives false as o/p
np.all([[True,False],[True,True]], axis=0) #array([ True, False], dtype=bool)
#Note: Not a Number (NaN), positive infinity and negative infinity evaluate to True because these are not equal to zero.
np.all([-1, 4, 5]) # true
np.all([1.0, np.nan]) #true
#2. any(a, axis=None, out=None, keepdims=False): Test whether any array element along a given axis evaluates to True.
np.any([[True, False], [True, True]]) # false
np.any([[True, False], [False, False]], axis=0) # true false
np.any([-1, 0, 5])
np.any(np.nan) #true.  naN,+/-number infinity are not equal to ZERO

## Array contents
#1. isfinite: Test element-wise for finiteness (not infinity or not Not a Number).
# Note: Not a Number, positive infinity and negative infinity are considered to be non-finite.
np.isfinite(np.nan) # false
np.isfinite(0) # true
np.isfinite(np.inf) # false
#2. isinf: Test element-wise for positive or negative infinity.
np.isinf(np.inf) # true
np.isinf(np.nan) # false
np.isinf(np.NINF) #true
#3. isnan: Test element-wise for NaN and return result as a boolean array.
np.isnan(np.nan) # true
np.isnan(np.inf) # false
#4. isneginf(x, y=None): Test element-wise for negative infinity, return result as bool array.
np.isneginf(np.NINF) #true
np.isneginf(np.inf) # fakse
#5. isposinf(x, y=None):Test element-wise for positive infinity, return result as bool array.

## Array type testing
#1. iscomplex(x): Returns a bool array, where True if input element is complex.
np.iscomplex([1+1j, 1+0j, 4.5, 3, 2, 2j]) #array([ True, False, False, False, False, True], dtype=bool
#2. iscomplexobj(x): Check for a complex type or an array of complex numbers.
np.iscomplexobj(1) # false
np.iscomplexobj(1+0j) # true
# others as :
#isreal(x): Returns a bool array, where True if input element is real.
#isrealobj(x): Check for a complex type or an array of complex numbers.
#isscalar(num): Returns True if the type of num is a scalar type.

## Logical operations
#1. logical_and(x1, x2 [ , out ] ) = <ufunc ‘logical_and’>: Compute the truth value of x1 AND x2 element-wise.
np.logical_and(True, False) # false
np.logical_and([True, False], [False, False]) # false, true
x = np.arange(5)
np.logical_and(x>1, x<4) # [False, False, True, True, False]
#2. logical_or
#3. logical_not:Compute the truth value of NOT x element-wise.
np.logical_not(3) # true
np.logical_not([True, False, 0, 1]) # [False,  True,  True, False]
np.logical_not(x<3) # where x= np.arange(5).o/p is [False, False, False,  True,  True]
#4. logical_xor: 

## Note: 798-to-1180 left for future rading if required. ******* #####

### Random sampling (numpy.random) ###
## Simple random data
#1. random.rand(d0, d1, ..., dn): Random values in a given shape.
np.random.rand(3,2)
#2. random.randn(d0, d1, ..., dn): Return a sample (or samples) from the “STANDARD NORMAL” distribution.
np.random.randn(3,2) # may include some negative numbers as well.
#3. random.randint: Return random integers from low (inclusive) to high (EXCLSUIVE).
np.random.randint(2, size=6) # [0, 0, 1, 1, 0, 1]
np.random.randint(2,8, size=10) # [4, 6, 3, 7, 6, 7, 7, 7, 2, 4]
np.random.randint(5,9, size=(2, 4)) #2D array created from 5-9 EXCLUSIVE integers
#4. .random_integers: Random integers of type np.int between low and high, INCLUSIVE.
np.random.random_integers(2,8, size=10) # [3, 7, 8, 6, 3, 4, 6, 5, 5, 3]
np.random.random_integers(5,10, size=(3,3)) #2D array inclusive element
np.random.random_integers(1, 6, 1000) # random integers b/w 1-6 , 1000 picked
#5. random.random_sample(size=None): Return random floats in the half-open interval [0.0, 1.0).
np.random.random_sample() # 0.9486623890742636
np.random.random_sample((5,)) # [0.88106937, 0.67473617, 0.29719984, 0.27329142, 0.97292897]
np.random.random_sample((4,2)) # 2D array of sample values
#6. .choice(a, size=None, replace=True, p=None): Generates a random sample from a given 1-D array
np.random.choice(5, 3) # Equivalent to np.random.randint(0,5,3) & may have DUPLICATE
np.random.choice(5, 3, replace=False) # will NOT have duplicates

## Permutations
#1. .shuffle(x): Modify a sequence in-place by shuffling its contents.
arr1d = np.arange(10) # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
np.random.shuffle(arr1d) # array is shuffled [7, 6, 4, 2, 9, 5, 3, 8, 0, 1]
arr2d = np.arange(9).reshape((3, 3))
np.random.shuffle(arr2d) # o/p is shuffled 2d array with same elements
#2. permutation(x): Randomly permute a sequence, or return a permuted range.
np.random.permutation(10) # [1, 7, 4, 6, 0, 3, 2, 8, 5, 9]
np.random.permutation([1, 4, 9, 12, 15]) # [12, 15,  9,  1,  4]
arr2d = np.arange(9).reshape((3, 3))
np.random.permutation(arr2d)

#### ********** Distributions of data *******##### beta, binomial, chisquare, exponential etc
# refer 1189 for details

### Random generator ###
#RandomState: Container for the Mersenne Twister pseudo-random number generator.
#seed([seed]): Seed the generator.
#get_state(): Return a tuple representing the internal state of the generator.
#set_state(state): Set the internal state of the generator from a tuple.
#1. seed(seed=None): Seed the generator.
#  intilization state of a pseudo random number generator. If you use the same seed you will get exactly the same pattern of numbers.

#### Set routines ###
## Making proper sets
#1. unique(ar, return_index=False, return_inverse=False, return_counts=False):Find the unique elements of an array.
np.unique([1, 1, 2, 2, 3, 3]) # 1,2,3
np.array([[1, 1], [2, 3]]) # 1,2,3
# returning the indices of the original array that give the unique values:
a = np.array([11, 22, 66, 44, 22, 33, 22])
u, indices = np.unique(a, return_inverse=True)
u # array([11, 22, 33, 44, 66]) ie unique set element
indices # [0, 1, 4, 3, 1, 2, 1] index position of uique elements
u[indices] # [11, 22, 66, 44, 22, 33, 22]

### Boolean operations ####
#1. in1d(ar1, ar2, assume_unique=False, invert=False): Test whether each element of a 1-D array is also present in a second array.
arr1 = np.array([0, 1, 2, 5, 0])
arr2 = [0, 2]
np.in1d(arr1,arr2) # [ True, False,  True, False,  True]
np.in1d(arr2,arr1) # [ True,  True]
#2. intersect1d(ar1, ar2, assume_unique=False): Find the intersection of two arrays.
np.intersect1d([1, 3, 4, 3], [3, 1, 2, 1]) # o/p is [1, 3]
np.intersect1d(arr1,arr2) # o/p is [0,2]
#3. setdiff1d(ar1, ar2, assume_unique=False): Find the set difference of two arrays.
a = np.array([1, 2, 3, 2, 4, 1])
b = np.array([3, 4, 5, 6])
np.setdiff1d(a, b) # [1,2]
#4. setxor1d(ar1, ar2, assume_unique=False): Find the set exclusive-or of two arrays.
np.setxor1d(a,b)
#5. union1d(ar1, ar2): Find the union of two arrays.
# Return the unique, sorted array of values that are in either of the two input arrays.
np.union1d([-1, 0, 1], [-2, 0, 2]) # [-2, -1,  0,  1,  2]
# union of more than two arrays, use functools.reduce
from functools import reduce
reduce(np.union1d, ([1, 3, 4, 3], [3, 1, 2, 1], [6, 3, 4, 2])) # o/p is [1, 2, 3, 4, 6]

### *** Sorting, searching, and counting ***  ######
#1. sort(a, axis=-1, kind=’quicksort’, order=None): Return a sorted copy of an array.
a = np.array([[1,4],[3,1]])
np.sort(a) # # sort along the last axis
np.sort(a, axis=None) # sort the flattened array
np.sort(a, axis=0) # sort along the first axis

b= np.array([4,7,1,99,11,345])
np.sort(b) # [  1,   4,   7,  11,  99, 345]
#2. partition(a, kth, axis=-1, kind=’introselect’, order=None): Return a partitioned copy of an array.
a = np.array([3, 4, 2, 1])
np.partition(a, 3)

### Searching
#1. argmax(a, axis=None, out=None): Returns the indices of the maximum values along an axis.
# Note: In case of multiple occurrences of the maximum values, the indices corresponding to the first occurrence are returned.
da = np.arange(6).reshape(2,3)
np.argmax(da) # 5 as the argument if max vakue from 2d array da
np.argmax(da, axis=0) # o/p is  array([1, 1, 1])
np.argmax(da, axis=1) # o/p is array([2, 2]) 
#2. nanargmax(a, axis=None): Return the indices of the maximum values in the specified axis ignoring NaNs
a = np.array([[np.nan, 4], [2, 3]])
np.argmax(a) # 0
a = np.array([[np.nan, 4], [2, 3]])
np.nanargmax(a) #1 , ignoring the nan index
np.nanargmax(a, axis=0) # array([1, 0])
np.nanargmax(a, axis=1) # array([1, 1])
#3. argmin(a, axis=None, out=None): Returns the indices of the minimum values along an axis.
#4. nanargmin(a, axis=None): Return the indices of the minimum values in the specified axis ignoring NaNs
#5. nonzero(a): Return the indices of the elements that are non-zero.
x = np.eye(3)
np.nonzero(x) # o/p is (array([0, 1, 2]), array([0, 1, 2]))
x[np.nonzero(x)] # o/p is array([1., 1., 1.])
#6. flatnonzero(a): Return indices that are non-zero in the flattened version of a
x = np.arange(-2, 3) # [-2, -1,  0,  1,  2]
np.flatnonzero(x) # index position as [0, 1, 3, 4]
#7. where(condition [ , x, y ] ): Return elements, either from x or y, depending on condition.
x = np.arange(9.).reshape(3, 3)
np.where( x > 5 ) # o/p is (array([2, 2, 2]), array([0, 1, 2]))
x[np.where( x > 3.0 )] # o/p is 1d array([4., 5., 6., 7., 8.])
np.where(x < 5, x, -1) # Note: broadcasting.
#8. searchsorted(a, v, side=’left’, sorter=None): Find indices where elements should be inserted to maintain order.
np.searchsorted([11,21,31,41,51], 3) # 3 is searched across 11-51 array elements
np.searchsorted([1,2,3,4,5], 3, side='left') # 2 ie starts from left towards '3'
np.searchsorted([1,2,3,4,5], 3, side='right') # 3 ie starts from right towards '3'
#9. extract(condition, arr): Return the elements of an array that satisfy some condition.
arr = np.arange(12).reshape((3, 4))
condition = np.mod(arr, 3)==0 # checks for array elemnt divided by 3  & which returns 0 as remainder
np.extract(condition, arr) #  array([0, 3, 6, 9])

### Counting
#count_nonzero(a): Counts the number of non-zero values in the array a.
daf= np.eye(4) # eye matric of 4by4
np.count_nonzero(daf) # 4 times of 1 as diagonal present
np.count_nonzero([[0,1,7,0,0],[3,0,0,2,19]]) # count is 5


##### Statistics ########
## Order statistics: 
#1. .amin(a, axis=None, out=None, keepdims=False): Return the minimum of an array or minimum along an axis.
a = np.arange(4).reshape((2,2))
np.amin(a) # min of flattened array . o/p is 0
np.amin(a, axis=0) # Minima along the first axis. o/p is [0, 1]
np.amin(a, axis=1) # Minima along the second axis. o/p is [0, 2]
np.amin(a, axis=2) # AxisError: axis 2 is out of bounds for array of dimension 2
#2. amax(a, axis=None, out=None, keepdims=False): Return the maximum of an array or maximum along an axis.
np.amax(a) # min of flattened array . o/p is 0
np.amax(a, axis=0) # Minima along the first axis. o/p is [0, 1]
np.amax(a, axis=1) # Minima along the second axis. o/p is [0, 2]
np.amax(a, axis=2) # AxisError: axis 2 is out of bounds for array of dimension 2
## Note: o is row axis, & 1 is column axis basically...
#3. nanmin(a, axis=None, out=None, keepdims=False): Return minimum of an array or minimum along an axis, ignoring any NaNs.
a = np.array([[4, 6], [2, np.nan]])
np.nanmin(a) # 2
np.nanmin(a, axis=0) # [2., 6.]
np.nanmin(a, axis=1) #[4., 2.]
# When positive infinity and negative infinity are present:
np.nanmin([1, 2, np.nan, np.inf]) #1.0
np.nanmin([1, 2, np.nan, np.NINF]) # -inf
#4. nanmax(a, axis=None, out=None, keepdims=False): Return the maximum of an array or maximum along an axis, ignoring any NaNs
np.nanmax(a) # 6
np.nanmax(a, axis=0) # [4., 6.]
np.nanmax(a, axis=1) #[6., 2.]
# When positive infinity and negative infinity are present:
np.nanmax([1, 2, np.nan, np.inf]) # inf
np.nanmax([1, 2, np.nan, np.NINF]) # 2
#5. ptp(a, axis=None, out=None): Range of values (maximum - minimum) along an axis.
# name comes from peak to peak 
x = np.arange(4).reshape((2,2))
np.ptp(x, axis=0) # o/pis [2, 2]
np.ptp(x, axis=1) # o/p is [1, 1]
#6. percentile: Compute the qth percentile of the data along the specified axis.
a = np.array([[10, 7, 4], [3, 2, 1]])
np.percentile(a, 50) #50th percentile is 3.5
np.percentile(a, 50, axis=0) # o/p is [6.5, 4.5, 2.5]
np.percentile(a, 50, axis=1) # o/p is [7., 2.]
np.percentile(a, 50, axis=1, keepdims=True) # o/p is array([[7.],[2.]])

### Averages and variances ###
#1. median: Compute the median along the specified axis.
a = np.array([[10, 7, 4], [3, 2, 11]])
np.median(a) # 5.5
np.median(a, axis=0) # [6.5, 4.5, 7.5]
np.median(a, axis=1) # [7., 3.]
#2. .Average: Compute the weighted average along the specified axis.
np.average(range(1,5)) # 2.5
np.average(range(1,11), weights=range(10,0,-1)) # 4.0

data = np.arange(6).reshape((3,2))
np.average(data, axis=1, weights=[1./4, 3./4]) # o/p is [0.75, 2.75, 4.75]
#3. .mean: Compute the arithmetic mean along the specified axis.
a = np.array([[1, 2], [3, 4]])
np.mean(a) # o/p is  2.5
np.mean(a, axis=0) # o/p is [2., 3.]
np.mean(a, axis=1) # o/p is [1.5, 3.5]
#4. ,std: Compute the standard deviation along the specified axis.
a = np.array([[1, 2], [3, 4]])
np.std(a) #  1.118033988749895
np.std(a, axis=0) # [1., 1.]
np.std(a, axis=1) # [0.5, 0.5]
#Note: Standard deviation in float64 is more accurate:
np.std(a, dtype=np.float64) # 1.118033988749895
#5. .var: Compute the variance along the specified axis.
a = np.array([[1, 2], [3, 4]])
np.var(a) # 1.25
np.var(a, axis=0) # [1., 1.]
np.var(a, axis=1) #[0.25, 0.25]
# Note: Computing the variance in float64 is more accurate:
np.var(a, dtype=np.float64) #   1.25
#6. .nanmedian: Compute the median along the specified axis, while ignoring NaNs.
a = np.array([[10.0, 7, np.nan], [3, 2, 1]])
np.median(a) # nan is o/p
np.nanmedian(a) # o/p is 3
np.nanmedian(a, axis=0) # [6.5, 4.5, 1. ]
np.median(a, axis=1) # [nan,  2.]
#7.nanmean: Compute the arithmetic mean along the specified axis, ignoring NaNs
np.nanmean(a) # 4.6
np.nanmean(a, axis=0) # [6.5, 4.5, 1. ]
np.nanmean(a, axis=1) # [8.5, 2. ]
# 8. nanstd: Compute the standard deviation along the specified axis, while ignoring NaNs.
np.nanstd(a) #  3.3823069050575527
np.nanstd(a, axis=0) # [3.5, 2.5, 0. ]
np.nanstd(a, axis=1) # [1.5, 0.81649658]
#9. nanvar: Compute the variance along the specified axis, while ignoring NaNs
np.var(a) # nan
np.nanvar(a, axis=0) # [12.25,  6.25,  0.  ]
np.nanvar(a, axis=1) # [2.25, 0.66666667]

### Correlating ### refer 1324 for further details
#corrcoef(x[, y, rowvar, bias, ddof]): Return Pearson product-moment correlation coefficients.
#correlate(a, v[, mode]): Cross-correlation of two 1-dimensional sequences.
#cov(m[, y, rowvar, bias, ddof, fweights, ...]): Estimate a covariance matrix, given data and weights.

### Histograms ######
#1. .histogram: Compute the histogram of a set of data.
np.histogram([1, 2, 1], bins=[0, 1, 2, 3]) # array([0, 2, 1]), array([0, 1, 2, 3])
np.histogram(np.arange(4), bins=np.arange(5), density=True) # array([0.25, 0.25, 0.25, 0.25]), array([0, 1, 2, 3, 4])
np.histogram([[1, 2, 1], [1, 0, 1]], bins=[0,1,2,3]) # (array([1, 4, 1]), array([0, 1, 2, 3])

a = np.arange(5)
hist, bin_edges = np.histogram(a, density=True) 
hist # array([0.5, 0. , 0.5, 0. , 0. , 0.5, 0. , 0.5, 0. , 0.5])
hist.sum() # 2.4999999999999996
np.sum(hist*np.diff(bin_edges)) #  1.0

#### Test Support (numpy.testing) ## refer 1337 

### Window functions ### 
#bartlett(M): Return the Bartlett window.
#blackman(M): Return the Blackman window.
#hamming(M): Return the Hamming window.
#hanning(M): Return the Hanning window.
#kaiser(M, beta): Return the Kaiser window

### *** NUMPY C-API REfer 1373 onwards for further readings.................... 












