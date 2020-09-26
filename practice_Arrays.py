"""
An array is a collection of items stored at contiguous memory locations. The idea is to store multiple items of the same type(HOMO-GENOUS) together.
This makes it easier to calculate the position of each element by simply adding an offset to a base value, i.e., the memory location of the first element of the array (generally denoted by the name of the array).
Think of an array a fleet of stairs where on each step is placed a value, of the same data type.
A user can treat lists as arrays. However, user cannot constraint the type of elements stored in a list.
If you create arrays using the array module, all elements of the array must be of the SAME TYPE.
Its MUTABLE, juyst like LIST
"""
import array as arr
# creating an array with integer type
a1 = arr.array('i', [1, 2, 3])
print(a1) # array('i', [1, 2, 3])
print(type(a1)) # <class 'array.array'>
print(len(a1)) # 3
# Alternativey import everything using *
from array import *
a2= array('f',[3,56.9,4])
# Printing a2
for i in range(len(a2)):
    print(a2[i]) # same values BUT each item in new line
    print(a2[i],end=" ") # 3.0 56.900001525878906 4.0

## ADDING elemets of array
#method1:Insert is used to insert one or more data elements into an array.
# Based on the requirement, a new element can be added at the beginning, end, or any given index of array.
# APPEND() is also used to add the value mentioned in its arguments at the END of the array.
a1.insert(3,56) # insert 56 at 3rd Index
print(a1) #array('i', [1, 2, 3, 56])
a1.insert(0,.1) # TypeError: integer argument expected, got float a sa 1 is INTEGER type Array
# append methid adds at the end of array
a1.append(8)
print(a1) #array('i', [1, 2, 3, 56, 8])

## ACCESSING elelemst of array
a1[2] # 3
a1[2:] # array('i', [3, 56,8])
a1[1:3] # array('i', [2, 3])
a1[::-2] # array('i', [8, 56, 2])
a1[-4::-2] # array('i', [3, 1]) ## :: used to get the REVERSED order
a1[:-2] # starts with 0th index & gioes till -2 index

## REMOVING element from array
#Method1: remove()
#  Remove() method only removes one element at a time, to remove range of elements, iterator is used.
# Note – Remove method in List will only remove the first occurrence of the searched element.
a1.remove(3) # removed 3 from array
a1.remove(8) # this has 8 twice, so only one 8 will be removed..array('i', [1, 2, 56, 8, 8])
print(a1) #array('i', [1, 2, 56, 8])
#Method2: pop()
#removes last element of array
a1.pop() # 8 is removed as this is the last eleemt of array
a1.pop(2) # 2nd index value 56 is removed
print(a1) # array('i', [1, 2])

## SEARCHING ELELEMNT is array
# index() method. This function returns the index of the first occurrence of value mentioned in arguments
a2 # array('f', [3.0, 56.900001525878906, 4.0])
a2.index(3.0) # retruns INDEX value of 3.0 as 0th
a2.index(4.0) # 2

## UPDATING elemenst of aaray
#simply reassign a new value to the desired index
a2[1]=3.4 #should replace 56.900 in 1st index
print(a2) # array('f', [3.0, 3.4000000953674316, 4.0])

##Sort
a2.sort() # AttributeError: 'array.array' object has no attribute 'sort'
#means no direct sorting method, Numopy has SORT function to handle this sorting issue.

""" Why to use Numpy Array when we have Python array
The need for NumPy arises when we are working with multi-dimensional arrays. 
The traditional array module does not support multi-dimensional arrays."""