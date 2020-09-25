##Set is an UNORDERED(means index is not fixed) collection of data type that is iterable, mutable and has no duplicate elements.
# The order of elements in a set is undefined though it may consist of various elements.
#The major ADVANTAGE of using a set, as opposed to a list, is that it has a highly optimized method for checking whether a specific element is contained in the set.
"""IMP NOTE
A set is an unordered collection of items.
Every set element is unique (no duplicates) and must be immutable (cannot be changed).
However, a set itself is mutable. We can add or remove items from it.
"""

## Creating a set
#Method1: Using {}
set1= {1,2,'amit'}
print(set1) # {'amit', 1, 2} : --O/P is always ORDERED
print(type(set1)) # <class 'set'>
#Set is unordered and can't access through IDNEX
print(type(set1[1])) ## TypeError: 'set' object is not subscriptable.
#Method2: Using set() built in functon
set2= set([3,4,'Srimal']) # mixed data type set
print(set2) # {'Srimal', 3, 4}: --O/P is always ORDERED
print(type(set2)) # <class 'set'>

#Note –
# 1. A set cannot have mutable elements like a list, set or dictionary, as its elements.
# Try creating a set from ineteger and list value
set3= {56,77,[3,4,6]} # TypeError: unhashable type: 'list'
#2. Set should have immutable elements like String, integer
set4= set('AmitSrimal')
print(set4) #{'a', 'A', 't', 'l', 'r', 'S', 'm', 'i'}
print(type(set4)) # <class 'set'>
print(len(set4)) # 8
# Check teh difference here with {} and set fun
set5={'AmitSrimal'}
print(set5) #{'AmitSrimal'}
print(type(set5)) # <class 'set'>
print(len(set5)) # 1

## ADDING element to set add()
# Only ONE element can be added at a time; loops are used to add multiple elements at a time with the use of add() method.
#Lists cannot be added to a set as elements because Lists are not hashable whereas Tuples can be added because tuples are immutable and hence Hashable
##NOTE IMP:
# In Python, any immutable object (such as an integer, boolean, string, tuple) is hashable, meaning its value does not change during its lifetime.
# This allows Python to create a unique hash value to identify it, which can be used by dictionaries to track unique keys and sets to track unique values.
# Assuming set 5 is {'AmitSrimal'}
set5.add(3,99) #add() takes exactly one argument. For more elements use update()
set5.add(99.88)
print(set5) ## {99.88, 'AmitSrimal'}

set5.add((3,99)) # will be terated as single element
print(set5) # {(3, 99), 99.88, 'AmitSrimal'}

##Update(): For addition of two or more elements Update() method.
#The update() method accepts lists, strings, tuples as well as other sets as its arguments.
# In all of these cases, duplicate elements are avoided.
set5.update([3,6.7,'Amit',['test1',56]]) # TypeError: unhashable type: 'list'
set5.update([3,6.7,'Amit',('test1',56)]) # try with TUPLE as its IMMUTABLE'
print(set5) # {6.7, 3, ('test1', 56), (3, 99), 99.88, 'Amit', 'AmitSrimal'}

## ACCESSing
# Set items cannot be accessed by referring to an index, since sets are unordered the items has no index.
# But you can loop through the set items using a for loop, or ask if a specified value is present in a set, by using the in keyword.
print(3 in set5) # TRUE
print(4 in set5) # FALSE

##REMOVING/ DISCARD
#Elements can be removed from the Set by using built-in remove() function but a KeyError arises if element doesn’t exist in the set.
# To remove elements from a set without KeyError, use discard(), if the element doesn’t exist in the set, it remains unchanged."""
set5.remove(6.7) # 6.7 is removed from Set
set5.remove(88.88) # KeyError: 88.88
set5.discard(88.88) # no error show if even when elelemt is not present in set

##POP: reoves the last element in set
#NOTE:  If the set is unordered then there’s no such way to determine which element is popped by using the pop() function.
set5.pop() # 3 removed randomy assuming this was last element in set

##Clear
set6={45, 'ewweeew',343.88}
set6.clear()
print(set6) # set()

"""
FROZEN SETS in Python are Frozen set is just an immutable version of a Python set object.
While elements of a set can be modified at any time, elements of the frozen set remain the same after creation.
If no parameters are passed, it returns an empty frozenset.
Due to this, frozen sets can be used as keys in Dictionary or as elements of another set. 
But like sets, it is not ordered (the elements can be set at any index).
"""
fs = ('G', 'e', 'e', 'k', 's', 'F', 'o', 'r')
print(type(fs)) # <class 'tuple'>
FS = frozenset(fs) # create set from TUPLE (is immutable)
print(FS) # frozenset({'r', 'F', 'G', 'o', 's', 'k', 'e'})









