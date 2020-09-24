################# https://data-flair.training/blogs/python-sequence/ ##########################--
## SEQUENCE ##-----------------------------------------------------------------------
# A sequence is a group of items with a deterministic ordering. The order in which we put them in is the order in which we get an item out from them.
# Python offers six types of sequences:
	#1. Strings
	#2. Lists
	#3. Tuples
	#4. Bytes Sequences
	#5. Bytes Arrays
	#6. range() object
	
## Python Strings: 	
# A string is a group of characters. Since Python has no provision for arrays, we simply use strings. This is how we declare a string.
# We can use a pair of single or double quotes. And like we’ve always said, Python is dynamically-typed. Every string object is of the type ‘str’.
# also, its ummutable by nature
type(name) ## <class ‘str’>

# To declare an empty string, we may use the function str():
name=str()
name # o/p is ''

name=str('Amit Srimal')
name # o/p is 'Amit Srimal'

name=str(Amit Srimal)
name #o/p is SyntaxError: invalid syntax

# can't access using index
name[5] # o/p IndexError: string index out of range

## Python Lists:
#Since Python does not have arrays, it has lists. 
#A list is an ordered group of items. To declare it, we use SQUARE [] brackets
groceries=['milk','bread','eggs']
groceries[1] # o/p is  'bread'
groceries[:2] # o/p is ['milk', 'bread']

# A Python list can hold all kinds of items; this is what makes it heterogenous.
mylist=[1,'2',3.0,False]

# A list is mutable. This means we can change a value.
groceries[0]='cheese'
groceries # o/p is ['cheese', 'bread', 'eggs']

## Python Tuples:
#A tuple, in effect, is an immutable group of items. 
# When we say immutable, we mean we cannot change a single value once we declare it.
    # APPROACH 1:To declare it, we use regular () brackets
name=('Amit','Srimal')
name # o/p is  ('Amit', 'Srimal')
type(name) # o/p is tuple
    # APPROACH 2: We can also use the function tuple().
name=tuple(['Amit','Srimal'])
name
type(name) #o/p is  ('Amit', 'Srimal')

# a tuple is immutable. Let’s try changing a value.
name[0]='Avery' # o/p TypeError: 'tuple' object does not support item assignment

## Python Bytes Sequences:
# The function bytes() returns an immutable bytes object
bytes(5) #o/p is b'\x00\x00\x00\x00\x00'
bytes([1,2,3,4,5]) #o/p is b'\x01\x02\x03\x04\x05'
bytes('hello','utf-8') #o/p si  b'hello'

# Since it is immutable, if we try to change an item, it will raise a TypeError.
a=bytes([1,2,3,4,5])
a[4]=3 # o/p says TypeError: ‘bytes’ object does not support item assignment

##Python Bytes Arrays
# A bytesarray object is like a bytes object, but it is mutable. 
# It returns an array of the given byte size.
a=bytearray(4)
a #o/p is bytearray(b'\x00\x00\x00\x00')

# byte array is mutable
a=bytearray([1,2,3,4,5]) 
a # o/p is bytearray(b'\x01\x02\x03\x04\x05')
a[4]=3 #o/p is bytearray(b'\x01\x02\x03\x04\x03')

## Python range() objects
# A range() object lends us a range to iterate on; it gives us a list of numbers.
a=range(4)
a # o/p is range(0, 4)
type(a) #range

for i in range(7,2,-1):
     print(i) # o/p is 7,6,5,4,3 (iterative fashion here; last element not included)

for i in range(2,7,1):
     print(i) # o/p is 2,3,4,5,6
# if we use increment 0.5, we get TypeError: 'float' object cannot be interpreted as an integer      

## COLLECTIONS ##-----------------------------------------------------------------------
# Python collection, unlike a sequence, does not have a deterministic ordering. 
# In a collection, while ordering is arbitrary, physically, they do have an order.
# Every time we visit a set, we get its items in the same order. However, if we add or remove an item, it may affect the order.
# Python offers two types of collections:
	#1. Sets
	#2. Dictionaries
    
## Python Sets    
#A set, in Python, is like a mathematical set in Python. It does not hold duplicates. 
# We can declare a set in two ways:	    
    # APPROACH 1: We can use CURLY {} brackets.
nums={2,1,3,2} # o/p is {1, 2, 3}
type(nums) # set
    # APPROACH 2: using function set
nums=set([1,3,'2'])
nums # o/p is  {1, '2', 3}  
type(nums) # set

# A set is mutable.
nums.discard(3) # o/p is {1, '2'}
# Note: But it may not contain mutable items like lists, dictionaries, or other sets.

## Python Dictionaries  
# It holds key-value pairs, and this is how we declare it
# We can declare a set in two ways:	
    #APPROACH 1:  Use curly {} brackets
a={'name':1,'dob':2}
type(a) # o/p is dict
    #APPRAOCH 2: declare and fill
a= dict()
a['name']='Amit' # can use 1 in place of 'Amit'
a['surname']= 'Srimal'  # can use 2 in place of 'Srimal'  
a # o/p is {'name': 'Amit', 'surname': 'Srimal'} or {'name': 1, 'surname': 2}
    # APPROACH 3: Slight variation to Approach 2 
b= dict(f_number=1,s_number=2)
b # o/p is {'f_number': 1, 's_number': 2}
type(b) # o/p is dict

a={i:2**i for i in range(4)}
a # o/p is {0: 1, 1: 2, 2: 4, 3: 8}

# A key should be of hashable type.Think of a hash as a unique value for each input. All mutable values are unhashable
a={[1,2,3]:1,1:[1,2,3]} # o/p is TypeError: unhashable type: 'list'


#-----------------------------------------------------------------------------------------------------

## PYTHON SEQUENCE OPERATION
# 1 Concatenation :Concatenation adds the second operand after the first one.
'Srimal'+'Amit' # o/p is 'SrimalAmit'
# 2 Integer Multiplication: We can make a string print twice by multiplying it by 2
'ba'+'na'*2 # o/p is bananan
# 3 Membership: To check if a value is a member of a sequence, we use the ‘in’ operator.
'men' in 'Disappointment' # o/p is True
# 4 Python Slice: Sometimes, we only want a part of a sequence, and not all of it. We do it with the slicing operator.
'AmitSrimal'[1:4] # o/p is mit

## PYTHON SEQUENCE FUNCTION 
# 1 len(): It returns the length of the Python sequence.
len('Srimal') # o/p is 6
# 2 min() and max(): min() and max() return the lowest and highest values, respectively, in a Python sequence.
min('Amit') # o/p is A
max('Amit') # o/p is t

## PYTHON SEQUENCE METHODS
# 1 Python index():This method returns the index of the first occurrence of a value.
'banana'.index('n')# o/p is 2
# 2 Python count(): count() returns the number of occurrences of a value in a Python sequence.
'banana'.count('na') # o/p is 2
'banana'.count('a') # o/p is 3

################# https://data-flair.training/blogs/python-sequence/ ##########################--






















