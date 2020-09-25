#TUPLES is a collection of Python objects much like a list but tuples are IMMUTABLE
# The sequence of values stored in a tuple can be of any type, and they are indexed by integers.
#Values of a tuple are syntactically separated by ‘commas’.
# Although it is not necessary, it is more common to define a tuple by closing the sequence of values in parentheses.
##Note – Creation of Python tuple without the use of parentheses is known as Tuple Packing.
tup1= 3,7,90,'test' ## TUPEL PACKING
print(type(tup1)) #<class 'tuple'>
tup2= (3,7,90,'test2')
print(type(tup2)) #<class 'tuple'>

# Blank/ empty tuple
blnk_tuple = ()
print(type(blnk_tuple)) # <class 'tuple'>

## Create TUPLE using TUPLE function
# Tuple from LIST
tuple_from_list= tuple([1,45,'Amit',[78,'Srimal']])
print(len(tuple_from_list)) # 4
print(len(tuple_from_list[3])) # 2
print(tuple_from_list) # (1, 45, 'Amit', [78, 'Srimal'])
print(type(tuple_from_list)) # <class 'tuple'>
# Tuple from String
tuple_from_string= tuple('Amit Srimal')
print(tuple_from_string) # ('A', 'm', 'i', 't', ' ', 'S', 'r', 'i', 'm', 'a', 'l')
print(type(tuple_from_string)) # <class 'tuple'>

## Nested tuple
tp1=(2,'apple')
tp2=('bike',56,'test')
nested_tup=(tp1, tp2)
print(nested_tup) # ((2, 'apple'), ('bike', 56, 'test'))
#Tuple with Repetition
repeated_tuple= ('Amit','Srimal',798)*3
print(repeated_tuple) #('Amit', 'Srimal', 798, 'Amit', 'Srimal', 798, 'Amit', 'Srimal', 798)

## Accessing TUPLE
# Via Index
nested_tup[1] # ('bike', 56, 'test')
# Via UNPACKING
a,b,c=nested_tup[1] # sets a,b,c to Bike, 56 & test
print(a,b,c) # bike 56 test

## Concatenation
# Concatenation is done by the use of ‘+’ operator.
# Concatenation of tuples is done always from the end of the original tuple.
# Other arithmetic operations do not apply on Tuples.
print(tp1+tp2) # (2, 'apple', 'bike', 56, 'test')

## SLICING
Tuple1 = tuple('GEEKSFORGEEKS') # ths is tuple from String
print(Tuple1[3]) #K
print(Tuple1[3:]) #('K', 'S', 'F', 'O', 'R', 'G', 'E', 'E', 'K', 'S')
print(type(Tuple1[3:])) #<class 'tuple'>
print(Tuple1[:5]) # ('G', 'E', 'E', 'K', 'S')
print(Tuple1[2:5]) # ('E', 'K', 'S')
#Reverse access [SHOULD BE READ FROM RIGHT to LEFT]
print(Tuple1[-2]) # K
print(type(Tuple1[-2])) ## <class 'str'>
print(Tuple1[::-1]) # ('S', 'K', 'E', 'E', 'G', 'R', 'O', 'F', 'S', 'K', 'E', 'E', 'G')
print(Tuple1[:-1]) # ('G', 'E', 'E', 'K', 'S', 'F', 'O', 'R', 'G', 'E', 'E', 'K')
print(Tuple1[-7:-2]) # ('O', 'R', 'G', 'E', 'E')
print(Tuple1[-2:-3]) # () ie Empty Tuple

## DEleting
# Tuples are immutable and hence they do not allow deletion of a part of it.
# Entire tuple gets deleted by the use of del() method.
tup_del=('test','element',56)
del tup_del
print(tup_del) # name 'tup_del' is not defined

# Sum, min, max, sort works for Integer. float TUPLES
list_to_sort=tuple([45,23,67,101])
print(sorted(list_to_sort)) #[23, 45, 67, 101]



