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