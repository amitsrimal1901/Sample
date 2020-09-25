
#################### LIST ###### LIST ###### LIST ###### LIST ###### LIST ###### LIST ###### LIST ###### LIST ###### LIST
#  A single list may contain DataTypes like Integers, Strings, as well as Objects.
#  Lists are mutable, and hence, they can be altered even after their creation.
blnk_list=[] # a blank list
print(type(blnk_list)) # <class 'list'>
# to add value, we may use insert/ extend/ append methods

my_list=['Amit',45,'$',56.09,['Srimal',45.7]] ## nested list
print(my_list) ##['Amit',45,'$',56.09,['Srimal',45.7]]
##Type info of list item wise
print(type(my_list))## <class 'list'>
print(type(my_list[2]))##<class 'str'>
print(type(my_list[4]))##<class 'list'>
print(type(my_list[4][1]))##<class 'float'>

##create list using list, sets etx
f_name=['Amit','Annu','Cocoon']
age=[34,30,2]
gender={'M','F','F'} ## Note: Set removes duplicate so single F will be displayed
family=[f_name,age,gender]
print(family) ## [['Amit', 'Annu', 'Cocoon'], [34, 30, 2], {'M', 'F'}]--  ## Note: Set removes duplicate so single F will be displayed
print(f_name.append(age)) ## None
print(f_name) ## ['Amit', 'Annu', 'Cocoon', [34, 30, 2]]
print(f_name.extend([34,30,2])) ## None
print(f_name)  ##['Amit', 'Annu', 'Cocoon',34, 30, 2]

## update any index value
my_list[1]=46
print(my_list) ## ['Amit', 46, '$', 56.09, ['Srimal', 45.7]]
#Size of list, typically means LENGTH
print(len(my_list)) # 6 , consider list inside list as 1 element; if if that consist of many items: ['Amit', 46, [34, 6.7, 'Cycle'], 'RoyalEnfield', 56.09, ['Srimal', 45.7]]

"""
ADDING Elements to a List:
Using append() method
Using insert() method
Using extend() method
"""
##add new items to list
my_list.append('Kumar')
print(my_list) # ['Amit', 46, '$', 56.09, ['Srimal', 45.7], 'Kumar']
#add item to index base
my_list.insert(2,'RoyalEnfield')
print(my_list) ##['Amit', 46, 'RoyalEnfield', 56.09, ['Srimal', 45.7]]
my_list.insert(2,[34,6.7,'Cycle'])
print(my_list) ##['Amit', 46, [34, 6.7, 'Cycle'], 'RoyalEnfield', 56.09, ['Srimal', 45.7]]

###remove using item Value from list
my_list.remove('Kumar')
print(my_list) ## ['Amit', 46, '$', 56.09, ['Srimal', 45.7]]
## remove using INDEX number
my_list.pop(2)
print(my_list) # ['Amit', 46, 56.09, ['Srimal', 45.7]]

## Access range of value
 ## Assumimg parent contains ['Amit', 46, 56.09, ['Srimal', 45.7]]
print(my_list[1:3]) ## [46, 56.09] -- IMP: -- Last index to be considered -1
print(my_list[3][0:1]) ## ['Srimal']
print(my_list[3][0:2]) ## ['Srimal', 45.7]  -- IMP: -- Last index to be considered -1
print(my_list[2:]) # [56.09, ['Srimal', 45.7]]
print(my_list[:2]) # ['Amit', 46] -- IMP: -- Last index to be considered -1
print(my_list[:3]) # ['Amit', 46, 56.09]  -- IMP: -- Last index to be considered -1
print(my_list[-1]) # ['Srimal', 45.7]
print(my_list[-1][-2]) # ['Srimal']
print(my_list[-1:-3]) # [] ---Not supported & returns EMPTY list

## min, max, sum, sort functions
min(my_list) ## not supported between instances of 'int' and 'str' as the LIST os HETREGENOUS
new_list=[23,65,1,34,980,3.67] # <class 'list'>
print(min(new_list)) # 1
print(max(new_list)) # 980
print(sum(new_list)) # 1106.67
## #Sorting funciton
new_list.sort()
print(new_list) # [1, 3.67, 23, 34, 65, 980]
new_list.sort(reverse=True)
print(new_list) # [980, 65, 34, 23, 3.67, 1]
-----------------------------------------------------
# A sorting function based on length of the value:
def myFunc(e):
  return len(e)

cars = ['Ford', 'Mitsubishi', 'BMW', 'VW']
print(len(cars[1])) # 10 for Mitsubishi
cars.sort(reverse=True, key=myFunc)
print(cars) ## ['Mitsubishi', 'Ford'', 'BMW', 'VW']
## without using key
cars.sort(reverse=True)
print(cars) ## ['Mitsubishi', 'Ford'', 'BMW', 'VW'] ALPHABETICAL ORDER
------------------------------------------------------

## APPEND
stack = ['a','b']
stack.append('c') # append one item
print(stack) ## ['a', 'b', 'c']
stack.append(['e','f']) # append LIST of item
print(stack) ## ['a', 'b', 'c', ['e', 'f']]
##EXTEND
stack = ['a','b']
stack.extend(['c']) #extends one item
print(stack) ## ['a', 'b', 'c']
stack.extend(['e','f']) # extends List
print(stack) ## ['a', 'b', 'c', 'e', 'f']

## CLEAR list
list2= [1,2,4,5,56]
list2.clear()
print(list2) ## create a blank List as []
## INDEX()
# It's an inbuilt function in Python, which searches for a given element from the start of the list and returns the lowest index where the element appears.
list3= [56, 564,232,1,2,4,5,564,4] # lets search for 4
print(list3.index(4)) # returns 5 as index.
## COUNT: 	Returns the count of number of items passed as an argument
list3.count(4) # 2 as 4 is repeated twice in list3
##REVERSE(): Reverse the order of items in the list
list3.reverse()
print(list3) # Reversed order : [4, 564, 5, 4, 2, 1, 232, 564, 56]

## COPY: Returns a copy of the list
list3_copy=list3
print(list3_copy) # [4, 564, 5, 4, 2, 1, 232, 564, 56] COPY of Original list is returned

## LIST COMPARSION funciton cmp()
# This function returns 1, if first list is “greater” than second list, -1 if first list is smaller than the second list else it returns 0 if both the lists are equal.
#NOTE: Got decapricated for Python 3, only works with Python 2