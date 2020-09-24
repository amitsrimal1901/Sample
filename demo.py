#################### LIST ###### LIST ###### LIST ###### LIST ###### LIST ###### LIST ###### LIST ###### LIST ###### LIST
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
cars.sort(reverse=True, key=myFunc)
print(cars) ## ['Mitsubishi', 'Ford'', 'BMW', 'VW']
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

