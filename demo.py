#################### LIST ###### LIST ###### LIST ###### LIST ###### LIST ###### LIST ###### LIST ###### LIST ###### LIST
my_list=['Amit',45,'$',56.09,['Srimal',45.7]] ## nested list
print(my_list) ##['Amit',45,'$',56.09,['Srimal',45.7]]
##Type info of list item wise
print(type(my_list))## <class 'list'>
print(type(my_list[2]))##<class 'str'>
print(type(my_list[4]))##<class 'list'>
print(type(my_list[4][1]))##<class 'float'>

## update any index value
my_list[1]=46
print(my_list) ## ['Amit', 46, '$', 56.09, ['Srimal', 45.7]]
##add new items to list
my_list.append('Kumar')
print(my_list) # ['Amit', 46, '$', 56.09, ['Srimal', 45.7], 'Kumar']
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
