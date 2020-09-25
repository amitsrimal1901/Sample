###### STRING###### STRING###### STRING###### STRING###### STRING###### STRING###### STRING###### STRING###### STRING
#A string is a sequence of characters. Strings are IMMUTABLE, i.e., they cannot be changed.
##Strings can be updated as WHOLE, but not in PARTIAL(hence called IMMUTABLE)
#Strings are arrays of bytes representing Unicode characters.
# #However, Python does not have a character data type, a single character is simply a string with a length of 1.
# #Square brackets can be used to access elements of the string.
a = "This is a string"
print (a)
type(a) # <class 'str'>
a[2] # 'i'

## String SLICING
a[2:8] # 'is is '
a[:8] # 'This is '
a[5:7] # 'is'

## Retrieve each element of String
for i in a:
    print(i, end="") # This is a string
    print(i) # each element appears in new line

## Use of Single, Double, Triple quotes
b1="It's there time now"
print(b1) ## It's there time now
b2= 'Sam"s home is here' # Sam"s home is here
print(b2)
b3='''Amit 
        Srimal'''
print(b3) # Amit Srimal in new line printed here.

## ESCAPE Characters
## \, \\, \t, \b, \n
txt1 = "We are the so-called \"Vikings\" from the north."
print(txt1) ## We are the so-called "Vikings" from the north.
txt2= 'This has there\'s belonging'
print(txt2) # This has there's belonging
txt3= "This will insert one \\ (backslash)."
print(txt3) # This will insert one \ (backslash).

##To ignore the escape sequences in a String, r or R is used, this implies that the string is a raw string and escape sequences inside it are to be ignored.
txt4 = "This is \x47\x65\x65\x6b\x73 in \x48\x45\x58"
print(txt4) ##This is Geeks in HEX : Special characters are recognized and interpraretd
txt5 = r"This is \x47\x65\x65\x6b\x73 in \x48\x45\x58"
print(txt5) ## Escape charaters IGNORED and we get This is \x47\x65\x65\x6b\x73 in \x48\x45\x58


##UPDATING/ DELETING Entire String (This is different from Immutability concept)
string1="this is sample string 1"
string1 = "this is update on sample string 1"
print(string1)
# Try deleting few character
del string1[3:5] # Immutable and TypeError: 'str' object does not support item deletion
del string1 ## Deletes the string as WHOLE

## FORMATTING
# Strings in Python can be formatted with the use of format() method
# Format method in String contains curly braces {} as placeholders which can hold arguments according to position or keyword to specify the order.
f1 = "{} {}".format('Amit','Srimal')
f2 = "{1} {0}".format('Srimal','Amit')
f3 = "{f} {l}".format(l='Srimal', f='Amit')
print(f1, f2, f3) # Amit Srimal Amit Srimal Amit Srimal
f4 = "{}+{}".format('Amit','Srimal')
print(f4) ## Amit+Srimal

## # Formatting of Integers
f5 = "{0:b}".format(16) ## Binary representation of 16 is 10000
# Formatting of Floats
f6 = "{0:e}".format(165.6458)  # Exponent representation of 165.6458 is 1.656458e+02
# Rounding off Integers
f7 = "{0:.2f}".format(1/6) ##one-sixth is 0.17 ; upto two float decimals
f8= "{0:.3f}".format(3.134327429) ## 3.134











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

