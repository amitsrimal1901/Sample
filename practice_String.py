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
## From Geeks from Geeks
'''A string of required formatting can be achieved by different methods.
1) Using %: %d – integer,%f – float,%s – string,%x – hexadecimal,%o – octal
2) Using {}.format
3) Using Template Strings
4) Using f-F 
'''
# Using % method
num = 12.3456789
#For retrieving Integer part
print('The value of num is %3.1d' %num) # The value of num is 12
print('The value of num is %3.2d' %num) # The value of num is 12
#For retrieving Floating part
print('The value of num is %3.2f' %num) # The value of num is 12.35
print('The value of num is %3.4f' %num) #The value of num is 12.3457
#NOTE: Notation 3 in %3.2f refers to Python 3

# Strings in Python can be formatted with the use of format() method
# Format method in String contains curly braces {} as placeholders which can hold arguments according to position or keyword to specify the order.
f1 = "{} {}".format('Amit','Srimal')
f2 = "{1} {0}".format('Srimal','Amit')
f3 = "{f} {l}".format(l='Srimal', f='Amit')
print(f1, f2, f3) # Amit Srimal Amit Srimal Amit Srimal
f4 = "{}+{}".format('Amit','Srimal')
print(f4) ## Amit+Srimal

## f-F Strings for formatting (f may stand for FAST)
#The syntax is similar to the one you used with str.format() but less verbose
#f-strings are faster than both %-formatting and str.format(). f-strings are expressions evaluated at runtime rather than constant values.
## f string expects Expression inside {} always like {name}, {age}
name = "Eric"
age = 74
intro= f"Hello, {name}. You are {age}."
print(intro) ## Hello, Eric. You are 74.
INTRO= f"Hello, {name}. You are {age}."
print(INTRO) ## Hello, Eric. You are 74.

## # Formatting of Integers
f5 = "{0:b}".format(16) ## Binary representation of 16 is 10000
# Formatting of Floats
f6 = "{0:e}".format(165.6458)  # Exponent representation of 165.6458 is 1.656458e+02
# Rounding off Integers
f7 = "{0:.2f}".format(1/6) ##one-sixth is 0.17 ; upto two float decimals
f8= "{0:.3f}".format(3.134327429) ## 3.134
f9 = "{0:.3f}".format(33.134327429) ## 3.134
# Roudnign off concept
print(round(3.134327429, 4) ) ## 3.1343

#SPLIT STRING
"""
  str.split(regexp = "", limit = string.count(str))
    // regexp is the delimiting regular expression; (may be space, comma, dot etc. we may use) 
    // limit is limit the number of splits to be made 
"""
line = "Geek1 Geek2 Geek3";
print (line.split()) # ['Geek1', 'Geek2', 'Geek3']
print (line.split(' ', 1) ) #['Geek1', 'Geek2 Geek3']







