## LAMBDA/ Anonymous Functions
# anonymous function is a function that is defined without a name.
# While normal functions are defined using the def keyword, in Python anonymous functions are defined using the lambda keyword.
# syntax of lambda fuction:>> lambda arguments: expression 
# use lambda functions when we require a nameless function for a short period of time.
# we generally use it as an argument to a higher-order function (a function that takes in other functions as arguments). 
# Lambda functions are used along with built-in functions like filter(), map() etc.

#with regular function:
def f(x):
    return x*2
f(7) # 14 is o/p
#with lambda
g= lambda x: x * 2
g(8) # 16 is o/p 
# Note: lambda will be written in Single Line ONLY
# eg1. case combining firstname & last name
fullName = lambda fn,ln: fn.strip().title()+" "+ln.strip().title() # strip whitespace & title to have camelcase
fullName('  Srimal   ','Amit  ')
# eg2. writing quadratic equation
def qd(a,b,c):
    return lambda x: a*x**2+ b*x+c
f= qd(3,4,1) # passes the values of constant of qd equations
f(-2) # f passes-2 to qd ,which already has 3,4,1 & gets the values from qd
#eg3. usage with filters, maps
my_list = [1, 5, 4, 6, 8, 11, 3, 12]
new_list_filter = list(filter(lambda x: (x%2 == 0) , my_list))
new_list_map = list(map(lambda x: x * 2 , my_list))
#eg.4 getting text based on condition
def p(x):
    if x==2:
        return " its EVEN"
    else:
        return "its not EVEN"
#eg.5 finding volume    
import math    
def v(r):
    print ('is vlume of result', (4*math.pi*r**3)/3)
v(3)    
#eg. 6 printing multitext o/p response
def sq(num):
    ans= num**2
    print "sq of",num,'is', ans # multiple text print with Comma separator 
sq(5) # sq of 5 is 25
    
##### function to convert to Centimeters
def cms(feet=0,inches=0):
    "Converting to cms"
    inches_to_cms= inches*2.54
    feet_to_cms=feet*12*2.54
    return inches_to_cms + feet_to_cms
# calling function with variable values
cms(feet=3) # 91.44
cms(inches=1) # 2.54
cms(feet=3, inches=1) # 93.98, which is sum of above two results
# type of arguments: 1) Keyword & 2) Required
# Note: def g(y, x=0): Y is required & MUST come prior to Keyword
# def g(x=0,y): gives Syntax error    

#_______________________________________________________________________________
## UNIT TESTING: to verify code 
def circle_area(r):
    return math.pi*(r**2)
# test function to verify circle area:
radii = [3,6,-2,True, 5+9j, 'radii']
message = "Area of circle with radius {radius} is {area}."    
for r in radii:
    A= circle_area(r) # name of area function with attribute r
    print (message.format(radius=r, area=A))

#_______________________________________________________________________________
### DIFFERENCE betwen LOC,ILOC utilities
# loc is label-based, which means that you have to specify rows and columns based on their row and column labels.
# iloc is integer index based, so you have to specify rows and columns by their integer index
import pandas as pd
ss= pd.read_excel('/home/akmsrimal/Downloads/Sample - Superstore.xls', index_col=None)
ss.head()
#1. getting data with .iloc: NEEDS TO HAVE NUMBERING ROW/COLUMN
ss.iloc[2,[1,6,10]]
ss.loc[[1,6,10],]
ss.iloc[1,5] # basically row & column numbers
ss.iloc[:,5] # all rows & 5th columns
#2. getting data with .loc : NEED TO NAME THE INDEX ROW/ COLUMN
ss.iloc[:,'Segment'] # gives error as loc expects NAME
ss.loc[:,'Segment']
#2.1 Selections using the loc method are based on the index of the data frame (if any)
ss.set_index("Segment", inplace=True) # Setting Index is MUST
ss.loc['Consumer'] #consumer is a entry in index Segment
ss.loc[['Consumer','Corporate']]
ss.loc[['Consumer','Corporate'],['Index','Ship Mode','Segment','Customer Name']] # gets as above with column selection
#2.2 Boolean / Logical indexing using .loc, without using INDEX. Unindex previous SEGMENT
ss.loc[ss['Ship Mode'] == 'Standard Class']#,['Index','Ship Mode','Segment','Customer Name']]
ss.loc[ss['Ship Mode'] == 'Standard Class',['Index','Ship Mode','Segment','Customer Name']]
ss.loc[ss['Ship Mode'] == ['Standard Class','Second Class'],['Index','Ship Mode','Segment','Customer Name']]

## sorting
test = ['Banana','Pineapple','Apple','Carrot','Mango','Orange','Turnip']
test.sort()
test.sort(reverse=True)

#_______________________________________________________________________________
## A. Maps: 
#applies a function to ALL the items in an input_list.
# Most of the times we want to pass all the list elements to a function one-by-one and then collect the output.
#eg1:
items = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x**2, items))
#eg2:
def multiply(x):
    return (x*x)
def add(x):
    return (x+x)
funcs = [multiply,add]
for i in range(5):
    value = list(map(lambda x: x(i), funcs))
    print(value)
## B. Filter: 
#creates a list of elements for which a function returns true.
# The filter resembles a for loop but it is a builtin function and faster.
number_list = range(-5, 5)
less_than_zero = list(filter(lambda x: x < 0, number_list))
print(less_than_zero) # Output: [-5, -4, -3, -2, -1]
## C. Reduce: 
# Nolonger a built in function in 3+ python, functool is used rather
# performing some computation on a list and returning the result.
# It applies a rolling computation to sequential pairs of values in a list.
from functools import reduce
product = reduce((lambda x, y: x * y), [1, 2, 3, 4])
#_______________________________________________________________________________

## Operation requiring i/p from END USERS
# use of unput built in function in python
input("Enter a number greater than 1:") # syntax to i/p from user
number = input("Enter a number greater than 1:") # storing it in variable for further usages
type(number) # if string
number2 = int(input("Enter a number greater than 1:")) # storing it in variable for further usages
type(number2) # its int now coz of int as start of function behaviour

name= raw_input("Enter your name: ") # python 3 support input, for 2.7 we need raw_input
type(name) # str. Can use int(input...) for converting to integer value
age= int(input("Enter your age: "))
print 'Welcome' ,name,'your age is',age,'years'

# take user i/p & proess as per function definition & lastly returns o/p
age= int(input("Enter your age: ")) #1. Step 1 i/p numeric 
def cube_age(age): #2. Step2. function to perform certain operation
    return age**3
y = cube_age(age) #3. Step 3.assigning function o/p to some other variable
print 'Cube of your age',age,'is', y # step 4. getting overall o/p as her

#_______________________________________________________________________________
## Loops using multiple examples
#eg1.
for i in range(1,11):
   for j in range(1,6):
       k=i*j
   print k
#eg2.
sharks = ['hammerhead', 'great white', 'dogfish', 'frilled', 'bullhead', 'requiem']
for item in range(len(sharks)):
   sharks.append('shark')
print(sharks)
#eg3.
integers = []
for i in range(10):
   integers.append(i)
print(integers)
#eg4.
fname = 'Amit Srimal'
for letter in fname:
   print(letter) # Amit Srimal
   
## Nested For Loops
#eg1.
num_list = [1, 2, 3]
alpha_list = ['a', 'b', 'c']
for number in num_list:
    print(number)
    for letter in alpha_list:
        print(letter)
#eg2.
list_of_lists = [['hammerhead', 'great white', 'dogfish'],[0, 1, 2],[9.9, 8.8, 7.7]] # list of lists
for list in list_of_lists:
    for item in list:
        print(item) # this will print individual items in here

#_______________________________________________________________________________
# While loops: implements the repeated execution of code based on a given Boolean condition
number = [] # set of array to check 
number_of_guesses = 0 # initializing the guess count
while number_of_guesses < 5:
    print('Guess a number between 1 and 25:')
    guess = int(input())
    number_of_guesses = number_of_guesses + 1
    if guess == 13: # if guess is 13, then function breaks & does not execute any further
        print '13 received'
    if guess == 14: # if guess is 14 , then break is applied , irrespective of number of GUESSES.
        break
 




















   



