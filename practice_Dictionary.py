# Dictionary: Dictionary in Python is an unordered collection of data values, used to store data values.
# Dictionary holds key:value pair.
# Note –
# 1. Dictionary keys are case sensitive, same name but different cases of Key will be treated distinctly.
#2. Keys in a dictionary doesn’t allows Polymorphism.

## Ceate Dictionary

#METHOD1: using {}
dict1={'fname':'Amit','l_name':'Srimal','age':33}
print(dict1) # {'fname':'Amit','l_name':'Srimal','age':33}
print(type(dict1)) #<class 'dict'>
print(len(dict1)) # 3
# Dictinary with MIXED keys
dict2 = {'Name': 'Amit', 1: [1, 2, 3, 4]}
print(dict2) #{'Name': 'Amit', 1: [1, 2, 3, 4]}

#METHOD 2: Using dict() functin
dict3=dict() # create EMPTY dictionary
print(dict) # {}
print(type(dict3)) #<class 'dict'>

dict4= dict(['f_name':'amit','age':'30']) # [] is ordered by nature hence cant be used for dict creation
dict4= dict({'f_name':'amit','age':'30'}) # set are unordered liek dict & hence used here

#NESTED dict
dict5= dict({'f_name':'Amit','l_name':'Srimal','Subj':{'subj1':'Computer','subj2':'English'}})
print(dict5)
print(type(dict5)) # <class 'dict'>

# Adding elemenst to Dictionary
"""
One value at a time can be added to a Dictionary by defining value along with the key e.g. Dict[Key] = ‘Value’. 
Updating an existing value in a Dictionary can be done by using the built-in update() method. Nested key values can also be added to an existing Dictionary.
"""
dict5['age']=31
print(dict5) # {'f_name': 'Amit', 'l_name': 'Srimal', 'Subj': {'subj1': 'Computer', 'subj2': 'English'}, 'age': 31}
# Note- While adding a value, if the key value already exists, the value gets updated otherwise a new Key with the value is added to the Dictionary.
dict5['age']=33
print(dict5['age']) # 33, update happened from 31 to 33 as key was already present

# update uysing update() fuction with other dict
dict6={'age':'90','city':'Ahmedabad'} # ths is the dict used for update
dict5.update(dict6)
print(dict5) # age gets updated to 90 & City field is ADDED

## ACCESSING elements
#Method1: Using key
dict5[1] # KeyError: 1 as Dict is unorderd
dict5['city'] # 'Ahmedabad'
#accessing nested element
dict5['Subj']['subj1'] # Computer
#Method2: using Get() method
print(dict5.get(1)) # None as its unordered
print(dict5.get('city')) # Ahmedabad

## REMOVE elemenets from Dcict
# Method1: del()
# Note- del Dict will delete the entire dictionary and hence printing it after deletion will raise an Error.
del dict5['f_name'] # specify name of th ekey tobe deleted
print(dict5) #{'l_name': 'Srimal', 'Subj': {'subj1': 'Computer', 'subj2': 'English'}, 'age': '90', 'city': 'Ahmedabad'}
del dict5['l_name','age'] # KeyError: ('l_name', 'age': del can DLEETE ONE aat a time
#Deleing nested key:valie
del dict5['Subj']['subj1']
print(dict5) # {'l_name': 'Srimal', 'Subj': {'subj2': 'English'}, 'age': '90', 'city': 'Ahmedabad'}

del dict5 # deletes entire dictionary
print(dict5) #Error as dict is no longer avaiable

#MEthod 2:
# The popitem() returns and removes an arbitrary element (key, value) pair from the dictionary.
dict5.popitem() # ('city', 'Ahmedabad') popped out arbitarirly

#Method3: clear()
#All the items from a dictionary can be deleted at once
dict5.clear()
print(dict5) # {} Empty dict

## COPY Dictionary
dict_copyof4=dict1.copy()
print(dict_copyof4) # {'fname': 'Amit', 'l_name': 'Srimal', 'age': 33}
# This method doesn't modify the original dictionary just returns copy of the dictionary.
print(type(dict_copyof4)) # <class 'dict'>

## ites of Dioctionary
dict_copyof4.items() # dict_items([('fname', 'Amit'), ('l_name', 'Srimal'), ('age', 33)])

# return the string, denoting all the dictionary keys with their values
str(dict_copyof4) # "{'fname': 'Amit', 'l_name': 'Srimal', 'age': 33}"
type(str(dict_copyof4)) # <class 'str'>

### HANDLING OF MISSING KEYS in dict
# When no et is present in Dict,it popped a runtime error.
# To avoid such conditions, and to make aware user that a particular key is absent or to pop a default message in that place.

#MEthod1:get() method
#get(key,def_val) method is useful when we have to check for the key.
# If the key is present, value associated with the key is printed, else the def_value passed in arguments is returned.
print(dict1) # {'fname': 'Amit', 'l_name': 'Srimal', 'age': 33}
print(dict1.get('city')) # retruns None, istead shoudldl print say "Key Not Found""
print(dict1.get('city','Key Not Found')) # Key Not Found

#Method2: setdefault(key, def_value) works in a similar way as get(),
# but the difference is that each time a key is absent, a new key is created with the def_value associated to the key passed in arguments.
##Case1: Ket present
print(dict1.setdefault('age',55))
print(dict1) # {'fname': 'Amit', 'l_name': 'Srimal', 'age': 33}
##Case2: Key not present
print(dict1.setdefault('city','Bengaluru')) ##Bengaluru
print(dict1) # {'fname': 'Amit', 'l_name': 'Srimal', 'age': 33, 'city': 'Bengaluru'}

#Method3: Using defaultdict()
""""
“defaultdict” is a container that is defined in module named “collections“. It takes a function(default factory) as its argument. 
By default, default factory is set to “int” i.e 0. If a key is not present is defaultdict, the default factory value is returned and displayed. 
It has advantages over get() or setdefault().
1. A default value is set at the declaration & so no need to invoke the function again and again and pass similar value as arguments. Hence saving time.
2. The implementation of defaultdict is faster than get() or setdefault().
""""
#Step1:
import collections
#Step2: # sets default value 'Key Not found' to absent keys
defdict_dict1 = collections.defaultdict(lambda : 'Key Not found')
#Step3: Check with dict1 {'fname': 'Amit', 'l_name': 'Srimal', 'age': 33, 'city': 'Bengaluru'}
print(defdict_dict1['Contact']) # Key Not found
##IMP: setting default dict has impact on the original doctionary, see below
print(dict1) # {'fname': 'Amit', 'l_name': 'Srimal', 'age': 33, 'city': 'Bengaluru'}