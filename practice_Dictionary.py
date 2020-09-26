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











