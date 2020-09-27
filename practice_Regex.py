'''
Regular Expressions(RE) specifies a set of strings(pattern) that matches it.
A total of 14 metacharacters are used in REGEX expression
\   Used to drop the special meaning of character
[]  Represent a character class
^   Matches the beginning
$   Matches the end
.   Matches any character except newline
?   Matches zero or one occurrence.
|   Means OR (Matches with any of the characters separated by it.
*   Any number of occurrences (including 0 occurrences)
+   One or more occurrences
{}  Indicate number of occurrences of a preceding RE to match.
()  Enclose a group of REs
'''

"""
OVERALL, we have SEARCH(), MATCH() AND FINDALL() built -in function in module re.
1. re.match() method finds match if it occurs at start of the string.
2. re.search() method is similar to re.match() but it doesn’t limit us to find matches at the beginning of the string only. 
3. re.findall() helps to get a list of all matching patterns. It searches from start or end of the given string. 
If we use method findall to search for a pattern in a given string it will return all occurrences of the pattern.
"""

## How RE works?
# Regular expressions are compiled into PATTERN OBJECTS, which have methods for various operations such as searching for pattern matches or performing string substitutions.

### FINDALL() method ###*****************************************************************************************
##### SAMPLE 1: Chatacters
# Module Regular Expression is imported using __import__().
import re
# compile() creates regular expression character class [a-e],
# which is equivalent to [abcde].
# class [abcde] will match with string with 'a', 'b', 'c', 'd', 'e'.
p = re.compile('[a-e]')
# findall() searches for the Regular Expression and return a list upon finding
print(p.findall("Aye, said Mr. Gibenson Stark")) # ['e', 'a', 'd', 'b', 'e', 'a']
## NOTE: Case Sensitive expression mtch performed.

'''
\d   Matches any decimal digit, this is equivalent to the set class [0-9].
\D   Matches any non-digit character.
\s   Matches any whitespace character.
\S   Matches any non-whitespace character
\w   Matches any alphanumeric character, this is
     equivalent to the class [a-zA-Z0-9_].
\W   Matches any non-alphanumeric character.
Set class [\s,.] will match any whitespace character, ‘,’, or,’.’ .
'''
##### SAMPLE 2:Numerics
import re
# \d is equivalent to [0-9].
p = re.compile('\d')
print(p.findall("I went to him at 11 A.M. on 4th July 1886")) # ['1', '1', '4', '1', '8', '8', '6']
# \d+ will match a group on [0-9], group of one or greater size
p = re.compile('\d+')
print(p.findall("I went to him at 11 A.M. on 4th July 1886")) # ['11', '4', '1886']

##### SAMPLE 3: Alphanumeric
# \w is equivalent to [a-zA-Z0-9_]
import re
p1=re.compile('\w')
print(p1.findall("We met at 9:00PM")) # ['W', 'e', 'm', 'e', 't', 'a', 't', '9', '0', '0', 'P', 'M']
#Alternatively we can write the custpom pattern object
p2= re.compile('[b-c0-7]')
print(p2.findall("ball eat 677 ceat 900")) #['b', '6', '7', '7', 'c', '0', '0']

# \W matches to non alphanumeric characters.
p1 = re.compile('\W')
print(p1.findall("he said *** in some_language.")) #[' ', ' ', '*', '*', '*', ' ', ' ', '.']
#Alternatively we can write the custpom pattern object
p2 = re.compile('[*$%]')
print(p2.findall("he said *** in some_language in $12.")) #['*', '*', '*', '$']

##### SAMPLE4" import re
# '*' replaces the no. of occurrence of a character.
p = re.compile('ab*')
print(p.findall("ababbaabbb")) # ['ab', 'abb', 'a', 'abbb']
'''Understanding the Output:
Our RE is ab*, which ‘a’ accompanied by any no. of ‘b’s, starting from 0.
Output ‘ab’, is valid because of singe ‘a’ accompanied by single ‘b’.
Output ‘abb’, is valid because of singe ‘a’ accompanied by 2 ‘b’.
Output ‘a’, is valid because of singe ‘a’ accompanied by 0 ‘b’.
Output ‘abbb’, is valid because of singe ‘a’ accompanied by 3 ‘b'''


## SPLITTING STRING
# string by the occurrences of a character or a pattern, upon finding that pattern, the remaining characters from the string are returned as part of the resulting list.
'''Syntax: re.split(pattern, string, maxsplit=0, flags=0)
1. pattern denotes the regular expression, 
2. string is the given string in which pattern will be searched for and in which splitting occurs, 
3. maxsplit if not provided is considered to be zero ‘0’, and if any nonzero value is provided, then at most that many splits occurs. 
If maxsplit = 1, then the string will split once only, resulting in a list of length 2. 
4. The flags are very useful and can help to shorten code, they are not necessary parameters, eg: flags = re.IGNORECASE, In this split, case will be ignored.'''
## SAMPLE 1
from re import split
a=split('[$]', 'I found $4 but dont have $34')
print(a) #['I found ', '4 but dont have ', '34']
print(a[1]) # 4 but dont have
print(type(a)) # <class 'list'>
print(split('[$]', 'I found $4 but dont have $34')) #['I found ', '4 but dont have ', '34']

## SAMPLE 2
# # Splitting will occurs only once, say at '12', returned list will have length 2
b=split('[12]',"At 12:00PM, we have 12 players",1)
print(b)
print(b[0]) # At
print(b[1]) #2:00PM, we have 12 players

##SAMPLE 3
# 'Boy' and 'boy' will be treated same when flags = re.IGNORECASE
c=re.split('[b]', 'Aey, Boy oh boy, come here', flags = re.IGNORECASE)
print(c) #['Aey, ', 'oy oh ', 'oy, come here']
print(c[0]) # Aey,
print(c[1]) # oy oh
print(c[2]) # oy, come here
len(c) # 3

### SUBSTRIMG functon
# The ‘sub’ in the function stands for SubString, a certain regular expression pattern is searched in the given string(3rd parameter),
# and upon finding the substring pattern is replaced by repl(2nd parameter), count checks and maintains the number of times this occurs.
#  Syntax: re.sub(pattern, repl, string, count=0, flags=0)
# SAMPEL1:
import re
z1= re.sub('ub', '~*', 'Subject has Uber booked already', flags=re.IGNORECASE)
print(z1) # S~*ject has ~*er booked already
print(type(z1)) # <class 'str'>

# SAMPLE 2:
# As count has been given value 1, the maximum times replacement occurs is 1
print(re.sub('ub', '~*', 'Subject has Uber booked already', count=1, flags=re.IGNORECASE))
# S~*ject has Uber booked already

# Sample 3:
# 'r' before the patter denotes RE, \s is for start and end of a String.
print(re.sub(r'\sAND\s', ' & ', 'Baked Beans And Spam', flags=re.IGNORECASE))
#Baked Beans & Spam

###*****************************************************************************************
# Methods Search, Match
# ## SEEARCH:
# re.search() : This method either returns None (if the pattern doesn’t match), or a re.MatchObject that contains information about the matching part of the string.
# This method stops after the first match, so this is best suited for testing a regular expression more than extracting data.

#step1:
import re
# Step2: defne the search parametr object
regex= r"([a-zA-Z]+) ([d+])" ## to match a date string - in the form of Month name followed by day number
#step3: Settimg pattern object
match = re.search(regex, "I was born on June 24")
#step4: look for regex in target object
if match !=None:
    print("Match at index %s, %s" % (match.start(), match.end()))
    print("Full match: %s" % (match.group(0)))
    print("Full match: %s" % (match.group(1)))
    print("Full match: %s" % (match.group(2)))
else:
    print("Regex didn't matched")
'''
We us group() method to get all the matches and captured groups. The groups contain the matched values.  In particular: 
1. match.group(0) always returns the fully matched string 
2. match.group(1) match.group(2), ... return the capture groups in order from left to right in the input string 
3. match.group() is equivalent to match.group(0) 
'''

## MATCH() fucntion
#  This function attempts to match pattern to whole string. The re.match function returns a match object on success, None on failure.
'''
Syntax: re.match(pattern, string, flags=0)
# specify multiple flags using bitwise OR (|).
#pattern : Regular expression to be matched.
##string : String where p attern is searched
#flags : We can specify different flags using bitwise OR (|).'''



