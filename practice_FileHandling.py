"""
EACH line of code includes a sequence of characters and they form text file.
Each line of a file is terminated with a special character, called the EOL or End of Line characters like comma {,} or newline character.
It ends the current line and tells the interpreter a new one has begun."""
# open () function in Python to open a file in read or write mode.
# open() function along with two arguments, that accepts file name and the mode, whether to read or write.
"""
There are three kinds of mode, that Python provides and how files can be opened:
“ r “, for reading.
“ w “, for writing.
“ a “, for appending.
“ r+ “, for both reading and writing
Note: If argument not passed,then Python will assume it to be “ r ” by default.
"""

## OPENING a file
#Method1: using iteration
file1 = open('C:/Users/amit_srimal/Desktop/Study/Python/Sample.txt', 'r')
# This will print every line one by one in the file
for i in file1:
        print (i)

#Method2: using read method to REad all file data
file2 = open('C:/Users/amit_srimal/Desktop/Study/Python/Sample.txt', 'r')
print(file2.read()) # retruns the text file data

#Method3: using read method to REad few characters of file data
file3 = open('C:/Users/amit_srimal/Desktop/Study/Python/Sample.txt', 'r')
print(file3.read(3)) # retruns FIRST 3 characters "Fil" as string
print(type(file3.read(3))) # <class 'str'>

## CREATING a file
file4 = open('C:/Users/amit_srimal/Desktop/Study/Python/Sample.txt', 'w')
file4.write("This is new line added 1")
file4.write("This is new line added 2")
file4.close() ## terminates all the resources in use and frees the system of this particular program.
# Its quite helpful because using this method any files opened will be closed automatically after one is done, so auto-cleanup.
print(file4.read())
##IMP:
# If file already exists, then All these sets of commands updates existing data
# If file doesn't exists, then All these sets of commands creates new file with data

##APPENDING data to a file
file5 = open('C:/Users/amit_srimal/Desktop/Study/Python/Sample.txt', 'a')
file5.write("This is new line added 1\n") # \n for new line creation
file5.write("This is new line added 2")
print(file5.read())

######### All read() can be done by using with() function
# Read operation
with open('C:/Users/amit_srimal/Desktop/Study/Python/Sample.txt') as file_to_read:
    print(file_to_read.read())
# Write operation
with open('C:/Users/amit_srimal/Desktop/Study/Python/Sample.txt',"w")as file_to_write:
    data=file_to_write.write("Written using with() method")
    print(data) # returns count of characters in file
##Appemding data
with open('C:/Users/amit_srimal/Desktop/Study/Python/Sample.txt',"a")as file_to_append:
    data=file_to_append.write("\nAppended using with() method")
    print(data) # returns count of characters in file


#********************************************************************************************************
# Workign owth CSV files
# For READING CSV files in python, there is an inbuilt module called csv.
##METHOD1: Using csv.reader() class
#step1: import the library
import csv
#step2: use with() to fetch csv file in desired mode
with open('C:/Users/amit_srimal/Desktop/Study/Python/csvsample.csv',mode ='r') as file:
    read_csv=csv.reader(file) # used to read the Giants.csv file which maps the data into lists.
#step3: start reading  the file line by line
    for i in read_csv:
        print(i) # prints entire list
        print(type(i))  # <class 'list'>
        print(i[0]) # print 0th index
        print(type(i[0])) # <class 'str'>
#NOTE: Indentation of all three steps are very important
#Extracting field name and count of records on csvfile
    csv_fields = next(read_csv)
    print(csv_fields) # ['f_name', 'l_name']##works only for first iteration n then jump to next line
    print('Field names are:' + ', '.join(field for field in csv_fields)) # Field names are:f_name, l_name
    print(read_csv.line_num) #4 including header

#METHOD2: Using csv.DictReader() class
# THis class maps the information in the CSV file into a dictionary.
#  The very first line of the file comprises of dictionary keys.
import csv
with open('C:/Users/amit_srimal/Desktop/Study/Python/csvsample.csv',mode ='r') as file:
    csv_read=csv.DictReader(file)
    for i in file:
        print(i)

# METHOD3: Using Pandas
import pandas as pd
data_file=pd.read_csv('C:/Users/amit_srimal/Desktop/Study/Python/csvsample.csv')
print(data_file)
# reads the Giants.csv file and maps its data into a 2D list.

### FOR WRITING TO CSV
import csv
# feild name
fields=['roll', 'subj','score']
# data
data=[[11,'computer',89],[23,'science',99]]
# define filename
file_name='C:/Users/amit_srimal/Desktop/Study/Python/scorecard.csv'

#write to taregt csv file now with data, field
with open(file_name,'w') as csv_file_to_read:
    # creating a csv writer object
    csvwriter = csv.writer(csv_file_to_read)
    # writing the fields into objects
    csvwriter.writerow(fields)
    # writing the data rows into objects
    csvwriter.writerows(data)
