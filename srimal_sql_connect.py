## Sample reading from SQL database-----------------------------------------------------------------------------
##Read from
import pyodbc 
cnxn = pyodbc.connect('Driver={SQL Server};'
                      'Server=.\SQLEXPRESS;'
                      'Database=Sample;'
                      'Trusted_Connection=yes;')

cursor = cnxn.cursor()
cursor.execute("SELECT * FROM Sample.dbo.Student")

for row in cursor.fetchall():
    print(row)
    
##Update THE TABLE -----------------------------------------------------------------------------------------------
import pyodbc 
cnxn = pyodbc.connect('Driver={SQL Server};'
                      'Server=.\SQLEXPRESS;'
                      'Database=Sample;'
                      'Trusted_Connection=yes;')

cursor = cnxn.cursor()
cursor.execute("update Sample.dbo.Student set F_Name='Ashu132432' where Id= '712'")
cnxn.commit() ##required if user is making chnages in db objects. Else query will remain in INFINITE LOOP.
# reading updates from target table
cursor.execute("SELECT * FROM Sample.dbo.tbl_Employee")

for row in cursor.fetchall():
    print(row)


