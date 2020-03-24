import sqlite3
import csv
conn = sqlite3.connect('C:\\sqlite3\\test.db')
c=conn.cursor()
print(conn)
if(conn==1):
	print("Opened database successfully")

#conn.execute('''CREATE TABLE people (
#   id INTEGER PRIMARY KEY AUTOINCREMENT,
#   condition text NOT NULL,
#   drugName text NOT NULL,
#   Rating INTEGER NOT NULL,
#   UsefulCount INTEGER NOT NULL
#);''')

print("Table created")
#conn.execute(''' DROP TABLE MEDICINE ;''')
logitdisease='Cough'
x=c.execute("select drugName from Medicine where condition=?;",([logitdisease]))
for row in c.fetchone():
	print(row)