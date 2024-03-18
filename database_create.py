import sqlite3

# Connect to the SQLite database (creates a new database if it doesn't exist)
conn = sqlite3.connect('example.db')

# Create a cursor object to execute SQL statements
cursor = conn.cursor()

# Create a table
cursor.execute('''CREATE TABLE IF NOT EXISTS attendance
                  (id INTEGER PRIMARY KEY AUTOINCREMENT,
                   student_id TEXT,
                   timestamp TEXT,
                   status TEXT)''')

# Insert some sample data
cursor.execute("INSERT INTO employees (name, age) VALUES (?, ?)", ('John Doe', 30))
cursor.execute("INSERT INTO employees (name, age) VALUES (?, ?)", ('Jane Smith', 28))
conn.commit()

# Perform a simple query
cursor.execute("SELECT * FROM employees")
rows = cursor.fetchall()

# Display the query results
for row in rows:
    print("ID:", row[0])
    print("Name:", row[1])
    print("Age:", row[2])
    print()

# Close the cursor and the connection
cursor.close()
conn.close()