import sqlite3

def create_tables():
    # Connect to the SQLite database
    conn = sqlite3.connect('attendance.db')

    # Create a cursor object
    cursor = conn.cursor()

    # Create the student table
    cursor.execute('''CREATE TABLE IF NOT EXISTS students
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id TEXT UNIQUE,
                    name TEXT,
                    date_of_birth TEXT,
                    class TEXT,
                    image BLOB)''')

    # Create the course table
    cursor.execute('''CREATE TABLE IF NOT EXISTS courses
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                    course_id TEXT,
                    course_name TEXT,
                    day_of_week TEXT,
                    periods TEXT,
                    location TEXT)''')

    # Create the attendance table
    cursor.execute('''CREATE TABLE IF NOT EXISTS attendance
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id TEXT,
                    course_id TEXT,
                    timestamp TEXT,
                    status TEXT,
                    FOREIGN KEY (student_id) REFERENCES students(student_id),
                    FOREIGN KEY (course_id) REFERENCES courses(id))''')

    # Commit the changes
    conn.commit()

    # Close the cursor and the connection
    cursor.close()
    conn.close()

def load_sample_data():
    # Connect to the SQLite database
    conn = sqlite3.connect('attendance.db')

    # Create a cursor object
    cursor = conn.cursor()

    # Insert sample student data
    student_data = [
        ('S001', 'Biden', '1998-05-15', 'Class A', 'images/biden.jpg'),
        ('S002', 'Nam', '1999-03-20', 'Class B', 'images/nam.jpg'),
        ('S003', 'Barack Obama', '2000-01-10', 'Class C', 'images/obama.jpg')
    ]

    for data in student_data:
        student_id, name, dob, class_name, image_path = data
        with open(image_path, 'rb') as f:
            image_data = f.read()
        cursor.execute('INSERT INTO students (student_id, name, date_of_birth, class, image) VALUES (?, ?, ?, ?, ?)',
                       (student_id, name, dob, class_name, image_data))

    
    # Insert sample course data
    course_data = [
        ('C001', 'Mathematics', 'Monday', '1-2', 'Room 101'),
        ('C002', 'Science', 'Tuesday', '3-4', 'Room 202'),
        ('C003', 'English', 'Wednesday', '5-6', 'Room 303')
    ]

    for data in course_data:
        course_id, course_name, day_of_week, periods, location = data
        cursor.execute('INSERT INTO courses (course_id, course_name, day_of_week, periods, location) VALUES (?, ?, ?, ?, ?)',
                       (course_id, course_name, day_of_week, periods, location))

    # Commit the changes
    conn.commit()

    # Close the cursor and the connection
    cursor.close()
    conn.close()

def print_table_data(table_name):
    # Connect to the SQLite database
    conn = sqlite3.connect('attendance.db')

    # Create a cursor object
    cursor = conn.cursor()

    # Execute a SELECT query to fetch all data from the specified table
    cursor.execute(f'SELECT * FROM {table_name}')

    # Fetch all rows from the result set
    rows = cursor.fetchall()

    # Print the table header
    header = [description[0] for description in cursor.description]
    print('\t'.join(header))

    # Print each row of data
    for row in rows:
        print('\t'.join(str(value) for value in row))

    # Close the cursor and the connection
    cursor.close()
    conn.close()

# # Create the database tables
# create_tables()

# # Load sample data
# load_sample_data()
    
print_table_data('courses')