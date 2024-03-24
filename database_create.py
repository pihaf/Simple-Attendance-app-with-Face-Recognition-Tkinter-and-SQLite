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

    # Create the student_courses table
    cursor.execute('''CREATE TABLE IF NOT EXISTS student_courses
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id TEXT,
                    course_id TEXT,
                    FOREIGN KEY (student_id) REFERENCES students(student_id),
                    FOREIGN KEY (course_id) REFERENCES courses(id))''')


    # Create the attendance table
    cursor.execute('''CREATE TABLE IF NOT EXISTS attendance
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id TEXT,
                    course_id TEXT,
                    timestamp TEXT,
                    status TEXT,
                    FOREIGN KEY (student_id) REFERENCES students(student_id),
                    FOREIGN KEY (course_id) REFERENCES courses(id))''')

    # Create the student_accounts table
    cursor.execute('''CREATE TABLE IF NOT EXISTS student_accounts
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id TEXT UNIQUE,
                    username TEXT,
                    password TEXT,
                    FOREIGN KEY (student_id) REFERENCES students(student_id))''')

    # Create the admin_accounts table
    cursor.execute('''CREATE TABLE IF NOT EXISTS admin_accounts
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE,
                    password TEXT)''')

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
        ('C001', 'Mathematics', 'Monday', '1-2', 'Đại học Quốc gia Hà Nội'),
        ('C002', 'Science', 'Tuesday', '3-4', 'Đại học Quốc gia Hà Nội'),
        ('C003', 'English', 'Wednesday', '5-6', 'Đại học Quốc gia Hà Nội')
    ]

    for data in course_data:
        course_id, course_name, day_of_week, periods, location = data
        cursor.execute('INSERT INTO courses (course_id, course_name, day_of_week, periods, location) VALUES (?, ?, ?, ?, ?)',
                       (course_id, course_name, day_of_week, periods, location))

    # Insert sample student account data
    student_account_data = [
        ('S001', 'student1', 'password1'),
        ('S002', 'student2', 'password2'),
        ('S003', 'student3', 'password3')
    ]

    for data in student_account_data:
        student_id, username, password = data
        cursor.execute('INSERT INTO student_accounts (student_id, username, password) VALUES (?, ?, ?)',
                       (student_id, username, password))

    # Insert sample admin account data
    admin_account_data = [
        ('admin1', 'adminpassword1'),
        ('admin2', 'adminpassword2'),
       ('admin3', 'adminpassword3')
    ]

    for data in admin_account_data:
        username, password = data
        cursor.execute('INSERT INTO admin_accounts (username, password) VALUES (?, ?)',
                       (username, password))

    # Insert sample student course data
    student_course_data = [
        ('S001', 'C001'),  # Student S001 is studying course C001 (Mathematics)
        ('S002', 'C002'),  # Student S002 is studying course C002 (Science)
        ('S003', 'C003')   # Student S003 is studying course C003 (English)
    ]

    for data in student_course_data:
        student_id, course_id = data
        cursor.execute('INSERT INTO student_courses (student_id, course_id) VALUES (?, ?)',
                       (student_id, course_id))

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

# Create the database tables
create_tables()

# Load sample data
load_sample_data()

# print_table_data('courses')