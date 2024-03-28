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
        ('INT2214 23', 'Operating system concepts','Monday', '1-2', 'Đại học Quốc gia Hà Nội'),
        ('INT3235E 20', 'Social network analysis', 'Monday','3-4', 'Đại học Quốc gia Hà Nội'),
        ('INT3229E 20', 'Big data', 'Monday', '5-6', 'Đại học Quốc gia Hà Nội'),
        ('INT3225E 20', 'Business intelligence','Monday', '7-8', 'Đại học Quốc gia Hà Nội'),
        ('INT2020E 20', 'Design information system','Monday', '9-10', 'Đại học Quốc gia Hà Nội'),
        ('INT3306E 20', 'Web development','Monday', '11-12', 'Đại học Quốc gia Hà Nội'),
        ('PES1015', 'Basketball', 'Tuesday', '3-4', 'Đại học Quốc gia Hà Nội'),
        ('PES1016', 'Football', 'Wednesday', '5-6', 'Đại học Quốc gia Hà Nội'),
        ('TEST2024', 'Test course', 'Monday', '1-2', 'Đại học Quốc gia Hà Nội'),
        ('TEST2024', 'Test course', 'Monday', '3-4', 'Đại học Quốc gia Hà Nội'),
        ('TEST2024', 'Test course', 'Monday', '5-6', 'Đại học Quốc gia Hà Nội'),
        ('TEST2024', 'Test course', 'Monday', '7-8', 'Đại học Quốc gia Hà Nội'),
        ('TEST2024', 'Test course', 'Monday', '9-10', 'Đại học Quốc gia Hà Nội'),
        ('TEST2024', 'Test course', 'Monday', '11-12', 'Đại học Quốc gia Hà Nội'),
        ('TEST2025', 'Test course', 'Tuesday', '1-2', 'Đại học Quốc gia Hà Nội'),
        ('TEST2025', 'Test course', 'Tuesday', '3-4', 'Đại học Quốc gia Hà Nội'),
        ('TEST2025', 'Test course', 'Tuesday', '5-6', 'Đại học Quốc gia Hà Nội'),
        ('TEST2025', 'Test course', 'Tuesday', '7-8', 'Đại học Quốc gia Hà Nội'),
        ('TEST2025', 'Test course', 'Tuesday', '9-10', 'Đại học Quốc gia Hà Nội'),
        ('TEST2025', 'Test course', 'Tuesday', '11-12', 'Đại học Quốc gia Hà Nội'),
        ('TEST2026', 'Test course', 'Wednesday', '1-2', 'Đại học Quốc gia Hà Nội'),
        ('TEST2026', 'Test course', 'Wednesday', '3-4', 'Đại học Quốc gia Hà Nội'),
        ('TEST2026', 'Test course', 'Wednesday', '5-6', 'Đại học Quốc gia Hà Nội'),
        ('TEST2026', 'Test course', 'Wednesday', '7-8', 'Đại học Quốc gia Hà Nội'),
        ('TEST2026', 'Test course', 'Wednesday', '9-10', 'Đại học Quốc gia Hà Nội'),
        ('TEST2026', 'Test course', 'Wednesday', '11-12', 'Đại học Quốc gia Hà Nội'),
                ('TEST2027', 'Test course', 'Thursday', '1-2', 'Đại học Quốc gia Hà Nội'),
        ('TEST2027', 'Test course', 'Thursday', '3-4', 'Đại học Quốc gia Hà Nội'),
        ('TEST2027', 'Test course', 'Thursday', '5-6', 'Đại học Quốc gia Hà Nội'),
        ('TEST2027', 'Test course', 'Thursday', '7-8', 'Đại học Quốc gia Hà Nội'),
        ('TEST2027', 'Test course', 'Thursday', '9-10', 'Đại học Quốc gia Hà Nội'),
        ('TEST2027', 'Test course', 'Thursday', '11-12', 'Đại học Quốc gia Hà Nội'),
                ('TEST2028', 'Test course', 'Friday', '1-2', 'Đại học Quốc gia Hà Nội'),
        ('TEST2028', 'Test course', 'Friday', '3-4', 'Đại học Quốc gia Hà Nội'),
        ('TEST2028', 'Test course', 'Friday', '5-6', 'Đại học Quốc gia Hà Nội'),
        ('TEST2028', 'Test course', 'Friday', '7-8', 'Đại học Quốc gia Hà Nội'),
        ('TEST2028', 'Test course', 'Friday', '9-10', 'Đại học Quốc gia Hà Nội'),
        ('TEST2028', 'Test course', 'Friday', '11-12', 'Đại học Quốc gia Hà Nội'),
    ]

    for data in course_data:
        course_id, course_name, day_of_week, periods, location = data
        cursor.execute('INSERT INTO courses (course_id, course_name, day_of_week, periods, location) VALUES (?, ?, ?, ?, ?)',
                       (course_id, course_name, day_of_week, periods, location))

    days = ['Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
    periods = ['1-2', '3-4', '5-6', '7-8', '9-10', '11-12']
    course_names = ['Database management', 'Data mining', 'Artificial intelligence', 'Software engineering', 'Computer networks']
    course_ids = ['INT4321E 20', 'INT5432E 20', 'INT6543E 20', 'INT7654E 20', 'INT8765E 20']

    for day in days:
        for period in periods:
            for i in range(len(course_names)):
                course_id = course_ids[i]
                course_name = course_names[i]
                location = 'Đại học Quốc gia Hà Nội'
                cursor.execute('INSERT INTO courses (course_id, course_name, day_of_week, periods, location) VALUES (?, ?, ?, ?, ?)',
                            (course_id, course_name, day, period, location))

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
        ('S001', 'INT2214 23'),  
        ('S002', 'TEST2024'),
        ('S002', 'TEST2025'),
        ('S002', 'TEST2026'),
        ('S002', 'TEST2027'),
        ('S002', 'TEST2028'),
        ('S003', 'INT2214 23')   
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
