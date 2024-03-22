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

