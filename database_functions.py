import sqlite3
import datetime

def get_all_students_data():
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM students')
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return [list(row) for row in rows]

def get_all_students_ids():
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    cursor.execute('SELECT student_id FROM students')
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return [row[0] for row in rows]

def get_all_students_names():
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    cursor.execute('SELECT name FROM students')
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return [row[0] for row in rows]

def get_all_students_images():
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    cursor.execute('SELECT image FROM students')
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return rows

def get_all_courses_data():
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM courses')
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return [list(row) for row in rows]

# Get all courses a student is enrolled in
def get_courses_of_student(student_id):
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    cursor.execute('SELECT courses.* FROM courses JOIN student_courses ON courses.id = student_courses.course_id WHERE student_courses.student_id = ?', (student_id,))
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return [list(row) for row in rows]

# Get the course the student is studying based on the day of week and periods and location
def get_student_course_by_schedule(student_id, day_of_week, periods):
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    cursor.execute('''SELECT courses.* FROM courses
                      JOIN student_courses ON courses.id = student_courses.course_id
                      WHERE student_courses.student_id = ?
                      AND courses.day_of_week = ? AND courses.periods = ? ''',
                   (student_id, day_of_week, periods))
    row = cursor.fetchone()
    cursor.close()
    conn.close()
    return list(row) if row else None

# Create an attendance record if relevant info matched with that of the course
def create_attendance_record(student_id, day_of_week, periods):
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    course = get_student_course_by_schedule(student_id, day_of_week, periods)
    if course:
        current_date = datetime.datetime.now().date()
        
        # Check if an attendance record already exists for the student, course, and current date
        cursor.execute('SELECT * FROM attendance WHERE student_id = ? AND course_id = ? AND DATE(timestamp) = ?',
                       (student_id, course[0], current_date))
        existing_record = cursor.fetchone()
        
        if existing_record:
            cursor.close()
            conn.close()
            return "Already marked."
        
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cursor.execute('INSERT INTO attendance (student_id, course_id, timestamp, status) VALUES (?, ?, ?, ?)',
                       (student_id, course[0], timestamp, 'Marked'))
        conn.commit()
        cursor.close()
        conn.close()
        return "Marked course {}!".format(course[0])
    else:
        cursor.close()
        conn.close()
        return "No course found."
    
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

print_table_data("student_courses")