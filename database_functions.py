import sqlite3
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, Binary
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.ext.declarative import declarative_base

def create_sqlitetables():
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

# Create the database engine
engine = create_engine('sqlite:///attendance.db')
Session = sessionmaker(bind=engine)
session = Session()

Base = declarative_base()

# Define the ORM models
class Student(Base):
    __tablename__ = 'students'
    id = Column(Integer, primary_key=True)
    student_id = Column(String, unique=True)
    name = Column(String)
    date_of_birth = Column(String)
    class_name = Column(String)
    image = Column(Binary)

class Course(Base):
    __tablename__ = 'courses'
    id = Column(Integer, primary_key=True)
    course_id = Column(String)
    course_name = Column(String)
    day_of_week = Column(String)
    periods = Column(String)
    location = Column(String)

class StudentCourse(Base):
    __tablename__ = 'student_courses'
    id = Column(Integer, primary_key=True)
    student_id = Column(String, ForeignKey('students.student_id'))
    course_id = Column(Integer, ForeignKey('courses.id'))
    student = relationship("Student")
    course = relationship("Course")

class Attendance(Base):
    __tablename__ = 'attendance'
    id = Column(Integer, primary_key=True)
    student_id = Column(String, ForeignKey('students.student_id'))
    course_id = Column(Integer, ForeignKey('courses.id'))
    timestamp = Column(String)
    status = Column(String)

class StudentAccount(Base):
    __tablename__ = 'student_accounts'
    id = Column(Integer, primary_key=True)
    student_id = Column(String, ForeignKey('students.student_id'), unique=True)
    username = Column(String)
    password = Column(String)

class AdminAccount(Base):
    __tablename__ = 'admin_accounts'
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True)
    password = Column(String)

# Create the tables
Base.metadata.create_all(engine)

# Load sample data
def load_sample_data():
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
        student = Student(student_id=student_id, name=name, date_of_birth=dob, class_name=class_name, image=image_data)
        session.add(student)

    # Insert sample course data
    course_data = [
        ('C001', 'Mathematics', 'Monday', '1-2', 'Room 101'),
        ('C002', 'Science', 'Tuesday', '3-4', 'Room 202'),
        ('C003', 'English', 'Wednesday', '5-6', 'Room 303')
    ]

    for data in course_data:
        course_id, course_name, day_of_week, periods, location = data
        course = Course(course_id=course_id, course_name=course_name, day_of_week=day_of_week, periods=periods, location=location)
        session.add(course)

    # Insert sample student course data
    student_course_data = [
        ('S001', 'C001'),  
        ('S002', 'C002'),  
        ('S003', 'C003')   
    ]

    for data in student_course_data:
        student_id, course_id = data
        student_course = StudentCourse(student_id=student_id, course_id=course_id)
        session.add(student_course)

    # Insert sample student account data
    student_account_data = [
        ('S001', 'student1', 'password1'),
        ('S002', 'student2', 'password2'),
        ('S003', 'student3', 'password3')
    ]

    for data in student_account_data:
        student_id, username, password = data
        student_account = StudentAccount(student_id=student_id, username=username, password=password)
        session.add(student_account)

    # Insert sample admin account data
    admin_account_data = [
        ('admin1', 'adminpassword1'),
        ('admin2', 'adminpassword2'),
        ('admin3', 'adminpassword3')
    ]

    for data in admin_account_data:
        username, password = data
        admin_account = AdminAccount(username=username, password=password)
        session.add(admin_account)

    # Commit the changes
    session.commit()

def print_table_data(table_name):
    # Fetch all rows from the specified table
    if table_name == 'students':
        rows = session.query(Student).all()
    elif table_name == 'courses':
        rows = session.query(Course).all()
    elif table_name == 'student_courses':
        rows = session.query(StudentCourse).all()
    elif table_name == 'attendance':
        rows = session.query(Attendance).all()
    elif table_name == 'student_accounts':
        rows = session.query(StudentAccount).all()
    elif table_name == 'admin_accounts':
        rows = session.query(AdminAccount).all()
    else:
        print(f"Table '{table_name}' does not exist.")
        return

    # Print the table header
    header = [column.name for column in rows[0].__table__.columns]
    print('\t'.join(header))

    # Print each row of data
    for row in rows:
        values = [str(getattr(row, column)) for column in row.__table__.columns]
        print('\t'.join(values))

# Load sample data
load_sample_data()

# Print the 'admin_accounts' table
print_table_data('admin_accounts')