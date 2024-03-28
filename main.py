import tkinter as tk
import cv2
import numpy as np
import face_recognition
from PIL import Image, ImageTk
import database_functions
import datetime

def get_day_of_week():
    current_date = datetime.date.today()

    # Convert the date to the day of the week
    day_of_week = current_date.strftime("%A")

    return day_of_week

def get_current_period():
    current_time = datetime.datetime.now().time()

    if current_time >= datetime.time(7, 0) and current_time < datetime.time(8, 50):
        return '1-2'
    elif current_time >= datetime.time(9, 0) and current_time < datetime.time(10, 50):
        return '3-4'
    elif current_time >= datetime.time(11, 0) and current_time < datetime.time(12, 50):
        return '5-6'
    elif current_time >= datetime.time(13, 0) and current_time < datetime.time(14, 50):
        return '7-8'
    elif current_time >= datetime.time(15, 0) and current_time < datetime.time(16, 50):
        return '9-10'
    elif current_time >= datetime.time(17, 0) and current_time < datetime.time(18, 50):
        return '11-12'
    else:
        return 'No matching period found.'

student_ids = database_functions.get_all_students_ids()
student_names = database_functions.get_all_students_names()
student_images = database_functions.get_all_students_images()
student_status = ['Unmark', 'Marked', 'Unmark']
student_image_encodings = []

for row in student_images:
    image_bytes = row[0]
    # Convert the image bytes to a numpy array
    image_array = np.frombuffer(image_bytes, np.uint8)
    # Decode the image array
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    # Convert the image from BGR to RGB (face_recognition uses RGB format)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Encode the face in the image
    face_encoding = face_recognition.face_encodings(image_rgb)[0]
    student_image_encodings.append(face_encoding)

student_id_found = ''
logged_in = False
username = ""
password = ""

def start_capture():
    global cap
    cap = cv2.VideoCapture(0)

def stop_capture():
    if 'cap' in globals():
        cap.release()

def exit_app():
    root.destroy()

def take_attendance_button():
    global student_id_found
    result = database_functions.create_attendance_record(student_id_found, get_day_of_week(), get_current_period())
    print(result)

def show_account():
    # Code for the "Account" functionality
    print("Showing account...")

def show_about():
    # Code for the "About" functionality
    print("Showing about...")

def show_menu():
    menu_frame.pack()
    attendance_frame.pack_forget()
    account_frame.pack_forget()
    login_frame.pack_forget()
    register_frame.pack_forget()
    about_frame.pack_forget()

    stop_capture()

def show_attendance():
    menu_frame.pack_forget()
    attendance_frame.pack()
    account_frame.pack_forget()
    login_frame.pack_forget()
    register_frame.pack_forget()
    about_frame.pack_forget()
    
    start_capture()
    show_frame()

# Live face capture using device camera
def show_frame():
    global student_id_found
    # Grab a single frame of video
    ret, frame = cap.read()
    if not ret:  # Check if the frame was successfully captured
        attendance_label.after(10, show_frame)  # Retry after a delay
        return
    frame = cv2.flip(frame, 1)
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    #rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []
    course_data = []
    course_name = 'Not found'
    student_id = 'Not found'
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(student_image_encodings, face_encoding)
        name = "Unknown"

        # Use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(student_image_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = student_names[best_match_index]
            student_id = student_ids[best_match_index]
            student_id_found = student_ids[best_match_index]

            # Get the course the student is supposed to study at this time 
            course_data = database_functions.get_student_course_by_schedule(student_id=student_id, day_of_week=get_day_of_week(), periods=get_current_period())
            # print("Student ID: ", student_id)
            # print("Day of week: ", get_day_of_week())
            # print("Periods: ", get_current_period())
            # print("Course retrieved: ", course_data)
            if course_data != None:
                course_name = course_data[1]
                
        face_names.append(name)

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Check status
        if name != 'Unknown':
            if student_status[best_match_index] == 'Unmark':
                color = (0, 0, 255)
                text = 'Unmark'
            elif student_status[best_match_index] == 'Marked':
                color = (0,128,0)
                text = 'Marked'
        else:
            color = (0, 0, 255)
            text = ''

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom + 40), color, cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, "Name: " + name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        cv2.putText(frame, "ID: " + student_id, (left + 6, bottom +30), font, 1.0, (255, 255, 255), 1)
        cv2.putText(frame, "Course: " + course_name, (left + 6, bottom + 60), font, 1.0, (255, 255, 255), 1)
        cv2.putText(frame, "Status: " + text, (left + 6, bottom +90), font, 1.0, (255, 255, 255), 1)

    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)  # Convert frame back to RGBA format
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    attendance_label.imgtk = imgtk
    attendance_label.configure(image=imgtk)

    attendance_label.after(10, show_frame)

def login():
    global logged_in
    global username
    global password
    # Code for the login functionality
    print("Login")
    username = username_entry.get()
    password = password_entry.get()
    # Add your login logic here
    if username == "admin" and password == "password":
        logged_in = True
        account_button.pack_forget()  
        show_menu()

def register():
    global logged_in
    global username
    global password
    # Code for the register functionality
    print("Register")
    username = username_entry.get()
    password = password_entry.get()
    # Add your register logic here
    logged_in = True
    account_button.pack_forget()  # Hide the login button when logged in
    show_menu()

def logout():
    global logged_in
    global username
    global password
    # Code for the logout functionality
    print("Logout")
    logged_in = False
    username = ""
    password = ""
    account_button.pack()  # Show the login button when logged out
    show_menu()  # Go back to the main menu after logout

def show_account_scene():
    menu_frame.pack_forget()
    attendance_frame.pack_forget()
    account_frame.pack()
    login_frame.pack_forget()
    register_frame.pack_forget()
    about_frame.pack_forget()

def show_login_scene():
    menu_frame.pack_forget()
    attendance_frame.pack_forget()
    account_frame.pack_forget()
    login_frame.pack()
    register_frame.pack_forget()
    about_frame.pack_forget()

def show_register_scene():
    menu_frame.pack_forget()
    attendance_frame.pack_forget()
    account_frame.pack_forget()
    login_frame.pack_forget()
    register_frame.pack()
    about_frame.pack_forget()    

def show_about_scene():
    menu_frame.pack_forget()
    attendance_frame.pack_forget()
    account_frame.pack_forget()
    login_frame.pack_forget()
    register_frame.pack_forget()
    about_frame.pack()

root = tk.Tk()
root.title("Attendance App")

# Configure window size and position
window_width = 600
window_height = 400

screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

x = (screen_width // 2) - (window_width // 2)
y = (screen_height // 2) - (window_height // 2)

root.geometry(f"{window_width}x{window_height}+{x}+{y}")
root.resizable(True, True)

# Make the window draggable
def drag_window(event):
    x = root.winfo_pointerx() - root._offset_x
    y = root.winfo_pointery() - root._offset_y
    root.geometry(f"+{x}+{y}")

def start_drag(event):
    root._offset_x = event.x
    root._offset_y = event.y

title_bar = tk.Frame(root, bg="gray")
title_bar.bind("<B1-Motion>", drag_window)
title_bar.bind("<Button-1>", start_drag)
title_bar.pack(fill="x")

# Create the main menu frame
menu_frame = tk.Frame(root)

attendance_button = tk.Button(menu_frame, text="Take Attendance", command=show_attendance)
attendance_button.pack(pady=10)

account_button = tk.Button(menu_frame, text="Account", command=show_account_scene)
account_button.pack(pady=10)

about_button = tk.Button(menu_frame, text="About", command=show_about_scene)
about_button.pack(pady=10)

exit_button = tk.Button(menu_frame, text="Exit", command=exit_app)
exit_button.pack(pady=10)

menu_frame.pack(padx=20, pady=20)

# Create the attendance frame
attendance_frame = tk.Frame(root)

attendance_label = tk.Label(attendance_frame)
attendance_label.pack()

button = tk.Button(attendance_frame, text="Take attendance", command=take_attendance_button)
button.pack(pady=10)

back_button_attendance = tk.Button(attendance_frame, text="Back", command=show_menu)
back_button_attendance.pack(pady=10)

attendance_frame.pack(padx=20, pady=20)

# Create the account frame
account_frame = tk.Frame(root)

account_label = tk.Label(account_frame, text="Account Scene", font=("Arial", 16))
account_label.pack(pady=10)

back_button_account = tk.Button(account_frame, text="Back", command=show_menu)
back_button_account.pack(pady=10)

login_button = tk.Button(account_frame, text="Login", command=show_login_scene)
login_button.pack(pady=10)

register_button = tk.Button(account_frame, text="Register", command=show_register_scene)
register_button.pack(pady=10)

account_frame.pack(padx=20, pady=20)

# Login frame
login_frame = tk.Frame(root)

login_label = tk.Label(login_frame, text="Login Scene", font=("Arial", 16))
login_label.pack(pady=10)

back_button_account = tk.Button(login_frame, text="Back", command=show_menu)
back_button_account.pack(pady=10)

username_label = tk.Label(login_frame, text="Username:")
username_label.pack(pady=5)
username_entry = tk.Entry(login_frame)
username_entry.pack(pady=5)

password_label = tk.Label(login_frame, text="Password:")
password_label.pack(pady=5)
password_entry = tk.Entry(login_frame, show="*")
password_entry.pack(pady=5)

login_button = tk.Button(login_frame, text="Login", command=login)
login_button.pack(pady=10)

login_frame.pack(padx=20, pady=20)

# Register frame
register_frame = tk.Frame(root)

register_label = tk.Label(register_frame, text="Login Scene", font=("Arial", 16))
register_label.pack(pady=10)

back_button_account = tk.Button(register_frame, text="Back", command=show_menu)
back_button_account.pack(pady=10)

username_label = tk.Label(register_frame, text="Username:")
username_label.pack(pady=5)
username_entry = tk.Entry(register_frame)
username_entry.pack(pady=5)

password_label = tk.Label(register_frame, text="Password:")
password_label.pack(pady=5)
password_entry = tk.Entry(register_frame, show="*")
password_entry.pack(pady=5)

register_button = tk.Button(register_frame, text="Create account", command=register)
register_button.pack(pady=10)

register_frame.pack(padx=20, pady=20)

# Create the about frame
about_frame = tk.Frame(root)

about_label = tk.Label(about_frame, text="About Scene", font=("Arial", 16))
about_label.pack(pady=10)

back_button_about = tk.Button(about_frame, text="Back", command=show_menu)
back_button_about.pack(pady=10)

about_frame.pack(padx=20, pady=20)

# Show the initial menu scene
show_menu()

root.mainloop()