import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import cv2
import sqlite3
import numpy as np
import face_recognition
from PIL import Image, ImageTk
import database_functions
import datetime
import os
import shutil
import io

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

def on_mousewheel(event):
    canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

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
new_account_face_encoding = None

def start_capture():
    global cap
    cap = cv2.VideoCapture(0)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) 
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

def stop_capture():
    if 'cap' in globals():
        cap.release()

def exit_app():
    root.destroy()

def take_attendance_button():
    global student_id_found
    result = database_functions.create_attendance_record(student_id_found, get_day_of_week(), get_current_period())
    print(result)

def destroy_contents(frame):
    for widget in frame.winfo_children():
        widget.destroy()

def show_menu():
    menu_frame.pack()
    attendance_frame.pack_forget()
    account_frame.pack_forget()
    login_frame.pack_forget()
    register_frame.pack_forget()
    about_frame.pack_forget()
    account_info_frame.pack_forget()

    stop_capture()

def show_attendance():
    menu_frame.pack_forget()
    attendance_frame.pack()
    account_frame.pack_forget()
    login_frame.pack_forget()
    register_frame.pack_forget()
    about_frame.pack_forget()
    account_info_frame.pack_forget()
    
    start_capture()
    show_video()

def show_account_info_frame():
    menu_frame.pack_forget()
    attendance_frame.pack_forget()
    account_frame.pack_forget()
    login_frame.pack_forget()
    register_frame.pack_forget()
    about_frame.pack_forget()

    display_student_info()
    account_info_frame.pack()

# Live face capture using device camera
def show_video():
    global student_id_found
    # Grab a single frame of video
    ret, frame = cap.read()
    if not ret:  # Check if the frame was successfully captured
        attendance_label.after(10, show_video)  # Retry after a delay
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

    attendance_label.after(10, show_video)

def display_student_info():
    global logged_in
    global username
    global password
    if logged_in:
        try:
            student_info = database_functions.get_student_data(username, password)
            
            student_id_label = tk.Label(info_display_frame, text=f"Student ID: {student_info['student_id']}")
            student_id_label.pack()

            name_label = tk.Label(info_display_frame, text=f"Name: {student_info['name']}")
            name_label.pack()

            dob_label = tk.Label(info_display_frame, text=f"Date of Birth: {student_info['date_of_birth']}")
            dob_label.pack()

            class_label = tk.Label(info_display_frame, text=f"Class: {student_info['class']}")
            class_label.pack()

            # Display student image
            image_data = student_info['image']
            image = Image.open(io.BytesIO(image_data))
            image = image.resize((200, 200)) 
            photo = ImageTk.PhotoImage(image)
            image_label = tk.Label(info_display_frame, image=photo)
            image_label.image = photo  # Store a reference to prevent the image from being garbage collected
            image_label.pack()
        except Exception as e:
            print("Error:", str(e))

def login():
    global logged_in
    global username
    global password
    print("Login")
    username = login_username_entry.get()
    password = login_password_entry.get()

    print("username: ", username)
    print("password: ", password)
    try:
        if "admin" in username:
            if database_functions.check_admin_login(username, password):
                logged_in = True
                account_small_frame1.pack_forget() 
                account_small_frame2.pack(pady=10)
                show_menu()
            else:
                raise ValueError('Invalid username or password of admin account.')
        else:
            if database_functions.check_student_login(username, password):
                logged_in = True
                account_small_frame1.pack_forget() 
                account_small_frame2.pack(pady=10)
                show_menu()
            else:
                raise ValueError('Invalid username or password of student account.')
    except Exception as e:
        print("Error:", str(e))

def register():
    global new_account_face_encoding
    global image_name
    global image_path
    global logged_in
    global username
    global password

    username = register_username_entry.get()
    password = register_password_entry.get()
    student_id = register_student_id_entry.get()
    name = register_name_entry.get()
    dob = register_date_of_birth_entry.get()
    student_class = register_class_entry.get()
    destination_path = os.path.join("images", image_name)  # Destination path in the "images" directory
    shutil.copy2(image_path, destination_path)  # Copy the file to the destination path
    path = 'images/' + image_name
    try:
        database_functions.create_student_record(student_id, name, dob, student_class, path)
        database_functions.create_student_account(student_id, username, password)
        student_image_encodings.append(new_account_face_encoding)
        logged_in = True
        account_small_frame1.pack_forget() 
        account_small_frame2.pack(pady=10)
        show_menu()
    except Exception as e:
        print("Error:", str(e))

def logout():
    global logged_in
    global username
    global password

    print("Logout")
    logged_in = False
    username = ""
    password = ""
    account_small_frame1.pack(pady=10)
    account_small_frame2.pack_forget()
    show_menu() 

def back_to_account():
    destroy_contents(info_display_frame)
    show_account_scene()

def show_account_scene():
    menu_frame.pack_forget()
    attendance_frame.pack_forget()
    account_frame.pack()
    login_frame.pack_forget()
    register_frame.pack_forget()
    about_frame.pack_forget()
    account_info_frame.pack_forget()

def show_login_scene():
    menu_frame.pack_forget()
    attendance_frame.pack_forget()
    account_frame.pack_forget()
    login_frame.pack()
    register_frame.pack_forget()
    about_frame.pack_forget()
    account_info_frame.pack_forget()

def show_register_scene():
    menu_frame.pack_forget()
    attendance_frame.pack_forget()
    account_frame.pack_forget()
    login_frame.pack_forget()
    register_frame.pack()
    about_frame.pack_forget()    
    account_info_frame.pack_forget()

def browse_image():
    global image_path
    global image_name
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if file_path:
        # print(file_path)
        file_name = os.path.basename(file_path)
        image_name = file_name
        image_path = file_path
        # print("Selected file name:", file_name)
        image_name_label.config(text=file_name)
        global image_object  # Store the image object as a global variable
        image_object = Image.open(file_path)
        # image_object = image_object.resize((200, 300))
        photo = ImageTk.PhotoImage(image_object.resize((200, 200)))
        image_label.config(image=photo)
        image_label.image = photo
        register_button.config(state=tk.DISABLED)# Disable register button when a new image is selected
        # destination_path = os.path.join("images", file_name)  # Destination path in the "images" directory
        # shutil.copy2(file_path, destination_path)  # Copy the file to the destination path

        # print(f"Selected file '{file_name}' copied to '{destination_path}'")
    else:
            print("No file is chosen. Please choose a file.")

def check_image():
    global new_account_face_encoding
    try: 
        image_array = np.array(image_object)  
        new_face_location = face_recognition.face_locations(image_array)
        if len(new_face_location) > 1:
            print("Too many faces detected. Please try a different image.")
            register_button.config(state=tk.DISABLED)  
        elif len(new_face_location) == 1:
            print("Detected 1 face.")
            try: 
                new_face_encoding = face_recognition.face_encodings(image_array)[0]
                new_account_face_encoding = new_face_encoding
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(student_image_encodings, new_face_encoding)

                # Use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(student_image_encodings, new_face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    print("Face already existed in the database. Please try a different image.")
                    register_button.config(state=tk.DISABLED)  
                else:
                    print("Image accepted.")
                    register_button.config(state=tk.NORMAL) 
            except Exception as e:
                print("Error:", str(e))
                print("Please try again or try a different image.")
        else:
            print("Image doesn't contain any faces. Please try a different image.")
    except Exception as e:
        print("Error:", str(e))
        print("No images selected to be checked.")
        
def show_about_scene():
    menu_frame.pack_forget()
    attendance_frame.pack_forget()
    account_frame.pack_forget()
    login_frame.pack_forget()
    register_frame.pack_forget()
    about_frame.pack()
    account_info_frame.pack_forget()

root = tk.Tk()
root.title("Attendance App")

# Configure window size and position
window_width = 1000
window_height = 600

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

menu_label = tk.Label(menu_frame, text="Attendance App", font=("Arial", 16))
menu_label.pack(pady=10)

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

account_small_frame1 = tk.Frame(account_frame)
account_small_frame1.pack(pady=10)

login_button = tk.Button(account_small_frame1, text="Login", command=show_login_scene)
login_button.grid(row=0, column=1, pady=5, padx=10, sticky="w")

register_button = tk.Button(account_small_frame1, text="Register", command=show_register_scene)
register_button.grid(row=0, column=2, pady=5, padx=10, sticky="w")

account_small_frame2 = tk.Frame(account_frame)
account_small_frame2.pack(pady=10)

account_info_button = tk.Button(account_small_frame2, text="Account info", command=show_account_info_frame)
account_info_button.grid(row=0, column=1, pady=5, padx=10, sticky="w")

logout_button = tk.Button(account_small_frame2, text="Logout", command=logout)
logout_button.grid(row=0, column=2, pady=5, padx=10, sticky="w")

account_small_frame2.pack_forget()
account_frame.pack(padx=20, pady=20)

 # Create the account info frame
account_info_frame = tk.Frame(root)

back_button_account_info = tk.Button(account_info_frame, text="Back", command=back_to_account)
back_button_account_info.pack(pady=10)

info_display_frame = tk.Frame(account_info_frame)
info_display_frame.pack(pady=10)

account_info_frame.pack(padx=20, pady=20)

# Login frame
login_frame = tk.Frame(root)

login_label = tk.Label(login_frame, text="Login Scene", font=("Arial", 16))
login_label.pack(pady=10)

back_button_account = tk.Button(login_frame, text="Back", command=show_menu)
back_button_account.pack(pady=10)

login_username_label = tk.Label(login_frame, text="Username:")
login_username_label.pack(pady=5)
login_username_entry = tk.Entry(login_frame)
login_username_entry.pack(pady=5)

login_password_label = tk.Label(login_frame, text="Password:")
login_password_label.pack(pady=5)
login_password_entry = tk.Entry(login_frame, show="*")
login_password_entry.pack(pady=5)

login_button = tk.Button(login_frame, text="Login", command=login)
login_button.pack(pady=10)

login_frame.pack(padx=20, pady=20)

# Register frame
register_frame = tk.Frame(root)
register_frame.pack(fill=tk.BOTH, expand=1)

# Create a canvas with a scrollbar
canvas = tk.Canvas(register_frame)
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

scrollbar = ttk.Scrollbar(register_frame, orient=tk.VERTICAL, command=canvas.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

canvas.configure(yscrollcommand=scrollbar.set)
canvas.bind("<MouseWheel>", on_mousewheel)
canvas.bind(
    '<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
)

# Create a frame inside the canvas
second_frame = tk.Frame(canvas)

register_label = tk.Label(second_frame, text="Login Scene", font=("Arial", 16))
register_label.pack(pady=10)

back_button_account = tk.Button(second_frame, text="Back", command=show_menu)
back_button_account.pack(pady=10)

register_username_label = tk.Label(second_frame, text="Username:")
register_username_label.pack(pady=5)
register_username_entry = tk.Entry(second_frame)
register_username_entry.pack(pady=5)

register_password_label = tk.Label(second_frame, text="Password:")
register_password_label.pack(pady=5)
register_password_entry = tk.Entry(second_frame, show="*")
register_password_entry.pack(pady=5)

register_student_id_label = tk.Label(second_frame, text="Student ID:")
register_student_id_label.pack(pady=5)
register_student_id_entry = tk.Entry(second_frame)
register_student_id_entry.pack(pady=5)

register_name_label = tk.Label(second_frame, text="Name:")
register_name_label.pack(pady=5)
register_name_entry = tk.Entry(second_frame)
register_name_entry.pack(pady=5)

register_date_of_birth_label = tk.Label(second_frame, text="Date of birth:")
register_date_of_birth_label.pack(pady=5)
register_date_of_birth_entry = tk.Entry(second_frame)
register_date_of_birth_entry.pack(pady=5)

register_class_label = tk.Label(second_frame, text="Class:")
register_class_label.pack(pady=5)
register_class_entry = tk.Entry(second_frame)
register_class_entry.pack(pady=5)

button_frame = tk.Frame(second_frame)
button_frame.pack(pady=10)

upload_image_label = tk.Label(button_frame, text="Photo:")
upload_image_label.grid(row=0, column=0, pady=5, sticky="w")

image_name = ''
image_path = ''
# Create a button to browse and upload an image
browse_button = tk.Button(button_frame, text="Browse", command=browse_image)
browse_button.grid(row=0, column=1, pady=5, padx=10, sticky="w")

check_image_button = tk.Button(button_frame, text="Check image", command=check_image)
check_image_button.grid(row=0, column=2, pady=5, padx=10, sticky="w")

# Create a label to display the uploaded image
image_label = tk.Label(second_frame)
image_label.pack(pady=10)

image_name_label = tk.Label(second_frame)
image_name_label.pack()

register_button = tk.Button(second_frame, text="Create account", command=register)
register_button.pack(pady=10)

fill_label = tk.Label(second_frame)
fill_label.pack(pady=100)

canvas.create_window((100, 0), window=second_frame, anchor="nw")

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