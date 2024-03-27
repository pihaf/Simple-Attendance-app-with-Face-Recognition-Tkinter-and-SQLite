import numpy as np
import face_recognition
import cv2
import sqlite3
import os
from geopy.geocoders import Nominatim
import database_functions
import datetime
import tkinter as tk
from PIL import Image, ImageTk

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

# The problem of matching the location which a student take attendance with that of a course is a bit complex
# and dependent on different devices GPS so I gave up on this part
# def get_location():
# Get latitude and longtitude by making request to a ip geolocation website
#
# Use geopy to find the name of the location
#     geolocator = Nominatim(user_agent="my_app")
#     location = geolocator.reverse((latitude, longitude))
#     return location

# Set up GUI
window = tk.Tk()  # Makes main window
window.wm_title("Digital Microscope")
window.config(background="#FFFFFF")

# Graphics window
imageFrame = tk.Frame(window, width=600, height=500)
imageFrame.grid(row=0, column=0, padx=10, pady=2)

# Capture video frames
lmain = tk.Label(imageFrame)
lmain.grid(row=0, column=0)

cap = cv2.VideoCapture(0)

student_ids = database_functions.get_all_students_ids()
student_names = database_functions.get_all_students_names()
student_images = database_functions.get_all_students_images()
status = ['Unmark', 'Marked', 'Unmark']
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

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
student_id_found = ''

# Live face capture using device camera
def show_frame():
    global process_this_frame  
    global student_id_found
    # Grab a single frame of video
    ret, frame = cap.read()
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
            print("Student ID: ", student_id)
            print("Day of week: ", get_day_of_week())
            print("Periods: ", get_current_period())
            print("Course retrieved: ", course_data)
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
            if status[best_match_index] == 'Unmark':
                color = (0, 0, 255)
                text = 'Unmark'
            elif status[best_match_index] == 'Marked':
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
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)

    lmain.after(10, show_frame)

# Button function to be executed when the button is pressed
def button_pressed():
    result = database_functions.create_attendance_record(student_id_found, get_day_of_week(), get_current_period())
    print(result)

# Button widget
button = tk.Button(window, text="Take attendance", command=button_pressed)
button.grid(row=1, column=0, pady=10)

# Slider window (slider controls stage position)
# sliderFrame = tk.Frame(window, width=600, height=100)
# sliderFrame.grid(row=2, column=0, padx=10, pady=2)

show_frame()  # Display video frames
window.mainloop()  # Starts GUI

# Release handle to the webcam
cap.release()
cv2.destroyAllWindows()