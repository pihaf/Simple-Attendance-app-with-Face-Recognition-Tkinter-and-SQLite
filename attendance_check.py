import numpy as np
import face_recognition
import cv2
import sqlite3
import os

video_capture = cv2.VideoCapture(0)

# Connect to database 

# Create lists to store images information
existed_image_ids = []
existed_image_names = ['Biden', 'Nam', 'Obama']
existed_status = ['Unmark', 'Marked', 'Unmark']
existed_image_encodings = []

# Load and encode existing images
image_dir = os.listdir('images')
for img_path in image_dir:
    # print(img_path.split('.')[0])
    image = face_recognition.load_image_file('images/' + img_path)
    face_encoding = face_recognition.face_encodings(image)[0]
    existed_image_ids.append(img_path.split('.')[0])
    existed_image_encodings.append(face_encoding)

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

# Live face capture using device camera
while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Only process every other frame of video to save time
    if process_this_frame:
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        #rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(existed_image_encodings, face_encoding)
            name = "Unknown"
            course_name = "None"

            # Use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(existed_image_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = existed_image_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Check status
        if name != 'Unknown':
            if existed_status[best_match_index] == 'Unmark':
                color = (0, 0, 255)
                text = 'Unmark'
            elif existed_status[best_match_index] == 'Marked':
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
        cv2.putText(frame, "Course: " + course_name, (left + 6, bottom + 30), font, 1.0, (255, 255, 255), 1)
        cv2.putText(frame, "Status: " + text, (left + 6, bottom + 60), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Attendance app', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()