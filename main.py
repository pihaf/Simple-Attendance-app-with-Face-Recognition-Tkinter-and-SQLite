import numpy as np
import face_recognition
import cv2
import os

video_capture = cv2.VideoCapture(0)

image_list = []
image_names = []
image_encodings = []

image_dir = os.listdir('images')
print(image_dir)