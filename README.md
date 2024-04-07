# Simple attendance app with Face Recognition, Tkinter and SQLite

This is a simple Attendance app made with face_recognition, tkinter and sqlite. The app takes UET attendance app 'FacePlus' as inspiration.

## Installation
### Requirements
- Python 3.3+
- dlib and cmake to run face_recognition
- VSCode is recommended for easy installation

### Installation guide
The Visual studio can be downloaded in the link https://visualstudio.microsoft.com/visual-cpp-build-tools/. After finishing the installation, you need to install additional packages for C, C++ programming, which is Packages CMake tools for Windows.

Next download cmake and dlib libraries
```pip install cmake```

```pip install dlib```
If you have trouble installing cmake and dlib, read this reference: https://medium.com/analytics-vidhya/how-to-install-dlib-library-for-python-in-windows-10-57348ba1117f

After that, install face_recognition
```pip install face_recognition```

### Running
Run `main.py` to run the whole app. There is also attendance_check module which can be run separately for testing.
The admin functions are still limited so to make operations on the database, you might update the code or have to manually change the data.

### Thanks
face_recognition is a library to recognize and manipulate faces from Python. Check out: https://github.com/ageitgey/face_recognition
