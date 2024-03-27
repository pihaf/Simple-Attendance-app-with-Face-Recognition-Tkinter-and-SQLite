import tkinter as tk

logged_in = False
username = ""
password = ""

def take_attendance():
    # Code for the "Take Attendance" functionality
    print("Taking attendance...")

def exit_app():
    root.destroy()

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

def show_attendance():
    menu_frame.pack_forget()
    attendance_frame.pack()
    account_frame.pack_forget()
    login_frame.pack_forget()
    register_frame.pack_forget()
    about_frame.pack_forget()

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

attendance_label = tk.Label(attendance_frame, text="Attendance Scene", font=("Arial", 16))
attendance_label.pack(pady=10)

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