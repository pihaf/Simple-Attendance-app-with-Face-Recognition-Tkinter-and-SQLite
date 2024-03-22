import tkinter as tk

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
    about_frame.pack_forget()

def show_attendance():
    menu_frame.pack_forget()
    attendance_frame.pack()
    account_frame.pack_forget()
    about_frame.pack_forget()

def show_account_scene():
    menu_frame.pack_forget()
    attendance_frame.pack_forget()
    account_frame.pack()
    about_frame.pack_forget()

def show_about_scene():
    menu_frame.pack_forget()
    attendance_frame.pack_forget()
    account_frame.pack_forget()
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

account_frame.pack(padx=20, pady=20)

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