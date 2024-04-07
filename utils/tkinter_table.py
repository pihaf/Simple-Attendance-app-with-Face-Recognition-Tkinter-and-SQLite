import tkinter as tk
from tkinter import ttk

class Table(tk.Frame):
    def __init__(self, parent, columns, data, column_width=None, column_minwidth=None):
        super().__init__(parent)

        self.tree = ttk.Treeview(self)
        self.tree["columns"] = columns

        # Define column properties
        if column_width:
            for column in columns:
                self.tree.column(column, width=column_width, minwidth=column_minwidth)
                self.tree.heading(column, text=column)
        else:
            for column in columns:
                self.tree.heading(column, text=column)

        # Insert data rows
        for item in data:
            self.tree.insert("", "end", values=item)

        # Pack the treeview widget
        self.tree.pack()

class ScrollableTable(tk.Frame):
    def __init__(self, parent, columns, data):
        super().__init__(parent)
        
        self.tree = ttk.Treeview(self)
        self.tree["columns"] = columns

        # Define column properties
        for column in columns:
            self.tree.column(column, width=100, minwidth=50)
            self.tree.heading(column, text=column)

        # Insert data rows
        for item in data:
            self.tree.insert("", "end", values=item)

        # Create a vertical scrollbar
        scrollbar = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)

        # Pack the treeview and scrollbar
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)