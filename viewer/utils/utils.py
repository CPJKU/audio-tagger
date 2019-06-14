"""This models provides helper functions
for the GUI.

"""
import tkinter as tk

def getScreenResolution():
    """Method which returns the screen resolution
    of the current system where the GUI is executed.

    Returns
    -------
    screen_width: int
        width of the currently used screen
    screen_heigth : int
        height of the currently used screen
    """
    root = tk.Tk()

    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    return screen_width, screen_height
