import tkinter as tk
from controller.controller_core import Controller



if __name__ == "__main__":
    root = tk.Tk()
    Controller(root).grid()
    root.mainloop()


