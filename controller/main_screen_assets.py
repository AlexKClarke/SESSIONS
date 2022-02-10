import tkinter as tk
import inspect
import readers

class ReaderBtn(tk.Button):
    def __init__(self, parent, model):
        tk.Button.__init__(self, parent)
        self.model = model
        self['text'] = "Select Reader"
        self['command'] = self.display_options

    def display_options(self):
        functions = self.get_functions()
        option_menu = tk.Toplevel()
        option_menu.wm_title("Reader Function")
        selected_function = tk.StringVar(option_menu, 'Available Readers')
        label = tk.Label(option_menu, text="Select reader for data extraction:")
        dropdown = tk.OptionMenu(option_menu, selected_function, functions)
        select_btn = tk.Button(option_menu,
                               text="Select",
                               command=lambda: self.select(option_menu, selected_function.get()))

        label.grid(row=0, column=0)
        dropdown.grid(row=1, column=0)
        select_btn.grid(row=2, column=0)

    def select(self, option_menu, selected):
        if selected == 'Available_Readers':
            selected_name = 'unselected'
            selected_function = None
        else:
            selected_name = selected[2:-3]
            selected_function = getattr(readers, selected_name)
        self.model.load_reader(selected_name, selected_function)
        option_menu.destroy()

    @staticmethod
    def get_functions():
        functions = []
        for name, obj in inspect.getmembers(readers):
            if inspect.isfunction(obj):
                functions.append(name)
        return functions


class LoadDataBtn(tk.Frame):
    def __init__(self, parent, model):
        tk.Button.__init__(self, parent)
        self.model = model
        self['text'] = "Select File"
        self['command'] = self.select_file
        
    def select_file(self):
        file_path = tk.filedialog.askopenfilename()
        self.model.load_data(file_path)






