import tkinter as tk
from controller import main_screen_assets
from controller import decomp_screen_assets
from model import model_core
from view import view_core


class Controller(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master=master)
        self.console = view_core.Console()
        self.model = model_core.Model(self.console)
        self.main_screen = MainScreen(self)
        self.decomp_screen = DecompositionScreen(self)

        self.main_screen.grid(row=0, column=0, sticky="nsew")
        self.decomp_screen.grid(row=0, column=0, sticky="nsew")

        self.main_screen.tkraise()


class MainScreen(tk.Frame):
    def __init__(self, controller):
        tk.Frame.__init__(self)
        reader_btn = main_screen_assets.ReaderBtn(self, controller.model)
        load_data_btn = main_screen_assets.LoadDataBtn(self, controller.model)
        decomp_button = tk.Button(self,
                                  text="Decomposition Screen",
                                  command=lambda: controller.decomp_screen.tkraise())

        reader_btn.grid(row=0, column=0)
        load_data_btn.grid(row=1, column=0)
        decomp_button.grid(row=2, column=0)
        controller.console.grid(row=3, column=0)


class DecompositionScreen(tk.Frame):
    def __init__(self, controller):
        tk.Frame.__init__(self)
        ms_button = tk.Button(self,
                              text="Back to Home",
                              command=lambda: controller.main_screen.tkraise())
        parameters_btn = decomp_screen_assets.ParametersBtn(self, controller.model)
        start_decomp_btn = decomp_screen_assets.StartDecompBtn(self, controller.model)

        parameters_btn.grid(row=1, column=0)
        start_decomp_btn.grid(row=2, column=0)
        ms_button.grid(row=3, column=0)
        controller.console.grid(row=4, column=0)


