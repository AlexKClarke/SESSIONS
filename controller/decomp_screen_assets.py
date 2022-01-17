import matplotlib
import numpy as np
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
matplotlib.use('TkAgg')
from parameter_defaults import parameters


class ParametersBtn(tk.Button):
    """
    Class to perform pre-processing steps before applying source separation. Performs filtering, extension and
    whitening of data.

    Attributes
    ----------
    emg : array
        The raw emg
    f_samp : int
        Sampling frequency of recording
    u_band : int
        Upper limit band of cut-off frequency for filtering
    l_band : int
        Lower limit band of cut-off frequency for filtering
    notch_freq : int
        Notch frequency to filter out
    factor: int
        Extension factor for extending the observations.

    Methods
    -------
    filter_data
        Filters data with bandpass filter
    extend_data
        Extends data by the extension factor
    whiten_data
        Spatial whitening of the data
    """

    def __init__(self, master, model):
        tk.Button.__init__(self, master=master)
        self.parameters = parameters
        self.model = model
        self['text'] = "Decomposition Settings"
        self['command'] = self.display_options

    def display_options(self):
        option_menu = tk.Toplevel()
        option_menu.wm_title("Decomposition Parameters")
        select_btn = tk.Button(option_menu, 
                               text="Confirm Choices",
                               command=lambda: self.select(option_menu))
        
        self.rows = []
        self.responses = []
        self.keys = list(self.parameters.keys())
        self.bools = [0] * len(self.keys)
        for i, key in enumerate(self.keys):
            if type(self.parameters[key]) == bool:
                self.responses.append(tk.IntVar(value=int(self.parameters[key])))
                self.rows.append(tk.Checkbutton(option_menu, variable=self.responses[i]))
                self.bools[i] = 1
            elif type(self.parameters[key]) == float:
                self.responses.append(tk.DoubleVar(value=self.parameters[key]))
                self.rows.append(tk.Entry(option_menu, text=self.responses[i]))
            elif type(self.parameters[key]) == int:
                self.responses.append(tk.IntVar(value=self.parameters[key]))
                self.rows.append(tk.Entry(option_menu, text=self.responses[i]))
            else:
                self.responses.append(tk.StringVar(value=self.parameters[key]))
                self.rows.append(tk.Entry(option_menu, text=self.responses[i]))
        
        for i, entry in enumerate(self.rows):
            tk.Label(option_menu, text=self.keys[i].replace('_', ' ')).grid(row=i, column=0)
            entry.grid(row=i, column=1)
        select_btn.grid(row=(i+1), column=1)

    def select(self, option_menu):
        for i, b in enumerate(self.bools):
            if b == 1: 
                self.responses[i] = bool(self.responses[i].get())
            else:
                self.responses[i] = self.responses[i].get()
        for i, key in enumerate(self.keys):
            self.parameters[key] = self.responses[i]
        self.model.modify_parameters(self.parameters)
        option_menu.destroy()
        
class StartDecompBtn(tk.Button):
    def __init__(self, master, model):
        tk.Button.__init__(self, master=master)
        self.model = model
        self['text'] = "Start Decomposition"
        self['command'] = self.start_decomposition

    def start_decomposition(self):
        self.model.start_decomposition()




def callback(event):
    print("clicked at", event.xdata, event.ydata)


class Application(tk.Tk):
    def __init__(self, parent):
        self.parent = parent
        self.build_buttons()
        self.build_canvas()
        self.build_interface()

    def build_interface(self):
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.canvas.tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.zoom_in_btn.pack()
        self.zoom_out_btn.pack()
        self.slide_left_btn.pack()
        self.slide_right_btn.pack()

    def build_buttons(self):
        self.zoom_in_btn = tk.Button(text='Zoom In', command=lambda: self.zoom_axis(0.8))
        self.zoom_out_btn = tk.Button(text='Zoom Out', command=lambda: self.zoom_axis(1.25))
        self.slide_left_btn = tk.Button(text='Slide Left', command=lambda: self.slide_axis(-0.2))
        self.slide_right_btn = tk.Button(text='Slide Right', command=lambda: self.slide_axis(0.2))

    def build_canvas(self):
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.parent)
        self.canvas.draw()
        self.canvas.mpl_connect('button_press_event', callback)
        self.ax = self.fig.add_subplot(111)

        t = np.arange(0.0, 3.0, 0.01)
        s = np.sin(2 * np.pi * t)
        self.ax.plot(t, s)

        self.x_limits = self.ax.get_xlim()

    def update_axis(self):
        self.ax.set_xlim(self.x_limits)
        self.canvas.draw()

    def zoom_axis(self, factor):
        x_start, x_end = self.x_limits
        half_distance = (x_end - x_start) / 2
        centre = x_start + half_distance
        new_half_distance = half_distance * factor
        self.x_limits = [centre - new_half_distance, centre + new_half_distance]
        self.update_axis()

    def slide_axis(self, factor):
        x_start, x_end = self.x_limits
        distance = x_end - x_start
        shift = distance * factor
        self.x_limits = [x_start + shift, x_end + shift]
        self.update_axis()
