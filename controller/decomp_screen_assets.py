import matplotlib
import tkinter as tk
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

    def __init__(self, parent, model):
        tk.Button.__init__(self, parent)
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
    def __init__(self, parent, model):
        tk.Button.__init__(self, parent)
        self.model = model
        self['text'] = "Start Decomposition"
        self['command'] = self.start_decomposition

    def start_decomposition(self):
        self.model.start_decomposition()



class ScreenController(tk.Frame):
    def __init__(self, parent, model, screen):
        tk.Frame.__init__(self, parent)
        self.model = model
        
        self.mouse_click_function = 'writeback'
        
        self.screen = screen
        self.screen.canvas.mpl_connect('button_press_event', 
                                       self.mouse_click)
        self._initialise_toolbar()
        self._initialise_screen()
        
    def mouse_click(self, data):
        time = data.xdata
        self.model.mouseclick(time)

    def _initialise_toolbar(self):
        self.shift_back_btn = tk.Button(self,
                                        text='Shift Forwards',
                                        command=lambda: self._shift_axis(0.2))
        self.shift_forward_btn = tk.Button(self,
                                           text='Shift Backwards',
                                           command=lambda: self._shift_axis(-0.2))
        self.zoom_in_btn = tk.Button(self,
                                     text='Zoom In',
                                     command=lambda: self._zoom_axis(0.8))
        self.zoom_out_btn = tk.Button(self,
                                      text='Zoom Out',
                                      command=lambda: self._zoom_axis(1.25))
        self.writeback_btn = tk.Button(self, 
                                      text='Write Back Mode',
                                      command=lambda: self._mode_change('writeback'))
        self.makeline_btn = tk.Button(self, 
                                      text='Make Line Mode',
                                      command=lambda: self._mode_change('makeline'))

    def _initialise_screen(self):
        self.screen.canvas.get_tk_widget().grid(row=1, column=0)
        self.zoom_in_btn.grid(row=1, column=1)
        self.zoom_out_btn.grid(row=1, column=2)
        self.shift_forward_btn.grid(row=1, column=3)
        self.shift_back_btn.grid(row=1, column=4)
        self.writeback_btn.grid(row=2, column=0)
        self.makeline_btn.grid(row=2, column=1)
        
    def _zoom_axis(self, factor):
        x_start, x_end = self.screen.ax.get_xlim()
        half_distance = (x_end - x_start) / 2
        centre = x_start + half_distance
        new_half_distance = half_distance * factor
        x_limits = [centre - new_half_distance, centre + new_half_distance]
        self.screen.update_axis(x_limits)

    def _shift_axis(self, factor):
        x_start, x_end = self.screen.ax.get_xlim()
        distance = x_end - x_start
        shift = distance * factor
        x_limits = [x_start + shift, x_end + shift]
        self.screen.update_axis(x_limits)
        
    def _mode_change(self, mode):
        self.model.change_mouseclick_mode(mode)
        
        
