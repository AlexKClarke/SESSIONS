import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class Console(tk.Text):
    """
    Console class acting as the main view of the UI 

    Methods
    -------
    message
        overwrites previous text in console with new text

    """

    def __init__(self, parent):
        tk.Text.__init__(self, parent)

    def message(self, input_text):
        self.delete(1.0, tk.END)
        self.insert(tk.END, str(input_text))

class IPTScreen(tk.Frame):
    def __init__(self, parent):
        tk.Frame.__init__(self)
        self.parent = parent
        self._initialise_plot()

    def _initialise_plot(self):
        self.fig = Figure(figsize=(20, 10))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, self.parent)
        self.canvas.draw()
     
    def update_axis(self, x_limits):
        self.ax.set_xlim(x_limits)
        self.canvas.draw()  
        self.update_idletasks()
    
    def update_plot(self, IPT):
        self.ax.clear()
        self.ax.plot(IPT)
        self.canvas.draw()  
        self.update_idletasks()
        
        


        
        
        

        


