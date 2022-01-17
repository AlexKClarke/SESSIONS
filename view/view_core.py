import tkinter as tk


class Console(tk.Text):
    """
    Console class acting as the main view of the UI 

    Methods
    -------
    message
        overwrites previous text in console with new text

    """

    def __init__(self, *args, **kwargs):
        tk.Text.__init__(self, *args, **kwargs)

    def message(self, input_text):
        self.delete(1.0, tk.END)
        self.insert(tk.END, str(input_text))
