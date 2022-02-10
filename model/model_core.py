import torch
from parameter_defaults import parameters
from model import decomposition


class Model(object):
    def __init__(self, console, screen):
        self.console = console
        self.screen = screen
        self.reader = None
        self.data = None
        self.data_loaded = False
        self.parameters = parameters
        self.check_gpu()
        self.decomp_algorithm = decomposition.RunCKC(self.console, self.screen)
        self.click_mode = 'writeback'
        
    def check_gpu(self):
        if torch.cuda.is_available():
            memory = torch.cuda.get_device_properties('cuda:0').total_memory
            memory = str(round(memory / 1024**3, 2))
            input_text = "GPU available with " + memory + " GB of memory."
        else:
            input_text = "No GPU detected - check Pytorch / CUDA."
        self.console.message(input_text)

    def load_reader(self, reader_name, reader):
        if reader_name == 'unselected':
            input_text = "Reader not selected."
        else:
            self.reader = reader
            input_text = "Reader " + reader_name + " has been successfully loaded."
        self.console.message(input_text)
        
    def load_data(self, file_path):
        if self.reader is None:
            input_text = "Cannot read file as a reader has not been selected."
            self.console.message(input_text)
        else:
            self.data = self.reader(file_path)
            self.check_data()

    def check_data(self):
        correct = True
        error_message = []
        if 'emg' not in list(self.data.keys()):
            correct = False
            error_message.append("Data dict does not contain emg variable.")
        if 'sampling_frequency' not in list(self.data.keys()):
            correct = False
            error_message.append("Data dict does not contain sampling_frequency variable.")
        if correct is False:
            input_text = ''
            for message in error_message:
                input_text = input_text + message + "\n"
            self.console.message(input_text)
        else:
            input_text = "Data has been successfully loaded and checked."
            self.console.message(input_text)
            self.data_loaded = True
            
    def modify_parameters(self, parameters):
        self.parameters = parameters
        keys = list(self.parameters.keys())
        input_text = ''
        for key in keys:
            message = key.replace('_', ' ') + ': ' + str(self.parameters[key])
            input_text = input_text + message + "\n"
        self.console.message(input_text)
        
    def start_decomposition(self):
        if self.data_loaded:
            self.decomp_algorithm.load_data_and_parameters(self.data, self.parameters)
            self.decomp_algorithm.decompose()
        else:
            self.console.message('Please load data.')
        
    def change_mouseclick_mode(self, mode):
        self.click_mode = mode
        input_text = 'Now in ' + mode + ' mode.'
        self.console.message(input_text)
         
    def mouseclick(self, time):
        if self.click_mode == 'writeback':
            input_text = str(time)
            self.console.message(input_text)
        if self.click_mode == 'makeline':
            import numpy as np
            x = time*np.ones(100)
            self.screen.update_plot(x)
        
    
    
      