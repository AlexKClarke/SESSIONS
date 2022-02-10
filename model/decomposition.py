from model import helpers as h
import torch


class RunCKC:
    """
    Class to run the gradient CKC algorithm for source separation

    Methods
    -------
    load_data(arguments)
        Loads the data.
    decompose()
        Implements the full source separation pipeline

    """

    def __init__(self, console, screen):

        super().__init__()

        # Data parameters
        self.emg = None
        self.m = None  # number of observations
        self.n = None  # number of channels
        self.f_samp = None

        # Initialised parameters
        self.num_iterations = 15
        self.cut_off_sil = 0.85
        self.func = 1
        self.factor = None  # extension factor
        self.est_muap = None  # estimated time support of muaps
        self.act = None  # activity index vector
        self.tolerance_roa = None  # rate of agreement tolerance

        # Pre processing
        self.prep_marker = None
        self.load_marker = None
        self.intramuscular = True  # used to decide on filtering
        self.high_pass_intra = False  # just high pass intramuscular (no high frequency cut off)
        self.notch_freq = 50  # Notch band-stop frequency

        # Other settings
        self.source_def = True
        self.clear_activations = True
        self.delete_repeats = True
        self.images = False
        
        self.console = console
        self.screen = screen


    def load_data_and_parameters(self, data, parameters):
        """
        Loading parameters. Check the data shape is correct.
        """
        self.data = data
        self.emg = self.data['emg']
        self.sampling_frequency = self.data['sampling_frequency']
        
        # Initialised parameters
        self.num_iterations = parameters['num_iterations']
        self.cut_off_sil = parameters['cut_off_SIL']
        self.tolerance_roa = parameters['tolerance_RoA']

        # Pre processing
        if parameters['emg_type'] == 'surface':
            self.intramuscular = False
        elif parameters['emg_type'] == 'intramuscular':
            self.intramuscular = True
        self.high_pass_intra = parameters['high_pass_only']
        self.notch_freq = parameters['notch_frequency']
        
        # Other settings
        self.delete_repeats = parameters['delete_repeats']
        self.clear_activations = parameters['clear_activations']
        self.source_def = parameters['source_deflate']

        error_message = h.check_data(self.emg, self.sampling_frequency)
        if not error_message:
            if parameters['extension_factor'] is None or parameters['use_auto_factor'] is True:
                self.factor = int(1000 / self.emg.shape[1])
            else:
                self.factor = parameters['extension_factor']
            self.est_muap = int(self.sampling_frequency * 15 / 1000)
            self.tolerance_roa = int(self.sampling_frequency * 5 / 1000)
            self.load_marker = True
            self.prep_marker = False
        else:
            self.console.message(error_message)

    def decompose(self):
        """
        Runs the full decomposition pipeline of gCKC, including preprocessing steps (filtering, extension and whitening)
        """

        if self.load_marker is False:
            self.console.message("Please load data before decomposition.")


        # Check if preprocessing was done. Run if not.
        if self.prep_marker is False:
            pre_processing = h.PreProcessing(
                self.emg, self.sampling_frequency, self.notch_freq, self.factor, self.intramuscular, self.high_pass_intra
            )
            self.emg = pre_processing.preprocess()
            self.prep_marker = True

        # Retrieve number of observations (m) and channels (n)
        self.m = self.emg.shape[0]
        self.n = self.emg.shape[1]
        self.act = h.activity_index(self.emg)
        self.emg = torch.tensor(self.emg)

        gradient_descent = h.GradDescent(
            self.emg,
            self.act,
            self.n,
            self.m,
            self.sampling_frequency,
            self.source_def,
            self.cut_off_sil,
            self.factor,
            self.est_muap,
            self.tolerance_roa,
            self.num_iterations,
            self.func,
            self.clear_activations,
            self.delete_repeats,
            self.console,
            self.screen
        )
        self.data.update(gradient_descent.source_separate())
        