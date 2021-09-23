import numpy as np
import helpers as h
from itertools import product
import readers as r
import torch


def t(pars):
    a, b = pars
    return np.size(np.intersect1d(b, a))


def f(pars):
    """
    Parameters
    ----------

    pars : takes as input a tuple of 4 variables, x (prediction), z (ground truth), the current shift and the tolerance

    Returns
    ----------

    val : accuracy of prediction
    true_pos : the true positives
    false_pos : the false positives
    false_neg : the false negatives
    shift : the shift that gives the highest accuracy
    precision : true positives / (true positives + false positives)
    recall : true positives / (true positives + false negatives)
    f_measure : (2 * precision * recall) / (precision + recall)
    """

    x, z, shift, tol_range = pars
    shifted_z = z + shift
    z2 = shifted_z.reshape(-1, 1).repeat(2 * tol_range, 1)
    z2 += np.arange(-tol_range, tol_range).reshape(1, -1)
    z2 = np.transpose(z2)
    x = np.reshape(x, (1, np.size(x)))

    inter = map(t, product(x, z2))

    true_pos = sum(np.array(list(inter)))
    false_neg = np.size(x) - true_pos
    false_pos = np.size(z) - true_pos

    if true_pos + false_pos + false_neg > 0:
        val = np.round((100 * true_pos) / (true_pos + false_pos + false_neg), 1)
        if true_pos + false_pos > 0:
            precision = true_pos / (true_pos + false_pos)
        else:
            precision = np.nan
        if true_pos + false_neg > 0:
            recall = true_pos / (true_pos + false_neg)
        else:
            recall = np.nan

        if precision + recall > 0:
            f_measure = (2 * precision * recall) / (precision + recall)
        else:
            f_measure = np.nan

    else:
        val = np.nan
        precision = np.nan
        recall = np.nan
        f_measure = np.nan

    return val, int(true_pos), int(false_pos), int(false_neg), int(shift), precision, recall, f_measure


def g(pars):
    """
    Parameters
    ----------
    pars : tuple
        Inputs the predicted timestamps and the ground truth to find the accuracy between them
    """
    x, z = pars
    tol_range = 10
    shift = 100
    runs = map(
        f,
        zip(
            [x for _ in range(2 * shift)],
            [z for _ in range(2 * shift)],
            range(-shift, shift),
            [tol_range for _ in range(2 * shift)],
        ),
    )
    return max(runs, key=lambda a: a[0])


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

    def __init__(self, reader):

        super().__init__()
        self.reader = reader

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

    def load_data(self, arguments):
        """
        Loading parameters. Check the data shape is correct.
        """
        self.emg, self.f_samp = self.reader(arguments)
        h.check_data(self.emg, self.f_samp)
        self.factor = int(1000 / np.shape(self.emg)[1])
        self.est_muap = int(self.f_samp * 15 / 1000)
        self.tolerance_roa = int(self.f_samp * 5 / 1000)
        self.load_marker = True
        self.prep_marker = False

    def decompose(self):
        """
        Runs the full decomposition pipeline of gCKC, including preprocessing steps (filtering, extension and whitening)
        """

        if self.load_marker is False:
            raise Exception("Please load data before decomposition.")

        # Save original emg
        keep = self.emg

        # Check if preprocessing was done. Run if not.
        if self.prep_marker is False:
            pre_processing = h.PreProcessing(
                self.emg, self.f_samp, self.notch_freq, self.factor, self.intramuscular, self.high_pass_intra
            )
            self.emg = pre_processing.preprocess()
            self.prep_marker = True

        # Retrieve number of observations (m) and channels (n)
        self.m = np.shape(self.emg)[0]
        self.n = np.shape(self.emg)[1]
        self.act = h.activity_index(self.emg)
        self.emg = torch.tensor(self.emg)

        gradient_descent = h.GradDescent(
            self.emg,
            self.act,
            self.n,
            self.m,
            self.f_samp,
            self.source_def,
            self.cut_off_sil,
            self.factor,
            self.est_muap,
            self.tolerance_roa,
            self.num_iterations,
            self.func,
            self.clear_activations,
            self.delete_repeats,
        )
        decomp = gradient_descent.source_separate()
        decomp["EMG"] = keep

        return decomp


if __name__ == "__main__":

    path = "/home/ag4916/Documents/PhD/Data/"
    filename = "Signal10to20.mat"

    run = RunCKC(r.get_mat)
    run.load_data((path, filename))
    decomposition = run.decompose()
