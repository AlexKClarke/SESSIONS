import numpy as np
import torch
from scipy.signal import butter, filtfilt, iirnotch, find_peaks
from itertools import product
# from torch_scatter import scatter_add


# def scatter_mean(src, index, dim=-1, out=None, dim_size=None, fill_value=0):
#
#     out = scatter_add(src, index, dim, out, dim_size, fill_value)
#     count = scatter_add(torch.ones_like(src), index, dim, None, out.size(dim))
#     return out / count.clamp(min=1)


def scatter_mean(src, index):
    out = torch.zeros((int(index.flatten().max()) + 1), dtype=src.dtype).scatter_add_(-1, index, src)
    count = torch.zeros((int(index.flatten().max()) + 1), dtype=src.dtype).scatter_add_(-1, index,
                                                                                                     torch.ones_like(
                                                                                                         src))

    return out / count.clamp(min=1)


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


def check_data(emg, sampling_frequency):
    """
    Checks the EMG is a samples x channels numpy matrix
    Checks sampling frequency is an int.

    Parameters
    ----------
    emg : ndarray
        The raw emg
    f_samp : int
        The sampling frequency of the recording
    """
    error_message = []
    try:
        emg
    except NameError:
        emg = None
    if emg is None:
        error_message.append("Please input the EMG matrix.\n")
    else:
        if type(emg) is not torch.Tensor:
            error_message.append("Please input EMG matrix as torch tensor.\n")
        else:
            if emg.shape[1] > emg.shape[0]:
                error_message.append("EMG matrix must be of shape (samples,channels)\n")
    try:
        sampling_frequency
    except NameError:
        sampling_frequency = None
    if sampling_frequency is None:
        error_message.append("Please input the sampling frequency.\n")
    else:
        if type(sampling_frequency) is not float and type(sampling_frequency) is not int:
            error_message.append("Please input sampling frequency as scalar int or float.\n")
    
    return error_message


def activity_index(emg):
    """
    Calculates the sum of squared absolute values as a metric of activity.
    Then uses peak finding algorithm to convert to timestamps.

    Parameters
    ----------
    emg : ndarray
        The post-processing emg
    """
    val = np.abs(np.transpose(emg)) ** 2
    val = val.sum(axis=0)

    # Normalise activity index
    val = val - np.median(val)
    q75, q25 = np.percentile(val, [75, 25])
    iqr = q75 - q25
    val = val / iqr

    # Find peak locations and values
    peak_locs, properties = find_peaks(val, height=0)
    peak_vals = properties["peak_heights"]
    peak_locs = np.expand_dims(peak_locs, axis=1)
    peak_vals = np.expand_dims(peak_vals, axis=1)
    val = np.concatenate([peak_locs, peak_vals], axis=1)

    # Remove error peaks near ends of signal
    val = val[np.where(val[:, 0] > 200)[0], :]
    val = val[np.where((emg.shape[0] - val[:, 0]) > 200)[0], :]

    # Order peaks based on activity index values
    val = np.flip(val[np.argsort(val[:, 1]), :])

    # Save the timestamps and convert to int
    val = val[:, 1].astype("int")

    return val


def create_dictionary():
    """
    Create the decomposition output dictionary.

    Returns
    -------
    decomposition : dict
        Empty decomposition dictionary
    """
    decomposition = {
        "timeStamps": [],
        "SIL": [],
        "sourceTrains": [],
        "sepMatrix": [],
        "delRepeats": [],
        "numIterations": 0,
        "selfRoA": 0,
        "EMG": [],
        "steps_optimisation": [],
    }

    return decomposition


def sil_func(distance, assign, c):
    """
    Calculates silhouette measure for K-means assessment.
    Parameters
    ----------
    distance : tensor
        Distance between the peaks and the centroids
    assign : tensor
        Indices
    c : tensor
        The chosen centroid

    Returns
    -------
    sil : float
        The silhouette value
    """
    t1 = torch.unsqueeze(scatter_mean(distance[:, 0], assign), 1)
    t2 = torch.unsqueeze(scatter_mean(distance[:, 1], assign), 1)
    segments = torch.cat((t1, t2), 1)
    furthest = segments[1, c].numpy()
    closest = segments[0, c].numpy()

    if closest > furthest:
        furthest = furthest / (closest + 1e-5)
        closest = 1
    elif furthest >= closest:
        closest = closest / (furthest + 1e-5)
        furthest = 1
    sil = np.abs(furthest - closest)

    return sil


def delete_repeats_func(old):
    """
    Uses the RoA to delete sources that occur at the same time.

    Parameters
    ----------
    old : dict
        The decomposition dictionary with the saved variables

    Returns
    -------
    new : dict
        The new decomposition dictionary with the unique timestamps
    """

    new = create_dictionary()
    new["selfRoA"] = old["selfRoA"]
    new["numIterations"] = old["numIterations"]
    new["delRepeats"] = list(np.ones(np.shape(old["selfRoA"])[0]))

    # Find matches in selfRoA and find coefficients of variance
    match = []
    coeff_variation = np.zeros(np.shape(old["selfRoA"])[0])
    for i in range(np.shape(old["selfRoA"])[0]):
        match.append(np.concatenate((np.array([[i]]), np.argwhere(old["selfRoA"][i, :] > 40))))
        if (old["timeStamps"][i]).ndim > 1:
            diff = np.diff(old["timeStamps"][i])
        else:
            diff = old["timeStamps"][i]
        coeff_variation[i] = torch.std(diff.type(torch.FloatTensor)) / torch.mean(diff.type(torch.FloatTensor))

    # Select lowest RoA for keeping
    for i in range(np.shape(old["selfRoA"])[0]):
        new["delRepeats"][match[i][np.argmin(coeff_variation[match[i]]), 0]] = 0

    # Make new decomposition file and only add selected sources
    first = True
    count = 0
    for i in range(np.size(new["delRepeats"])):
        if new["delRepeats"][i] == 0:
            if first is True:
                new["timeStamps"].append(old["timeStamps"][i])
                new["steps_optimisation"].append(old["steps_optimisation"][i])
                new["sepMatrix"] = old["sepMatrix"][[i], :]
                new["sourceTrains"] = old["sourceTrains"][:, [i]]
                new["SIL"].append(old["SIL"][i])
                first = False
            else:
                count = count + 1
                new["timeStamps"].append(old["timeStamps"][i])
                new["steps_optimisation"].append(old["steps_optimisation"][i])
                new["sepMatrix"] = list(np.concatenate((new["sepMatrix"], old["sepMatrix"][[i], :]), axis=0))
                new["sourceTrains"] = list(
                    np.concatenate((new["sourceTrains"], old["sourceTrains"][:, [i]]), axis=1)
                )
                new["SIL"].append(old["SIL"][i])

    return new


class PreProcessing:
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

    def __init__(self, emg, f_samp, notch_freq, factor, intramuscular, high_pass_intra):

        self.emg = emg
        self.f_samp = f_samp
        self.notch_freq = notch_freq
        self.factor = factor
        self.intramuscular = intramuscular
        self.high_pass_intra = high_pass_intra

        self.u_band = None
        self.l_band = None
        self.bandpass = None

    def filter_data(self):
        """
        Filter input data with a butterworth bandpass filter based on set parameters.

        Returns
        -------
        emg : ndarray
            The filtered emg
        """
        if self.intramuscular:
            if self.high_pass_intra is True:
                self.l_band = 60  # Lower bound of bandpass filter
                self.bandpass = False

            else:
                self.bandpass = True
                self.l_band = 60
                self.u_band = 1024

        else:
            self.l_band = 10
            self.u_band = 500
            self.bandpass = True

        nyq = self.f_samp / 2

        if self.notch_freq > self.l_band:
            b, a = iirnotch(self.notch_freq / nyq, 30)
            self.emg = filtfilt(b, a, self.emg + 1e-5, axis=0)

        if self.bandpass:
            # Apply band pass filter
            if self.u_band <= self.l_band:
                raise Exception("Lower bound of bandpass higher than upper bound!")
            if self.u_band > nyq:
                raise Exception("Upper band of bandpass greater than Nyquist.")
            b, a, *_ = butter(5, (self.l_band, self.u_band) / np.array(nyq), btype="bandpass")
            b = np.round(b, 4)
            self.emg = filtfilt(b, a, self.emg.T, axis=0, padtype="odd", padlen=3 * (max(len(b), len(a)) - 1)).T

        else:
            b, a, *_ = butter(5, self.l_band / nyq, btype="high", analog=False)
            self.emg = filtfilt(b, a, self.emg.T).T

        return self.emg

    def extend_data(self):
        """
        Extend the EMG with past values by setting an extension factor. Removes mean value of extended data.

        Returns
        -------
        emg : array
            The extended emg
        """

        factor = self.factor

        n_chans = np.shape(self.emg)[1]
        n_samples = np.shape(self.emg)[0]
        emg = np.zeros((n_samples + factor, n_chans * factor))
        for index in range(factor):
            emg[index: (n_samples + index), index * n_chans: (index + 1) * n_chans] = self.emg

        emg = emg[factor: -(factor + 1), :]
        mean_emg = np.mean(emg, axis=0)
        self.emg = emg - (np.tile(mean_emg, (np.shape(emg)[0], 1)))

        return self.emg

    def whiten_data(self):
        """
        Performs whitening (convolutive sphering) operation using ZCA method, with Singular Value Decomposition (SVD).

        Returns
        -------
        emg : ndarray
            The whitened emg
        """

        emg = np.array(self.emg)

        # remove mean
        emg = emg - np.repeat(np.mean(emg, axis=0, keepdims=True), np.shape(emg)[0], axis=0)

        # calculate covariance matrix
        cov = np.cov(emg.T, rowvar=True)

        # ZCA whitening with SVD
        u, s, v = np.linalg.svd(cov)
        s[s == 0] = 1e-15  # add a small number to avoid division by zero errors

        # calculate whitening transform and apply to signal
        transform = np.dot(u, np.dot(np.diag(1.0 / np.sqrt(s)), v))
        self.emg = np.dot(transform, emg.T).T

        return self.emg

    def preprocess(self):
        """
        Function to perform filtering, extension and whitening.

        Returns
        -------
        emg : ndarray
            The post-processing emg
        """

        self.emg = self.filter_data()
        self.emg = self.extend_data()
        self.emg = self.whiten_data()

        return self.emg


class GradDescent:
    """
    This class provides the logic flow for the gCKC algorithm.
    """

    def __init__(
            self,
            emg,
            act,
            n,
            m,
            f_samp,
            source_def,
            cut_off_sil,
            factor,
            est_muap,
            tolerance_roa,
            num_iterations,
            func,
            clear_activations,
            delete_repeats,
            console,
            screen
    ):
        """
        Parameters
        ----------
        emg : ndarray
            The post-processed emg
        act : array
            The timestamps with highest activity
        n : int
            The number of channels
        m : int
            The number of observations
        f_samp : int
            The sampling frequency of the recording
        source_def : bool
            True if it uses source deflation
        cut_off_sil : float
            Threshold of SIL to accept newly found source
        factor : int
            The extension factor
        est_muap : float
            Estimated time support of muaps for peel off approach
        tolerance_roa : int
            Tolerance to calculate rate of agreement between timestamps
        num_iterations : int
            Number of iterations of gCKC to run
        func : int
            The contrast function to use
        clear_activations : bool
            True if user wants to clear activations in initialisation library 5ms either side
        delete_repeats : bool
            True if user wants to delete sources that occur at the same time
        console : view object
            The console to direct readback data to
        screen : view object
            The screen to direct graphical info to

        """

        self.emg = emg
        self.act = act
        self.n = n
        self.m = m
        self.f_samp = f_samp
        self.source_def = source_def
        self.cut_off_sil = cut_off_sil
        self.factor = factor
        self.est_muap = est_muap
        self.tolerance_roa = tolerance_roa
        self.num_iterations = num_iterations
        self.func = func
        self.clear_activations = clear_activations
        self.delete_repeats = delete_repeats
        
        self.console = console
        self.screen = screen

        self.i = None
        self.weights = None
        self.library = None
        self.sil = None
        self.source = None
        self.end_run = None
        self.sep_matrix = None
        self.step_saved = None
        

    def initialise(self):
        """
        Passes the next largest activation for initialisation then deletes the activation from the list.

        Returns
        -------
        weights : array
            The weights vector, selected from the emg at time of highest activation
        """
        if self.act.size == 0:
            weights = self.emg[0, :]
            self.end_run = True
        elif self.act.size == 1:
            weights = self.emg[int(self.act), :]
            self.act = np.empty(0)
        else:
            weights = self.emg[self.act[0], :]
            self.act = self.act[1:]

        self.weights = torch.unsqueeze(weights, 1)

        return self.weights

    def estimate_source(self):
        """
        Weights the Mahalanobis distance with the whitened emg matrix
        Follows with z-scoring for ease of use.

        Returns
        -------
        source : union[array, ndarray]
            The estimated source
        """
        source = torch.tensordot(torch.transpose(self.weights, 0, 1), torch.transpose(self.emg, 0, 1), 1)
        self.source = source / source.std(-1, unbiased=False)
        
        # self.screen.update_plot(self.source.detach().cpu().numpy())

        return self.source

    def contrast(self, x):
        """
        Selection of sparsity estimator's grads (g = G')

        Parameters
        ----------
        x : tensor
            The data to calculate the sparsity on (the source)

        Returns
        -------
        x : tensor
            The data passed through the sparsity estimator
        """
        function = self.func
        if function == 1:
            x = 1 / 2 * torch.square(x)
        elif function == 2:
            x = torch.tanh(x)
        elif function == 3:
            x = torch.exp(torch.tensor(-(x ** 2 / 2))) * torch.tensor(x)
        elif function == 4:
            x = torch.log(1 + x ** 2)
        elif function == 5:
            x = x ** 3

        return x

    def loss_func(self):
        """
        Applies a sparsity estimator to find loss for gradient descent.

        Returns
        -------
        loss : tensor
            The loss term
        """
        source = self.estimate_source()
        loss = self.contrast(source)

        # Zero edge effects
        fudge = torch.zeros_like(loss)[:, :100]
        loss = torch.cat([fudge, loss[:, 100:-100], fudge], 1)
        loss = torch.transpose(torch.tensordot(loss, self.emg, 1) / self.emg.shape[0], 0, 1)

        return loss

    def source_deflation(self):
        """
        Gram–Schmidt orthogonalization step to increase the number of unique
        sources that are estimated.
        Implemented to overcome the problem of convergence to the same source.

        Returns
        -------
        weights: ndarray
            The orthogonalised weights
        """

        weights = self.weights - torch.matmul(self.sep_matrix, (torch.matmul(torch.transpose(self.sep_matrix, 0, 1), self.weights)))
        self.weights = weights / torch.norm(weights, p=2, dim=0)

        return self.weights

    def peel_off_func(self):
        """
        Remove found source templates from emg.

        Returns
        -------
        emg : tensor
            The emg with the newly found source removed
        """

        library = np.squeeze(self.library)

        spikes = np.zeros((self.emg.shape[0], 1))
        spikes[library, :] = 1
        spikes = torch.tensor(spikes, dtype=torch.float64)

        part1 = torch.matmul(torch.transpose(spikes, 1, 0), spikes) ** -1
        for roll in range(int(-2 * self.est_muap), int(2 * self.est_muap)):
            part2 = torch.matmul(torch.transpose(torch.roll(spikes, roll, 0), 1, 0), self.emg)
            muap = part1 * part2
            self.emg = self.emg - torch.matmul(torch.roll(spikes, roll, 0), muap)

        return self.emg

    def update_weights(self, loss, momentum, step):
        # todo: add a couple of different updates for the user to choose from
        """
        Updates weights using step-adjusted momentum.

        Parameters
        ----------
        loss : ndarray
            The loss term
        momentum :  float
            The momentum term
        step : int
            The step in the optimisation

        Returns
        -------
        weights : array
            The updated weights vector
        momentum: int
            The updated momentum term
        """
        weights = self.weights
        alpha = 0.001
        beta = 0.9
        momentum = (beta * momentum) + ((1 - beta) * loss)
        m_bc = momentum / (1 - beta ** step)

        weights = weights + alpha * m_bc
        self.weights = weights/torch.norm(weights, p=2, dim=0)

        return self.weights, momentum

    def gradient_descent(self):
        """
        Uses gradient descent to maximise sparsity of source vector
        Stops optimisation when the new weights converge to the previous weights. Saves number of step updates.

        Returns
        -------
        weights : array
            The optimised weights vector
        """

        mom = torch.zeros_like(self.weights, dtype=torch.float64)

        for step in range(1, 501):
            prev_weights = self.weights
            loss = self.loss_func()

            self.weights, mom = self.update_weights(loss, mom, step)

            # Gram-Schmidt orthogonalization step
            if self.source_def:
                self.weights = self.source_deflation()

            diff = torch.abs(torch.tensordot(torch.transpose(self.weights, 1, 0), prev_weights, 1) - 1)
            if diff < 1e-6:
                self.step_saved = step
                break

        if self.step_saved is None:
            self.step_saved = 500

        return self.weights

    def k_means(self):
        """
        K means clustering algorithm (K = 2).

        Returns
        -------
        library : array
            Activations selected from optimised centroids
        sil : float
            The silhouette value
        """
        source = self.source.T
        distance = 2 * self.est_muap

        # Find peaks in source
        locs, heights = find_peaks(source[:, 0], height=-100, distance=distance)
        locs = torch.tensor(locs)
        heights = torch.unsqueeze(torch.tensor(heights["peak_heights"]), 1)
        heights = heights.type(torch.FloatTensor)
        distance = []
        assign = []

        # Use values of peaks to optimise centroid locations
        centroids = np.array([0, 0.5])
        centroids = torch.tensor(centroids, dtype=torch.float64)
        for i in range(100):
            centroids = torch.unsqueeze(centroids, 0)
            centroids = centroids.repeat(heights.shape[0], 1)
            distance = torch.abs(heights.repeat_interleave(2, axis=1) - centroids)
            assign = torch.argmin(distance, dim=1)
            centroids = scatter_mean(heights[:, 0], assign)

        # Use optimised centroids to select activations
        c = torch.argmax(centroids)
        select = torch.where(torch.argmin(distance, 1) == c)
        library = torch.squeeze(locs[select])
        sil = sil_func(distance, assign, c)

        return library, sil

    def optimise_weights(self):
        """
        Runs repeated iterations of K means clustering and updates weights with each iteration.

        Returns
        -------
        weights : array
            The updated weight vector
        library : array
            The selected timestamps
        sil : float
            The silhouette measure
        source : array
            The newly found source
        """

        sil = 1
        prev_sil = 2

        for i in range(200):
            if np.abs(sil - prev_sil) < 0.00001:
                break
            prev_sil = sil

            # k-means optimisation
            self.library, sil = self.k_means()
            self.weights = torch.mean(self.emg[self.library, :], dim=0)

            if self.weights.size() == 0:
                break
            else:
                self.weights = torch.unsqueeze(self.weights, 1)
                self.weights = self.weights/torch.norm(self.weights, p=2, dim=0)
                self.source = self.estimate_source()

        return self.weights, self.library, sil, self.source

    def new_source(self):
        """
        Initialises weights on activity index list and then optimises,
        first with gCKC and then K Means Clustering algorithm.

        Returns
        -------
        weights : array
            The updated weight vector
        library : array
            The selected timestamps
        sil : float
            The silhouette measure
        source : array
            The newly found source

        """

        self.weights = self.initialise()
        self.library, self.sil = 0, 0

        if self.end_run is False:
            self.weights = self.gradient_descent()
            self.weights, self.library, self.sil, self.source = self.optimise_weights()

        return self.weights, self.library, self.sil, self.source

    def rate_of_agreement_finder(self, library):
        """
        Calculates the rate of agreement between activation time stamps.

        Parameters
        ----------
        library : union[list, ndarray]
            The newly computed source timestamps

        Returns
        -------
        roa : rate of agreement between timestamps
        """

        if isinstance(library, np.ndarray) and library.size == 0:
            roa = 0

        else:
            if isinstance(library, np.ndarray):
                range_row = 1
                range_col = 1
            else:
                range_row = len(library)
                range_col = len(library)

            roa = np.zeros([range_row, range_col])
            for row in range(range_row):
                for col in range(range_col):
                    maximum = 0
                    for shift in range(int(-self.est_muap), int(self.est_muap)):
                        true_pos = 0
                        for tol in range(-self.tolerance_roa, self.tolerance_roa):
                            true_pos = true_pos + np.size(np.intersect1d(library[row], library[col] + tol + shift))
                        false_pos = library[row].size()[0] - true_pos
                        false_neg = library[col].size()[0] - true_pos
                        if (true_pos + false_pos + false_neg) > 0:
                            val = np.round(((100 * true_pos) / (true_pos + false_pos + false_neg)), 1)
                        else:
                            val = 0
                        if row == col:
                            val = 0
                        if val > maximum:
                            maximum = val
                    roa[row, col] = maximum

        return roa

    def clear_act(self):
        """
        Clear activations in initialisation library 5ms either side.
        """

        for i in range(-self.tolerance_roa, self.tolerance_roa):
            novel = np.in1d(self.act, (self.library + i))
            self.act = self.act[np.invert(novel)]
        if self.act.size == 0:
            self.end_run = True

    def source_separate(self):
        """
        Iterates gCKC for each new source and then adds accepted sources to the separation matrix.
        """

        emg = self.emg

        # Initialise the empty separation matrix and sources
        self.sep_matrix = np.zeros([emg.shape[1], 1])
        self.sep_matrix = torch.tensor(self.sep_matrix)
        all_sources = np.zeros([emg.shape[0], 1])

        # Initialise a decomposition output library
        decomposition = create_dictionary()

        # Run source separation operation
        first = True
        self.end_run = False
        for i in range(self.num_iterations):
            self.i = i
            if self.end_run is True:
                break

            self.weights, self.library, self.sil, self.source = self.new_source()

            decomposition["numIterations"] = decomposition["numIterations"] + 1
            decomposition["steps_optimisation"].append(self.step_saved)

            input_text = 'Iteration ' + str(i + 1) + ' finished with SIL of ' + str(self.sil) + '.'
            print(input_text)
            self.console.message(input_text)
            if self.sil > self.cut_off_sil:
                decomposition["timeStamps"].append(self.library)
                if first is True:
                    roa_timestamps = 0
                else:
                    roa_timestamps = np.max(self.rate_of_agreement_finder(decomposition["timeStamps"]))
                if roa_timestamps > 30:
                    del decomposition["timeStamps"][-1]
                else:
                    if self.clear_activations:
                        self.clear_act()
                    if self.source_def is not True:
                        self.emg = self.peel_off_func()
                    decomposition["SIL"].append(self.sil)
                    self.sep_matrix = torch.cat([self.sep_matrix, self.weights], 1)
                    all_sources = np.concatenate((all_sources, self.source.numpy().T), 1)
                    if first is True:
                        first = False
                        self.sep_matrix = self.sep_matrix[:, 1:]
                        all_sources = all_sources[:, 1:]

        # Update separation matrix and apply to emg
        self.sep_matrix = torch.tensordot(torch.transpose(self.sep_matrix, 1, 0), torch.transpose(self.emg, 1, 0), 1)
        decomposition["sepMatrix"] = self.sep_matrix.numpy()
        decomposition["sourceTrains"] = all_sources

        # Find repeated sources with RoA measure and optionally delete
        if len(decomposition["timeStamps"]) > 1:
            decomposition["selfRoA"] = self.rate_of_agreement_finder(decomposition["timeStamps"])
            if self.delete_repeats:
                decomposition = delete_repeats_func(decomposition)

        return decomposition
