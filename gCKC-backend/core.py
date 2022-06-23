import numpy as np
import helpers as h
from itertools import product, groupby
import readers as r
import torch
from multiprocessing import Pool
import toolz
import os
import pandas as pd
import pickle as pkl
import time


def get_ground_truth_ann(path, f_samp, t1, t2):
    df = np.array(pd.read_csv(path, sep='^([^\s]*)\s', engine='python', header=None).drop(0, axis=1))
    df[:, 0] = df[:, 0] * f_samp
    if t2 == -1:
        df = df[(df[:, 0] > t1 * f_samp)]
    else:
        df = df[(df[:, 0] > t1 * f_samp) & (df[:, 0] < t2 * f_samp)]
    # df[:, 0] = (df[:, 0]) #.astype(int)
    sorted_spikes = df[np.argsort(df[:, 1])]
    num_units = int(max(sorted_spikes[:, 1]))
    ground_truth_mutrain_all = []

    for i in range(1, num_units + 1):
        mu = np.transpose(np.expand_dims(sorted_spikes[np.isin(sorted_spikes[:, 1], i)][:, 0], axis=1))
        ground_truth_mutrain_all.append(np.sort(mu))

    return list(ground_truth_mutrain_all), num_units


def rate_of_accuracy_tol05(pars):
    x, y = pars
    runs = map(rate_of_agreement05, zip(x, y))

    return max(runs, key=lambda x: x[0])


def rate_of_accuracy_tol5(pars):
    x, y = pars
    runs = map(rate_of_agreement5, zip(x, y))

    return max(runs, key=lambda x: x[0])


def rate_of_agreement5(pars):
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
    """

    ground_truth, timestamps = pars
    ground_truth = ground_truth.astype(int)
    tol_range = 50
    true_pos = 0
    best_tol = 0
    false_pos = 0
    false_neg = 0
    best_shift = 0
    shift_range = 100

    timestamps = np.expand_dims(timestamps, axis=0)
    RoA = 0
    maximum = 0
    for shift in range(-shift_range, shift_range):
        true_pos = 0
        for tol in range(-tol_range, tol_range + 1):
            true_pos = true_pos + np.size(np.intersect1d(timestamps[0], ground_truth[0] + tol + shift))

            false_pos = np.size(ground_truth[0]) - true_pos
            false_neg = np.size(timestamps[0]) - true_pos
            val = np.round(((100 * true_pos) / (true_pos + false_pos + false_neg)), 1)
            if val > maximum:
                maximum = val
                best_shift = shift
                best_tol = tol
            RoA = maximum

    return RoA, int(true_pos), int(false_pos), int(false_neg), int(best_shift), int(best_tol), ground_truth, timestamps


def rate_of_agreement05(pars):
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
    """

    ground_truth, timestamps = pars
    ground_truth = ground_truth.astype(int)
    tol_range = 5
    true_pos = 0
    false_pos = 0
    false_neg = 0
    best_shift = 0
    best_tol = 0
    shift_range = 100

    timestamps = np.expand_dims(timestamps, axis=0)
    RoA = 0
    maximum = 0
    for shift in range(-shift_range, shift_range):
        true_pos = 0
        for tol in range(-tol_range, tol_range + 1):
            true_pos = true_pos + np.size(np.intersect1d(timestamps[0], ground_truth[0] + tol + shift))

            false_pos = np.size(ground_truth[0]) - true_pos
            false_neg = np.size(timestamps[0]) - true_pos
            val = np.round(((100 * true_pos) / (true_pos + false_pos + false_neg)), 1)
            if val > maximum:
                maximum = val
                best_shift = shift
                best_tol = tol
            RoA = maximum

    return RoA, int(true_pos), int(false_pos), int(false_neg), int(best_shift), int(best_tol), ground_truth, timestamps


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

    def __init__(self, reader, num_iterations=30, cut_off_sil=0.85, intramuscular=True, high_pass_intra=True,
                 source_def=True, delete_repeats=True):

        super().__init__()
        self.reader = reader

        # Data parameters
        self.original_emg = None
        self.emg = None
        self.m = None  # number of observations
        self.n = None  # number of channels
        self.f_samp = None

        # Initialised parameters
        self.num_iterations = num_iterations
        self.cut_off_sil = cut_off_sil
        self.func = None
        self.factor = None  # extension factor
        self.est_muap = None  # estimated time support of muaps
        self.act = None  # activity index vector
        self.new_act = None  # activation after decomposition
        self.tolerance_roa = None  # rate of agreement tolerance
        self.opt_func = None
        self.cost_func = None
        self.start = None
        self.end = None

        # Pre processing
        self.prep_marker = None
        self.load_marker = None
        self.intramuscular = intramuscular  # used to decide on filtering
        self.high_pass_intra = high_pass_intra  # just high pass intramuscular (no high frequency cut off)
        self.notch_freq = 50  # Notch band-stop frequency
        self.extension_factor = 1000

        # Other settings
        self.source_def = source_def
        self.clear_activations = True
        self.delete_repeats = delete_repeats
        self.images = True

    def load_data(self, arguments):
        """
        Loading parameters. Check the data shape is correct.
        """
        path, name, self.start, self.end = arguments
        self.emg, self.f_samp = self.reader((path, name))
        self.emg = h.check_data(self.emg, self.f_samp)

        if self.end == -1:
            self.emg = self.emg[self.start * self.f_samp:self.end, :]
        else:
            self.emg = self.emg[self.start * f_samp:self.end * f_samp, :]
        self.factor = int(np.ceil(self.extension_factor / np.shape(self.emg)[1]))
        self.est_muap = int(self.f_samp * 15 / 1000)
        self.tolerance_roa = int(self.f_samp * 5 / 1000)
        self.load_marker = True
        self.prep_marker = False

    def decompose(self, opt_func, cost_func):
        """
        Runs the full decomposition pipeline of gCKC, including preprocessing steps (filtering, extension and whitening)
        """

        self.opt_func = opt_func
        self.func = cost_func
        self.start = start
        self.end = end

        if self.load_marker is False:
            raise Exception("Please load data before decomposition.")

        # Save original emg
        keep = self.emg

        # Check if preprocessing was done. Run if not.
        if self.prep_marker is False:
            pre_processing = h.PreProcessing(
                self.emg, self.f_samp, self.notch_freq, self.factor, self.intramuscular, self.high_pass_intra
            )
            self.emg, self.original_emg = pre_processing.preprocess()
            print('Pre-processing: filter, extend and whiten.')
            self.prep_marker = True

        # Retrieve number of observations (m) and channels (n)
        self.m = np.shape(self.emg)[0]
        self.n = np.shape(self.emg)[1]
        self.act = h.activity_index(self.emg)
        self.emg = torch.tensor(self.emg)

        gradient_descent = h.GradDescent(
            self.original_emg,
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
            self.opt_func,
            self.intramuscular
        )

        print('Starting decomposition')
        decomp, self.new_act = gradient_descent.source_separate()
        decomp["EMG"] = keep

        return decomp

    def peel_off_func(self, all_timestamps, iterations, source_def, opt_func, func, sil_cut):
        """
        Remove found source templates from emg.

        Returns
        -------
        emg : tensor
            The emg with the newly found source removed
        """

        self.num_iterations = iterations
        self.source_def = source_def
        self.opt_func = opt_func
        self.func = func
        self.cut_off_sil = sil_cut
        original_emg2 = self.original_emg

        for i in range(len(all_timestamps)):
            library = np.squeeze(all_timestamps[i])
            spikes = np.zeros((original_emg2.shape[0], 1))
            spikes[library, :] = 1
            spikes = torch.tensor(spikes, dtype=torch.float64)

            sta = h.spike_triggered_averaging(original_emg2, library, self.f_samp, self.intramuscular)
            timestamps = spikes.unsqueeze(0).unsqueeze(0).squeeze(-1)
            sta_reshape = torch.tensor(sta).unsqueeze(1)
            conv_out = torch.nn.functional.conv1d(timestamps, sta_reshape.flip(2), padding='same')

            arr_conv_out = np.array(conv_out.squeeze(0)).T
            original_emg2 = original_emg2[0:len(arr_conv_out)] - arr_conv_out

        pre_processing = h.PreProcessing(original_emg2, self.f_samp, 50, self.factor, True, True)
        original_emg2_whiten = pre_processing.whiten_data()
        self.emg = torch.tensor(original_emg2_whiten)
        self.act = h.activity_index(self.emg)

        gradient_descent = h.GradDescent(
            self.emg,
            self.emg,
            self.act,  # self.new_act
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
            self.opt_func,
            self.intramuscular
        )

        print('Starting decomposition')
        # todo: check self.new_act
        # todo: give it the decomposition
        decomp, self.new_act = gradient_descent.source_separate()
        decomp["EMG"] = original_emg2

        return decomp


if __name__ == "__main__":

    f_samp = 10240
    start = 20
    end = 40
    num_iter = 150
    num_iter2 = 40
    type_removal = 'pf_20-40'
    type_removal2 = 'pf_pf_20-40'
    optimiser = 16
    cost = 3
    cut_off1 = 0.85
    cut_off2 = 0.85
    mvc = '20'

    filepath = "C:/Users/agnes/OneDrive - Imperial College London/Documents/AI4Health/Data/TA/" + mvc + "/"
    filename = "/Signal.mat"
    ground_truth_mutrain, num_mu = get_ground_truth_ann(os.path.join(filepath, 'el_ta' + mvc + ".ann"), f_samp, start,
        end)

    run = RunCKC(r.get_mat, num_iterations=num_iter, cut_off_sil=cut_off1, intramuscular=True, high_pass_intra=True,
                 source_def=False, delete_repeats=True)
    run.load_data((filepath, filename, start, end))
    start_time = time.time()
    decomposition = run.decompose(optimiser, cost)
    end_time = time.time()

    tot_time = end_time - start_time
    print('Time taken: %d' % tot_time)

    timeStamps = [x + start * f_samp for x in decomposition['timeStamps']]

    with Pool() as po:
        runs_5 = po.map(rate_of_agreement5, product(ground_truth_mutrain, timeStamps))
        runs_05 = po.map(rate_of_agreement05, product(ground_truth_mutrain, timeStamps))

    roa_05 = []
    roa_5 = []
    for i in range(0, len(timeStamps)):
        all_t = np.expand_dims(timeStamps[i], 0)
        each_roa05 = ([item for item in runs_05 if all_t in item[7][0]])
        each_roa5 = ([item for item in runs_5 if all_t in item[7][0]])

        max_roa05 = sorted(each_roa05, key=lambda tup: tup[0], reverse=True)[0]
        max_roa5 = sorted(each_roa5, key=lambda tup: tup[0], reverse=True)[0]
        roa_05.append(max_roa05)
        roa_5.append(max_roa5)

    path_changing = os.path.join('_' + mvc + '_' + type_removal + '_' + str(num_iter) + 'it' + '_opt' + str(optimiser) +
                                 '_cost' + str(cost) + '_sil' + str(cut_off1) + '.pkl')

    with open(os.path.join('data/decomposition' + path_changing), 'wb') as f:
        pkl.dump(decomposition, f)

    with open(os.path.join('data/timestamps' + path_changing), 'wb') as f:
        pkl.dump(timeStamps, f)

    with open(os.path.join('data/roa5' + path_changing), 'wb') as f:
        pkl.dump(roa_5, f)

    with open(os.path.join('data/roa05' + path_changing), 'wb') as f:
        pkl.dump(roa_05, f)

    # # FIRST PEEL OFF
    # start_time = time.time()
    # decomposition_afterpeeloff = run.peel_off_func(decomposition['timeStamps'], num_iter2, False, optimiser, cost, cut_off2)
    # end_time = time.time()
    #
    # tot_time = end_time - start_time
    # print('Time taken: %d' % tot_time)
    #
    # timeStamps_afterpeeloff = [x + start * f_samp for x in decomposition_afterpeeloff['timeStamps']]
    #
    # with Pool() as po:
    #     runs_5_afterpeeloff = po.map(rate_of_agreement5, product(ground_truth_mutrain, timeStamps_afterpeeloff))
    #     runs_05_afterpeeloff = po.map(rate_of_agreement05, product(ground_truth_mutrain, timeStamps_afterpeeloff))
    #
    # roa_05_afterpeeloff = []
    # roa_5_afterpeeloff = []
    # for i in range(0, len(timeStamps)):
    #     all_t = np.expand_dims(timeStamps[i], 0)
    #     each_roa05 = ([item for item in runs_05 if all_t in item[7][0]])
    #     each_roa5 = ([item for item in runs_5 if all_t in item[7][0]])
    #
    #     max_roa05 = sorted(each_roa05, key=lambda tup: tup[0], reverse=True)[0]
    #     max_roa5 = sorted(each_roa5, key=lambda tup: tup[0], reverse=True)[0]
    #     roa_05_afterpeeloff.append(max_roa05)
    #     roa_5_afterpeeloff.append(max_roa5)
    #
    # path_changing2 = os.path.join('_'+ mvc + '_' + type_removal2 + '_' + str(num_iter2) + '_it' + '_opt' + str(optimiser) +
    #                              '_cost' + str(cost) + '_sil' + str(cut_off2) + '.pkl')
    #
    # with open(os.path.join('data/decomposition' + path_changing2), 'wb') as f:
    #     pkl.dump(decomposition_afterpeeloff, f)
    #
    # with open(os.path.join('data/timestamps' + path_changing2), 'wb') as f:
    #     pkl.dump(timeStamps_afterpeeloff, f)
    #
    # with open(os.path.join('data/roa5' + path_changing2), 'wb') as f:
    #     pkl.dump(roa_5_afterpeeloff, f)
    #
    # with open(os.path.join('data/roa05' + path_changing2), 'wb') as f:
    #     pkl.dump(roa_05_afterpeeloff, f)
    #
    # # # # prev_emg, f_samp = r.get_mat((filepath, filename))
    # # # # next_emg = prev_emg[20*f_samp:25*f_samp, :]
    # # # # sources = np.linalg.pinv(decomposition['sepMatrix'])*next_emg
