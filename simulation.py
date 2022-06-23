import numpy as np
import random
from scipy.signal import resample, welch


def add_noise(emg_samples, snr):
    """
    Calculates and adds gaussian noise of desired SNR to the simulated emg.
    """

    data = np.transpose(emg_samples)
    noisy_emg_samples = np.zeros(np.shape(emg_samples))
    for i in range(np.shape(data)[0]):
        sig_welch = welch(data[i, :], 2048)
        sig_power = np.sum(sig_welch[1]) * (sig_welch[0][1] - sig_welch[0][0])
        noise_power_target = sig_power / (10 ** (snr / 10))
        noise = np.random.normal(0, np.sqrt(noise_power_target), np.shape(data)[1])
        noise_welch = welch(noise, 2048)
        noise_power = np.sum(noise_welch[1]) * (noise_welch[0][1] - noise_welch[0][0])
        real_snr = 10 * np.log10(sig_power / noise_power)
        if (real_snr - snr) > 1:
            print("SNR warning")
            print(real_snr)
        noisy_emg_samples[:, i] = emg_samples[:, i] + np.transpose(noise)
    return noisy_emg_samples


class GenerateMuapTrain:
    """
    This class uses templates from Silvia's model to build emg from MUAP trains.
    The convolution function used causes edge effects, so to avoid this the muap train simulated is longer than
    specified by sample_len, and then it is cut to the desired length.
    """

    def __init__(self, sample_time, num_mu):  # number of samples of length sample_len

        self.sampleLen = int(sample_time * 2048) + 200  # length of sample (samp Freq 4026Hz)
        self.numMU = num_mu  # number of motor units in sim
        self.lambdas = np.random.randint(10, 15, num_mu) / 2048  # firing rate of motor units
        self.mu_train = None
        self.emg_samples = None
        self.muap_trains = None
        self.all_muaps = None

        # load simulated MUAP templates:
        templates = np.load("templates.npy")[1000:, :, :, :]
        locations = np.load("fibreLocs.npy")[1000:, :]

        distance = np.sqrt(locations[:, 0] ** 2 + locations[:, 1] ** 2)
        templates = templates[np.argsort(distance), :, :, :]
        templates = templates[1 : (num_mu + 1), :, :, :]
        self.num_sensors = np.shape(templates)[1] * np.shape(templates)[2]
        templates = np.transpose(templates, (0, 2, 1, 3))
        reshape_templates = np.zeros([np.shape(templates)[0], self.num_sensors, np.shape(templates)[3]])
        count = 0
        for i in range(np.shape(templates)[1]):
            for j in range(np.shape(templates)[2]):
                reshape_templates[:, count, :] = templates[:, i, j, :]
                count += 1
        self.templates = reshape_templates
        self.temp_max = np.trapz(np.abs(self.templates), axis=1)

    def next_sequence(self, lam):
        """
        Finds the timestamps of MU activations (using a poisson distribution).
        """

        time_stamps = np.zeros(1, dtype=int)
        while time_stamps[-1] <= self.sampleLen:
            next_spike = int(random.expovariate(lam))
            while next_spike < 70 or next_spike > 500:
                next_spike = int(random.expovariate(lam))
            time_stamps = np.append(time_stamps, time_stamps[-1] + next_spike)
        return time_stamps[1:-1]

    def spike_train(self):
        """
        Converts the timestamps into a binary time series MUAP train
        """

        spikes = np.zeros((self.numMU, self.sampleLen))
        for i, j in enumerate(self.lambdas):
            spikes[i, self.next_sequence(j) - 1] = 1
        return spikes

    def gen_spikes(self):
        """
        Uses the spike_train function to build an array of MU activation trains of size and shape specified on
        initialisation.
        """

        mu_train = np.transpose(self.spike_train())
        self.mu_train = mu_train

    def init_sim(self):
        """
        This function builds the emg from the MU activation trains by convolving the binary MU trains with the templates
        """

        self.gen_spikes()
        emg_samples = np.zeros([self.sampleLen, self.num_sensors])

        for k in range(self.num_sensors):
            emg = np.zeros([self.numMU, self.sampleLen])
            all_templates = np.zeros([self.numMU, np.shape(self.templates)[2]])
            for j in range(self.numMU):
                train = np.squeeze(self.mu_train[:, j])
                template = np.squeeze(self.templates[j, k, :])
                template = resample(template, 128)
                all_templates[j, :] = template
                emg[j, :] = np.convolve(train, template, "same")
            self.all_muaps = all_templates
            self.muap_trains = emg
            emg = np.sum(emg, axis=0, keepdims=True)
            emg_samples[:, k] = np.squeeze(np.transpose(emg))
        self.mu_train = self.mu_train[100:-100, :]
        self.emg_samples = emg_samples[100:-100, :]

    def return_output(self):
        """
        Returns the MU spike trains.
        """

        return self.mu_train

    def return_input(self, snr):
        """
        Returns the noisy emg, i.e. the MUAP trains with SNR as defined by the parameter.
        """

        return add_noise(self.emg_samples, snr)
