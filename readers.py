def get_mat(path):
    import torch
    import scipy.io as sio

    mat = sio.loadmat(path)

    data = {}
    data['emg'] = torch.tensor(mat["EMG"])
    data['sampling_frequency'] = 2048

    return data