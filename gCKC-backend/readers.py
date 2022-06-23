def get_mat(arguments):
    """
    Extracts the EMG (input shape = channels x samples) and sampling frequency
    from the MAT file.
    Returns the EMG in array format, output shape = samples x channels
    """
    import numpy as np
    import scipy.io as sio
    import sys

    if len(arguments) != 2:
        raise Exception("GetMat needs two arguments, path and filename")
    path = arguments[0]
    filename = arguments[1]
    if filename[-3:] != "mat":
        raise Exception("File should end .mat, is this a matlab file?")
    if sys.platform.startswith("linux"):
        if path[-1] != "/":
            path += "/"
    elif sys.platform.startswith("win"):
        if path[-1] != "\\":
            path += "\\"

    file = path + filename
    mat = sio.loadmat(file)

    emg = np.array(mat["Signal"]).T
    f_samp = 10240

    return emg, f_samp


def get_mat_muscle(arguments):
    """
    Extracts the EMG (input shape = channels x samples) and sampling frequency
    from the MAT file.
    Returns the EMG in array format, output shape = samples x channels
    """
    import numpy as np
    import scipy.io as sio
    import sys

    if len(arguments) != 2:
        raise Exception("GetMat needs two arguments, path and filename")
    path = arguments[0]
    filename = arguments[1]
    if filename[-3:] != "mat":
        raise Exception("File should end .mat, is this a matlab file?")
    if sys.platform.startswith("linux"):
        if path[-1] != "/":
            path += "/"
    elif sys.platform.startswith("win"):
        if path[-1] != "\\":
            path += "\\"

    file = path + filename
    mat = sio.loadmat(file)

    emg = np.array(mat["EMG"]).T
    f_samp = 2048
    emg_muscle = emg[:, 0:65]

    return emg_muscle, f_samp
