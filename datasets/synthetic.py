import datasets
import numpy as np

class EightGaussians:

    class Data:

        def __init__(self, data):
            self.x = data.astype(np.float32)
            self.N = self.x.shape[0]

    def __init__(self):
        file = datasets.root + 'synthetic/8gaussians.npy'
        trn, val, tst = load_data_normalised(file)

        self.trn = self.Data(trn)
        self.val = self.Data(val)
        self.tst = self.Data(tst)

        self.n_dims = self.trn.x.shape[1]


class Checkerboard:

    class Data:

        def __init__(self, data):
            self.x = data.astype(np.float32)
            self.N = self.x.shape[0]

    def __init__(self):

        file = datasets.root + 'synthetic/checkerboard.npy'
        trn, val, tst = load_data_normalised(file)

        self.trn = self.Data(trn)
        self.val = self.Data(val)
        self.tst = self.Data(tst)

        self.n_dims = self.trn.x.shape[1]


class TwoSpirals:

    class Data:

        def __init__(self, data):
            self.x = data.astype(np.float32)
            self.N = self.x.shape[0]

    def __init__(self):
        file = datasets.root + 'synthetic/2spirals.npy'
        trn, val, tst = load_data_normalised(file)

        self.trn = self.Data(trn)
        self.val = self.Data(val)
        self.tst = self.Data(tst)

        self.n_dims = self.trn.x.shape[1]


def load_data(root_path):
    rng = np.random.RandomState(seed=42)
    data = np.load(root_path)
    rng.shuffle(data)
    n_train = int((1/3) * data.shape[0])
    n_val = int((1/3) * data.shape[0])

    data_train = data[0:n_train]
    data_val = data[n_train:n_train+n_val]
    data_test = data[n_train+n_val:]

    return data_train, data_val, data_test


def load_data_normalised(root_path):
    data_train, data_val, data_test = load_data(root_path)
    
    mu = data_train.mean(axis=0)
    s = data_train.std(axis=0)

    data_train = (data_train - mu) / s
    data_val = (data_val - mu) / s
    data_test = (data_test - mu) / s

    return data_train, data_val, data_test