import datasets
import numpy as np

seed = 42
rng = np.random.RandomState(seed=seed)

class MNIST_16x16():
    class Data():
        def __init__(self, data):
            self.x = data.astype(np.float32)
            self.N = self.x.shape[0]

    def __init__(self):
        file = datasets.root + 'mnist/MNIST_16x16.npy'

        trn, val, tst = load_data_normalised(file)

        self.trn = self.Data(trn)
        self.val = self.Data(val)
        self.tst = self.Data(tst)

        self.n_dims = self.trn.x.shape[1]

def load_data(root_path):
    data = np.load(root_path)
    rng.shuffle(data)
    n_train = int((1/3) * data.shape[0])
    n_val = int((1/3) * data.shape[0])

    data_train = data[0:n_train]
    data_val = data[n_train:n_train+n_val]
    data_test = data[n_train+n_val:]

    data_train = data_train.reshape((data_train.shape[0], -1))
    data_val = data_val.reshape((data_val.shape[0], -1))
    data_test = data_test.reshape((data_test.shape[0], -1))

    return data_train, data_val, data_test


def load_data_normalised(root_path):
    data_train, data_val, data_test = load_data(root_path)

    mu = data_train.mean(axis=0)
    s = np.add(0.000001, data_train.std(axis=0))

    data_train = (data_train - mu) / s
    data_val = (data_val - mu) / s
    data_test = (data_test - mu) / s

    return data_train, data_val, data_test

if __name__ == '__main__':
    data = MNIST_16x16()