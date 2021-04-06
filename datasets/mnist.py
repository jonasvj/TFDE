import datasets
import numpy as np
import random
from utils import order_variables_partial_correlation

seed = 42
rng = np.random.RandomState(seed=seed)

class MNIST():
    class Data():
        def __init__(self, data):
            self.x = data.astype(np.float32)
            self.N = self.x.shape[0]

    def __init__(self, optimal_order):
        trn, val, tst, self.ordering = load_data_normalised(self.file, optimal_order)

        self.trn = self.Data(trn)
        self.val = self.Data(val)
        self.tst = self.Data(tst)

        self.n_dims = self.trn.x.shape[1]

    def convert_to_image(self, generated_data):
        n_sq = int(np.sqrt(self.n_dims))
        n_images = generated_data.shape[0]
        output_images = [np.zeros((n_sq, n_sq)) for _ in range(n_images)]
        for i in range(n_images):
            for j in range(self.n_dims):
                output_images[i][self.ordering[j]//n_sq, self.ordering[j]%n_sq] = generated_data[i, j]
                this_img = output_images[i]
                val = 123
        return output_images



class MNIST_4x4(MNIST):
    def __init__(self, optimal_order):
        self.file = datasets.root + 'mnist/MNIST_4x4.npy'
        super().__init__(optimal_order)

class MNIST_7x7(MNIST):
    def __init__(self, optimal_order):
        self.file = datasets.root + 'mnist/MNIST_7x7.npy'
        super().__init__(optimal_order)

class MNIST_8x8(MNIST):
    def __init__(self, optimal_order):
        self.file = datasets.root + 'mnist/MNIST_8x8.npy'
        super().__init__(optimal_order)

class MNIST_16x16(MNIST):
    def __init__(self, optimal_order):
        self.file = datasets.root + 'mnist/MNIST_16x16.npy'
        super().__init__(optimal_order)

class MNIST_28x28(MNIST):
    def __init__(self, optimal_order):
        self.file = datasets.root + 'mnist/MNIST_28x28.npy'
        super().__init__(optimal_order)

def load_data(root_path, optimal_order):
    data = np.add(0.001, np.load(root_path))
    rng.shuffle(data)
    n_train = int((1/3) * data.shape[0])
    n_val = int((1/3) * data.shape[0])

    data = data.reshape((data.shape[0], -1))
    ordering = np.arange(data.shape[-1]) # default ordering
    if optimal_order:
        ordering = order_variables_partial_correlation(data)
        data = data[:, ordering]

    data_train = data[0:n_train]
    data_val = data[n_train:n_train+n_val]
    data_test = data[n_train+n_val:]

    return data_train, data_val, data_test, ordering


def load_data_normalised(root_path, optimal_order):
    data_train, data_val, data_test, ordering = load_data(root_path, optimal_order)

    mu = data_train.mean(axis=0)
    s = np.add(0.000001, data_train.std(axis=0))

    data_train = (data_train - mu) / s
    data_val = (data_val - mu) / s
    data_test = (data_test - mu) / s

    return data_train, data_val, data_test, ordering

if __name__ == '__main__':
    data = MNIST_16x16()