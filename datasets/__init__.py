root = 'data/'

from ffjord.datasets.power import POWER
from ffjord.datasets.gas import GAS
from ffjord.datasets.hepmass import HEPMASS
from ffjord.datasets.miniboone import MINIBOONE
from ffjord.datasets.bsds300 import BSDS300
from .synthetic import EightGaussians
from .synthetic import Checkerboard
from .synthetic import TwoSpirals
from .mnist import MNIST_4x4, MNIST_7x7, MNIST_8x8, MNIST_16x16, MNIST_28x28

def load_data(name, optimal_order=False):

    if name == 'power':
        return POWER()

    elif name == 'gas':
        return GAS()

    elif name == 'hepmass':
        return HEPMASS()

    elif name == 'miniboone':
        return MINIBOONE()

    elif name == 'bsds300':
        return BSDS300()

    elif name == '8gaussians':
        return EightGaussians()

    elif name == 'checkerboard':
        return Checkerboard()

    elif name == '2spirals':
        return TwoSpirals()

    elif name == 'mnist_4x4':
        return MNIST_4x4(optimal_order)

    elif name == 'mnist_7x7':
        return MNIST_7x7(optimal_order)

    elif name == 'mnist_8x8':
        return MNIST_8x8(optimal_order)

    elif name == 'mnist_16x16':
        return MNIST_16x16(optimal_order)

    elif name == 'mnist_28x28':
        return MNIST_28x28(optimal_order)