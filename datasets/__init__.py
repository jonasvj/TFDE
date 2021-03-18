root = 'data/'

from ffjord.datasets.power import POWER
from ffjord.datasets.gas import GAS
from ffjord.datasets.hepmass import HEPMASS
from ffjord.datasets.miniboone import MINIBOONE
from ffjord.datasets.bsds300 import BSDS300
from .synthetic import EightGaussians
from .synthetic import Checkerboard
from .synthetic import TwoSpirals


def load_data(name):
    
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

    
