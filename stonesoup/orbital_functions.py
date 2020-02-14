# -*- coding: utf-8 -*-
"""Functions used within multiple orbital classes in Stone Soup

"""
import numpy as np


def stumpf_s(z):
    """The Stumpf S function"""
    if z > 0:
        sqz = np.sqrt(z)
        return (sqz - np.sin(sqz)) / sqz ** 3
    elif z < 0:
        sqz = np.sqrt(-z)
        return (np.sinh(sqz) - sqz) / sqz ** 3
    elif z == 0:
        return 1 / 6
    else:
        raise ValueError("Shouldn't get to this point")


def stumpf_c(z):
    """The Stumpf C function"""
    if z > 0:
        sqz = np.sqrt(z)
        return (1 - np.cos(sqz)) / sqz ** 2
    elif z < 0:
        sqz = np.sqrt(-z)
        return (np.cosh(sqz) - 1) / sqz ** 2
    elif z == 0:
        return 1 / 2
    else:
        raise ValueError("Shouldn't get to this point")
