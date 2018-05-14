# -*- coding: utf-8 -*-
"""Mathematical functions used within Stone Soup"""

import numpy as np


def tria(matrix):
    """Square Root Matrix Triangularization

    Given a rectangular square root matrix obtain a square lower-triangular
    square root matrix

    Parameters
    ==========
    matrix : numpy.ndarray
        A `n` by `m` matrix that is generally not square.

    Returns
    =======
    numpy.ndarray
        A square lower-triangular matrix.
    """
    _, upper_triangular = np.linalg.qr(matrix.T)
    lower_triangular = upper_triangular.T

    index = [col
             for col, val in enumerate(np.diag(lower_triangular))
             if val < 0]

    lower_triangular[:, index] *= -1

    return lower_triangular


def jacobian(fun, x):
    """Compute Jacobian through complex step differentiation

    Parameters
    ----------
    fun : function handle
        A(non-linear) transition function
        Must be of the form "y = fun(x)"
    x : :class:`numpy.ndarray` of shape (Ns,1)
        A state vector

    Returns
    -------
    jac: :class:`numpy.ndarray` of shape (Ns,Ns)
        The computed Jacobian
    """

    ndim = np.shape(x)[0]
    h = ndim*np.finfo(float).eps
    jac = np.divide(np.imag(fun(np.tile(x, ndim)+np.eye(ndim)*h*1j)), h)

    return jac
