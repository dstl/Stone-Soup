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
    """Compute Jacobian through finite difference calculation

    Parameters
    ----------
    fun : function handle
        A (non-linear) transition function
        Must be of the form "y = fun(x)", where y can be a scalar or \
        :class:`numpy.ndarray` of shape (Nd, 1) or (Nd,)
    x : :class:`numpy.ndarray` of shape (Ns, 1)
        A state vector

    Returns
    -------
    jac: :class:`numpy.ndarray` of shape (Nd, Ns)
        The computed Jacobian
    """

    if isinstance(x, (int, float)):
        ndim = 1
    else:
        ndim = np.shape(x)[0]

    # For numerical reasons the step size needs to large enough
    delta = 100*ndim*np.finfo(float).eps

    f1 = fun(x)
    if isinstance(f1, (int, float)):
        nrows = 1
    else:
        nrows = f1.size

    F2 = np.empty((nrows, ndim))
    X1 = np.tile(x, ndim)+np.eye(ndim)*delta
    for col in range(0, X1.shape[1]):
        F2[:, [col]] = fun(X1[:, [col]])
    jac = np.divide(F2-f1, delta)

    return jac


def cart2pol(x, y):
    """Convert Cartesian coordinates to Polar

    Parameters
    ----------
    x : float
        The x coordinate
    y : float
        the y coordinate

    Returns
    -------
    (float,float)
        A tuple of the form `(range,bearing)`

    """

    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return (rho, phi)


def cart2sphere(x, y, z):
    """Convert Cartesian coordinates to Spherical

    Parameters
    ----------
    x : float
        The x coordinate
    y : float
        the y coordinate
    z : float
        the z coordinate

    Returns
    -------
    (float,float, float)
        A tuple of the form `(range,bearing, elevation)`
        bearing and elevation in radians. Elevation is measured from x,y plane

    """

    rho = np.sqrt(x**2 + y**2 + z**2)
    phi = np.arctan2(y, x)
    theta = np.arcsin(z/rho)
    return (rho, phi, theta)


def cart2angles(x, y, z):
    """Convert Cartesian coordinates to Angles

    Parameters
    ----------
    x : float
        The x coordinate
    y : float
        the y coordinate
    z : float
        the z coordinate

    Returns
    -------
    (float, float)
        A tuple of the form `(bearing, elevation)`
        bearing and elevation in radians. Elevation is measured from x,y plane

    """
    _, phi, theta = cart2sphere(x, y, z)
    return (phi, theta)


def pol2cart(rho, phi):
    """Convert Polar coordinates to Cartesian

    Parameters
    ----------
    rho : float
        Range (a.k.a. radial distance)
    phi : float
        Bearing, expressed in radians

    Returns
    -------
    (float,float)
        A tuple of the form `(x,y)`
    """

    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)


def sphere2cart(rho, phi, theta):
    """Convert Polar coordinates to Cartesian

    Parameters
    ----------
    rho : float
        Range (a.k.a. radial distance)
    phi : float
        Bearing, expressed in radians
    theta : float
        Elevation expressed in radians, measured from x,y plane

    Returns
    -------
    (float,float,float)
        A tuple of the form `(x,y,z)`
    """

    x = rho * np.cos(phi) * np.cos(theta)
    y = rho * np.sin(phi) * np.cos(theta)
    z = rho * np.sin(theta)
    return (x, y, z)
