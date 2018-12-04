# -*- coding: utf-8 -*-
"""Mathematical functions used within Stone Soup"""

import numpy as np

# from .types.state import StateVector


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
    x: float
        The x coordinate
    y: float
        the y coordinate

    Returns
    -------
    (float, float)
        A tuple of the form `(range, bearing)`

    """

    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return (rho, phi)


def cart2sphere(x, y, z):
    """Convert Cartesian coordinates to Spherical

    Parameters
    ----------
    x: float
        The x coordinate
    y: float
        the y coordinate
    z: float
        the z coordinate

    Returns
    -------
    (float, float, float)
        A tuple of the form `(range, bearing, elevation)`
        bearing and elevation in radians. Elevation is measured from x, y plane

    """

    rho = np.sqrt(x**2 + y**2 + z**2)
    phi = np.arctan2(y, x)
    theta = np.arcsin(z/rho)
    return (rho, phi, theta)


def cart2angles(x, y, z):
    """Convert Cartesian coordinates to Angles

    Parameters
    ----------
    x: float
        The x coordinate
    y: float
        the y coordinate
    z: float
        the z coordinate

    Returns
    -------
    (float, float)
        A tuple of the form `(bearing, elevation)`
        bearing and elevation in radians. Elevation is measured from x, y plane

    """
    _, phi, theta = cart2sphere(x, y, z)
    return (phi, theta)


def pol2cart(rho, phi):
    """Convert Polar coordinates to Cartesian

    Parameters
    ----------
    rho: float
        Range(a.k.a. radial distance)
    phi: float
        Bearing, expressed in radians

    Returns
    -------
    (float, float)
        A tuple of the form `(x, y)`
    """

    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)


def sphere2cart(rho, phi, theta):
    """Convert Polar coordinates to Cartesian

    Parameters
    ----------
    rho: float
        Range(a.k.a. radial distance)
    phi: float
        Bearing, expressed in radians
    theta: float
        Elevation expressed in radians, measured from x, y plane

    Returns
    -------
    (float, float, float)
        A tuple of the form `(x, y, z)`
    """

    x = rho * np.cos(phi) * np.cos(theta)
    y = rho * np.sin(phi) * np.cos(theta)
    z = rho * np.sin(theta)
    return (x, y, z)


def rotx(theta):
    r"""Rotation matrix for rotations around x-axis

    For a given rotation angle: math: `\theta`, this function evaluates\
    and returns the rotation matrix:

    .. math: :
        : label: Rx
        R_{x}(\theta) = \begin{bmatrix}
                        1 & 0 & 0 \\
                        0 & cos(\theta) & -sin(\theta) \\
                        0 & sin(\theta) & cos(\theta)
                        \end{bmatrix}

    Parameters
    ----------
    theta: float
        Rotation angle specified as a real-valued number. The rotation angle\
        is positive if the rotation is in the clockwise direction\
        when viewed by an observer looking down the x-axis towards the\
        origin. Angle units are in radians.

    Returns
    -------
    : : class: `numpy.ndarray` of shape (3, 3)
        Rotation matrix around x-axis of the form eq: `Rx`
    """

    c, s = np.cos(theta), np.sin(theta)

    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s, c]])


def roty(theta):
    r"""Rotation matrix for rotations around y-axis

    For a given rotation angle: math: `\theta`, this function evaluates\
    and returns the rotation matrix:

    .. math: :
        : label: Ry
        R_{y}(\theta) = \begin{bmatrix}
                        cos(\theta) & 0 & sin(\theta) \\
                        0 & 1 & 0 \\
                        - sin(\theta) & 0 & cos(\theta)
                        \end{bmatrix}

    Parameters
    ----------
    theta: float
        Rotation angle specified as a real-valued number. The rotation angle\
        is positive if the rotation is in the clockwise direction\
        when viewed by an observer looking down the y-axis towards the\
        origin. Angle units are in radians.

    Returns
    -------
    : : class: `numpy.ndarray` of shape (3, 3)
        Rotation matrix around y-axis of the form eq: `Ry`
    """

    c, s = np.cos(theta), np.sin(theta)

    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])


def rotz(theta):
    r"""Rotation matrix for rotations around z-axis

    For a given rotation angle: math: `\theta`, this function evaluates\
    and returns the rotation matrix:

    .. math: :
        : label: Rz
        R_{z}(\theta) = \begin{bmatrix}
                        cos(\theta) & -sin(\theta) & 0 \\
                        sin(\theta) & cos(\theta) & 0 \\
                        0 & 0 & 1
                        \end{bmatrix}

    Parameters
    ----------
    theta: float
        Rotation angle specified as a real-valued number. The rotation angle\
        is positive if the rotation is in the clockwise direction\
        when viewed by an observer looking down the z-axis towards the\
        origin. Angle units are in radians.

    Returns
    -------
    : : class: `numpy.ndarray` of shape (3, 3)
        Rotation matrix around z-axis of the form eq: `Rz`
    """

    c, s = np.cos(theta), np.sin(theta)

    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]])
