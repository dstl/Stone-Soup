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
    delta = 1.e-8  # 100*ndim*np.finfo(float).eps

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


def gauss2sigma(mean, covar, alpha=None, beta=None, kappa=None):
    """Approximate a given distribution to a Gaussian, using a
    deterministically selected set of sigma points.

    Parameters
    ----------
    mean : :class:`numpy.ndarray` of shape `(Ns, 1)`
        Mean of the Gaussian
    covar : :class:`~.CovarianceMatrix` of shape `(Ns, Ns)`
        Covariance of the Gaussian
    alpha : float, optional
        Spread of the sigma points. Typically 1e-3.
        (default is 1e-3)
    beta : float, optional
        Used to incorporate prior knowledge of the distribution
        2 is optimal is the state is normally distributed.
        (default is 2)
    kappa : float, optional
        Secondary spread scaling parameter
        (default is calculated as `3-Ns`)

    Returns
    -------
    : :class:`numpy.ndarray` of shape `(Ns, 2*Ns+1)`
        An array containing the locations of the sigma points
    : :class:`numpy.ndarray` of shape `(2*Ns+1,)`
        An array containing the sigma point mean weights
    : :class:`numpy.ndarray` of shape `(2*Ns+1,)`
        An array containing the sigma point covariance weights
    """

    ndim_state = np.shape(mean)[0]

    if alpha is None:
        alpha = 1.0
    if beta is None:
        beta = 0.0
    if kappa is None:
        kappa = 3.0 - ndim_state

    # Compute Square Root matrix via Colesky decomp.
    sqrt_sigma = np.linalg.cholesky(covar)

    # Calculate scaling factor for all off-center points
    alpha2 = np.power(alpha, 2)
    lamda = alpha2 * (ndim_state + kappa) - ndim_state
    c = ndim_state + lamda

    # Calculate sigma point locations
    sigma_points = np.tile(mean, (1, 2 * ndim_state + 1))
    sigma_points[:, 1:(ndim_state + 1)] += sqrt_sigma * np.sqrt(c)
    sigma_points[:, (ndim_state + 1):] -= sqrt_sigma * np.sqrt(c)

    # Calculate weights
    mean_weights = np.ones(2 * ndim_state + 1)
    mean_weights[0] = lamda / c
    mean_weights[1:] = 0.5 / c
    covar_weights = np.copy(mean_weights)
    covar_weights[0] = lamda / c + (1 - alpha2 + beta)

    return sigma_points, mean_weights, covar_weights


def sigma2gauss(sigma_points, mean_weights, covar_weights, covar_noise=None):
    """Calculate estimated mean and covariance from a given set of sigma points

    Parameters
    ----------
    sigma_points : :class:`numpy.ndarray` of shape `(Ns, 2*Ns+1)`
        An array containing the locations of the sigma points
    mean_weights : :class:`numpy.ndarray` of shape `(2*Ns+1,)`
        An array containing the sigma point mean weights
    covar_weights : :class:`numpy.ndarray` of shape `(2*Ns+1,)`
        An array containing the sigma point covariance weights
    covar_noise : :class:`~.CovarianceMatrix` of shape `(Ns, Ns)`, optional
        Additive noise covariance matrix
        (default is 0)

    Returns
    -------
    : :class:`numpy.ndarray` of shape `(Ns, 1)`
        Calculated mean
    : :class:`~.CovarianceMatrix` of shape `(Ns, Ns)`
        Calculated covariance
    """

    mean = sigma_points@mean_weights[:, np.newaxis]

    points_diff = sigma_points - mean

    covar = points_diff@(np.diag(covar_weights))@(points_diff.T)
    if covar_noise is not None:
        covar = covar + covar_noise
    return mean, covar


def unscented_transform(sigma_points, mean_weights, covar_weights,
                        fun, points_noise=None, covar_noise=None):
    """ Apply the Unscented Transform to a set of sigma points

    Apply f to points (with secondary argument points_noise, if available),
    then approximate the resulting mean and covariance. If sigma_noise is
    available, treat it as additional variance due to additive noise.

    Parameters
    ----------
    sigma_points : :class:`numpy.ndarray` of shape `(Ns, 2*Ns+1)`
        An array containing the locations of the sigma points
    mean_weights : :class:`numpy.ndarray` of shape `(2*Ns+1,)`
        An array containing the sigma point mean weights
    covar_weights : :class:`numpy.ndarray` of shape `(2*Ns+1,)`
        An array containing the sigma point covariance weights
    fun : function handle
        A (non-linear) transition function
        Must be of the form "y = fun(x,w)", where y can be a scalar or \
        :class:`numpy.ndarray` of shape (Ns, 1) or (Ns,)
    covar_noise : :class:`~.CovarianceMatrix` of shape `(Ns, Ns)`, optional
        Additive noise covariance matrix
        (default is 0)
    points_noise : :class:`numpy.ndarray` of shape `(Ns, 2*Ns+1,)`, optional
        points to pass into f's second argument
        (default is 0)

    Returns
    -------
    : :class:`numpy.ndarray` of shape `(Ns, 1)`
        Transformed mean
    : :class:`~.CovarianceMatrix` of shape `(Ns, Ns)`
        Transformed covariance
    : :class:`~.CovarianceMatrix` of shape `(Ns,Nm)`
        Calculated cross-covariance matrix
    : :class:`numpy.ndarray` of shape `(Ns, 2*Ns+1)`
        An array containing the locations of the transformed sigma points
    : :class:`numpy.ndarray` of shape `(2*Ns+1,)`
        An array containing the transformed sigma point mean weights
    : :class:`numpy.ndarray` of shape `(2*Ns+1,)`
        An array containing the transformed sigma point covariance weights
    """

    ndim_state, n_points = sigma_points.shape

    # Transform points through f
    sigma_points_t = np.zeros((ndim_state, n_points))
    if points_noise is None:
        sigma_points_t = np.asarray(
            [fun(sigma_points[:, i:i+1])
             for i in range(n_points)]).squeeze(2).T
    else:
        sigma_points_t = np.asarray(
            [fun(sigma_points[:, i:i+1], points_noise[:, i:i+1])
             for i in range(n_points)]).squeeze(2).T

    # Calculate mean and covariance approximation
    mean, covar = sigma2gauss(
        sigma_points_t, mean_weights, covar_weights, covar_noise)

    # Calculate cross-covariance
    cross_covar = (
        (sigma_points-sigma_points[:, 0:1])
        @np.diag(mean_weights)
        @(sigma_points_t-mean).T
    )

    return mean, covar, cross_covar,\
        sigma_points_t, mean_weights, covar_weights


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


def dayOfTheWeek(number):
    """Returns a string with the day of the week, given a number.

    Parameters
    ----------
    number : int
        The number, by order of precedence Mon-Sun, of the day to print out.

    Returns
    -------
    string
        The day of the week that corresponds to the given number.
    """

    day = ""

    if(number < 1 or number > 7):
        day = "Nope"
    # TODO:
    #   Add your code below, possibly starting with "else if....."
    elif(number == 3):
        day = "Wednesday"
    elif(number == 2):
        day = "Tuesday"
    elif number == 1:
        day = "Monday"
    elif number == 4:
        day = "Thursday"
    elif(number == 5):
        day = "Friday"

    return day
