# -*- coding: utf-8 -*-
"""Mathematical functions used within Stone Soup"""

import numpy as np
from copy import copy

from .types.numeric import Probability
from .types.array import StateVector, StateVectors, CovarianceMatrix


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
        :class:`numpy.ndarray` of shape `(Nd, 1)` or `(Nd,)`
    x : :class:`State`
        A state with state vector of shape `(Ns, 1)`

    Returns
    -------
    jac: :class:`numpy.ndarray` of shape `(Nd, Ns)`
        The computed Jacobian
    """

    ndim, _ = np.shape(x.state_vector)

    # For numerical reasons the step size needs to large enough. Aim for 1e-8
    # relative to spacing between floating point numbers for each dimension
    delta = 1e8*np.spacing(x.state_vector.astype(np.float_).ravel())
    # But at least 1e-8
    # TODO: Is this needed? If not, note special case at zero.
    delta[delta < 1e-8] = 1e-8

    f1 = np.atleast_2d(fun(x))
    nrows, _ = np.shape(f1)

    x2 = copy(x)  # Create a clone of the input
    F2 = np.empty((nrows, ndim))
    X1 = np.tile(x.state_vector, ndim)+np.eye(ndim)*delta

    for col in range(0, X1.shape[1]):
        x2.state_vector = X1[:, [col]]
        F2[:, [col]] = fun(x2)

    jac = np.divide(F2-f1, delta)
    return jac.astype(np.float_)


def gauss2sigma(state, alpha=1.0, beta=2.0, kappa=None):
    """
    Approximate a given distribution to a Gaussian, using a
    deterministically selected set of sigma points.

    Parameters
    ----------
    state : :class:`~State`
        A state object capable of returning a :class:`~.StateVector` of
        shape `(Ns, 1)` representing the Gaussian mean and a
        :class:`~.CovarianceMatrix` of shape `(Ns, Ns)` which is the
        covariance of the distribution
    alpha : float, optional
        Spread of the sigma points. Typically `1e-3`.
        (default is 1)
    beta : float, optional
        Used to incorporate prior knowledge of the distribution
        2 is optimal is the state is normally distributed.
        (default is 2)
    kappa : float, optional
        Secondary spread scaling parameter
        (default is calculated as `3-Ns`)

    Returns
    -------
    : :class:`list` of length `2*Ns+1`
        An list of States containing the locations of the sigma points.
        Note that only the :attr:`state_vector` attribute in these
        States will be meaningful. Other quantities, like :attr:`covar`
        will be inherited from the input and don't really make sense
        for a sigma point.
    : :class:`numpy.ndarray` of shape `(2*Ns+1,)`
        An array containing the sigma point mean weights
    : :class:`numpy.ndarray` of shape `(2*Ns+1,)`
        An array containing the sigma point covariance weights
    """

    ndim_state = np.shape(state.state_vector)[0]

    if kappa is None:
        kappa = 3.0 - ndim_state

    # Compute Square Root matrix via Colesky decomp.
    sqrt_sigma = np.linalg.cholesky(state.covar)

    # Calculate scaling factor for all off-center points
    alpha2 = np.power(alpha, 2)
    lamda = alpha2 * (ndim_state + kappa) - ndim_state
    c = ndim_state + lamda

    # Calculate sigma point locations
    sigma_points = StateVectors([state.state_vector for _ in range(2 * ndim_state + 1)])
    # Can't use in place addition/subtraction as casting issues may arise when mixing float/int
    sigma_points[:, 1:(ndim_state + 1)] = \
        sigma_points[:, 1:(ndim_state + 1)] + sqrt_sigma*np.sqrt(c)
    sigma_points[:, (ndim_state + 1):] = \
        sigma_points[:, (ndim_state + 1):] - sqrt_sigma*np.sqrt(c)

    # Put these sigma points into s State object list
    sigma_points_states = []
    for sigma_point in sigma_points.T:
        state_copy = copy(state)
        state_copy.state_vector = StateVector(sigma_point)
        sigma_points_states.append(state_copy)

    # Calculate weights
    mean_weights = np.ones(2 * ndim_state + 1)
    mean_weights[0] = lamda / c
    mean_weights[1:] = 0.5 / c
    covar_weights = np.copy(mean_weights)
    covar_weights[0] = lamda / c + (1 - alpha2 + beta)

    return sigma_points_states, mean_weights, covar_weights


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
        (default is `None`)

    Returns
    -------
    : :class:`numpy.ndarray` of shape `(Ns, 1)`
        Calculated mean
    : :class:`~.CovarianceMatrix` of shape `(Ns, Ns)`
        Calculated covariance
    """

    mean = np.average(sigma_points, axis=1, weights=mean_weights)

    points_diff = sigma_points - mean

    covar = points_diff@(np.diag(covar_weights))@(points_diff.T)
    if covar_noise is not None:
        covar = covar + covar_noise
    return mean.view(StateVector), covar.view(CovarianceMatrix)


def unscented_transform(sigma_points_states, mean_weights, covar_weights,
                        fun, points_noise=None, covar_noise=None):
    """
    Apply the Unscented Transform to a set of sigma points

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
        :class:`numpy.ndarray` of shape `(Ns, 1)` or `(Ns,)`
    covar_noise : :class:`~.CovarianceMatrix` of shape `(Ns, Ns)`, optional
        Additive noise covariance matrix
        (default is `None`)
    points_noise : :class:`numpy.ndarray` of shape `(Ns, 2*Ns+1,)`, optional
        points to pass into f's second argument
        (default is `None`)

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
    # Reconstruct the sigma_points matrix
    sigma_points = StateVectors([
        sigma_points_state.state_vector for sigma_points_state in sigma_points_states])

    # Transform points through f
    if points_noise is None:
        sigma_points_t = StateVectors([
            fun(sigma_points_state) for sigma_points_state in sigma_points_states])
    else:
        sigma_points_t = StateVectors([
            fun(sigma_points_state, points_noise)
            for sigma_points_state, point_noise in zip(sigma_points_states, points_noise.T)])

    # Calculate mean and covariance approximation
    mean, covar = sigma2gauss(sigma_points_t, mean_weights, covar_weights, covar_noise)

    # Calculate cross-covariance
    cross_covar = (
        (sigma_points-sigma_points[:, 0:1])
        @np.diag(mean_weights)
        @(sigma_points_t-mean).T
    )

    return mean, covar, cross_covar, sigma_points_t, mean_weights, covar_weights


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

    For a given rotation angle: :math:`\theta`, this function evaluates \
    and returns the rotation matrix:

    .. math:: R_{x}(\theta) = \begin{bmatrix}
                        1 & 0 & 0 \\
                        0 & cos(\theta) & -sin(\theta) \\
                        0 & sin(\theta) & cos(\theta)
                        \end{bmatrix}
       :label: Rx

    Parameters
    ----------
    theta: Union[float, np.ndarray]
        Rotation angle specified as a real-valued number or an \
        :class:`np.ndarray` of reals. The rotation angle is positive if the \
        rotation is in the clockwise direction when viewed by an observer \
        looking down the x-axis towards the origin. Angle units are in radians.

    Returns
    -------
    : :class:`numpy.ndarray` of shape (3, 3) or (3, 3, n) for array input
        Rotation matrix around x-axis of the form :eq:`Rx`.
    """

    c, s = np.cos(theta), np.sin(theta)
    zero = np.zeros_like(theta)
    one = np.ones_like(theta)
    return np.array([[one, zero, zero],
                     [zero, c, -s],
                     [zero, s, c]])


def roty(theta):
    r"""Rotation matrix for rotations around y-axis

    For a given rotation angle: :math:`\theta`, this function evaluates \
    and returns the rotation matrix:

    .. math::
        R_{y}(\theta) = \begin{bmatrix}
                        cos(\theta) & 0 & sin(\theta) \\
                        0 & 1 & 0 \\
                        - sin(\theta) & 0 & cos(\theta)
                        \end{bmatrix}
       :label: Ry

    Parameters
    ----------
    theta: Union[float, np.ndarray]
        Rotation angle specified as a real-valued number or an \
        :class:`np.ndarray` of reals. The rotation angle is positive if the \
        rotation is in the clockwise direction when viewed by an observer \
        looking down the y-axis towards the origin. Angle units are in radians.

    Returns
    -------
    : :class:`numpy.ndarray` of shape (3, 3) or (3, 3, n) for array input
        Rotation matrix around y-axis of the form :eq:`Ry`.
    """

    c, s = np.cos(theta), np.sin(theta)
    zero = np.zeros_like(theta)
    one = np.ones_like(theta)
    return np.array([[c, zero, s],
                     [zero, one, zero],
                     [-s, zero, c]])


def rotz(theta):
    r"""Rotation matrix for rotations around z-axis

    For a given rotation angle: :math:`\theta`, this function evaluates \
    and returns the rotation matrix:

    .. math::
        R_{z}(\theta) = \begin{bmatrix}
                        cos(\theta) & -sin(\theta) & 0 \\
                        sin(\theta) & cos(\theta) & 0 \\
                        0 & 0 & 1
                        \end{bmatrix}
       :label: Rz

    Parameters
    ----------
    theta: Union[float, np.ndarray]
        Rotation angle specified as a real-valued number or an \
        :class:`np.ndarray` of reals. The rotation angle is positive if the \
        rotation is in the clockwise direction when viewed by an observer \
        looking down the z-axis towards the origin. Angle units are in radians.

    Returns
    -------
    : :class:`numpy.ndarray` of shape (3, 3) or (3, 3, n) for array input
        Rotation matrix around z-axis of the form :eq:`Rz`.
    """

    c, s = np.cos(theta), np.sin(theta)
    zero = np.zeros_like(theta)
    one = np.ones_like(theta)
    return np.array([[c, -s, zero],
                     [s, c, zero],
                     [zero, zero, one]])


def gm_reduce_single(means, covars, weights):
    """Reduce mixture of multi-variate Gaussians to single Gaussian

    Parameters
    ----------
    means : :class:`~.StateVectors`
        The means of the GM components
    covars : np.array of shape (num_dims, num_dims, num_components)
        The covariance matrices of the GM components
    weights : np.array of shape (num_components,)
        The weights of the GM components

    Returns
    -------
    : :class:`~.StateVector`
        The mean of the reduced/single Gaussian
    : :class:`~.CovarianceMatrix`
        The covariance of the reduced/single Gaussian
    """
    # Normalise weights such that they sum to 1
    weights = weights/Probability.sum(weights)

    # Calculate mean
    mean = np.average(means, axis=1, weights=weights)

    # Calculate covar
    delta_means = means - mean
    covar = np.sum(covars*weights, axis=2, dtype=np.float_) + weights*delta_means@delta_means.T

    return mean.view(StateVector), covar.view(CovarianceMatrix)


def mod_bearing(x):
    r"""Calculates the modulus of a bearing. Bearing angles are within the \
    range :math:`-\pi` to :math:`\pi`.

    Parameters
    ----------
    x: float
        bearing angle in radians

    Returns
    -------
    float
        Angle in radians in the range math: :math:`-\pi` to :math:`+\pi`
    """

    x = (x+np.pi) % (2.0*np.pi)-np.pi

    return x


def mod_elevation(x):
    r"""Calculates the modulus of an elevation angle. Elevation angles \
    are within the range :math:`-\pi/2` to :math:`\pi/2`.

    Parameters
    ----------
    x: float
        elevation angle in radians

    Returns
    -------
    float
        Angle in radians in the range math: :math:`-\pi/2` to :math:`+\pi/2`
    """
    x = x % (2*np.pi)  # limit to 2*pi
    N = x//(np.pi/2)   # Count # of 90 deg multiples
    if N == 1:
        x = np.pi - x
    elif N == 2:
        x = np.pi - x
    elif N == 3:
        x = x - 2.0 * np.pi
    return x
