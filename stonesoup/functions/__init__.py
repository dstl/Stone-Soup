"""Mathematical functions used within Stone Soup"""

import copy
import warnings
from functools import lru_cache

import numpy as np
from numpy import linalg as LA

from scipy.stats import ortho_group

from ..types.array import CovarianceMatrix, StateVector, StateVectors
from ..types.numeric import Probability
from ..types.state import State


def grid_creation(xp_aux, Pp_aux, sFactor, nx, Npa):
    """Grid for point mass filter

    Create a PMF grid based on center, covariance matrix, and sigma probability

    Parameters
    ==========
    xp_aux : numpy.ndarray
        `nx` by `1` center of the grid
    Pp_aux : numpy.ndarray
        'nx' by 'nx' covariance matrix
    sFactor : int
        Parameter for the size of the grid
    nx : int
        Dimension of the grid
    Npa : numpy.ndarray
        'nx' by '' number of points per axis of the grid

    Returns
    =======
    predGrid : numpy.ndarray
        'nx' by prod(Npa) predictive grid
    predGridDelta : list
        grid step per dimension
    gridDim : list of numpy.ndarrays
        grid coordinates per dimension before rotation and translation
    xp_aux : numpy.ndarray
        grid center
    eigVect : numpy.ndarray
        eigenvectors describing the rotation of the grid

    """

    eigVal, eigVect = LA.eig(
        Pp_aux
    )  # eigenvalue and eigenvectors for setting up the grid
    gridBound = np.sqrt(eigVal) * sFactor  # Boundaries of grid

    # Ensure the grid steps are in the right order
    sortInd = np.argsort(np.diag(Pp_aux))
    sortInd = np.argsort(sortInd)

    pom = np.sort(gridBound)
    Ipom = np.argsort(gridBound)
    gridBound = pom[sortInd]

    pom2 = eigVect[:, Ipom]
    eigVect = pom2[:, sortInd]
    gridDim = []  # Reset gridDim for each cycle
    gridStep = []  # Reset gridStep for each cycle
    for ind3 in range(0, nx):  # Creation of propagated grid
        # New grid with middle in 0
        gridDim.append(np.linspace(-gridBound[ind3], gridBound[ind3], Npa[ind3]))
        gridStep.append(np.absolute(gridDim[ind3][0] - gridDim[ind3][1]))  # Grid step

    combvec_predGrid = np.array(np.meshgrid(*gridDim, indexing='ij')).reshape(nx, -1).T
    predGrid = np.dot(eigVect, combvec_predGrid.T)
    # Grid rotation by eigenvectors and translation to the counted unscented mean
    predGrid += xp_aux
    predGridDelta = gridStep  # Grid step size
    return predGrid, predGridDelta, gridDim, xp_aux, eigVect


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


def cholesky_eps(A, lower=False):
    """Perform a Cholesky decomposition on a nearly positive-definite matrix.

    This should return similar results to NumPy/SciPy Cholesky decompositions,
    but compromises for cases for non positive-definite matrix.

    Parameters
    ----------
    A : numpy.ndarray
        Symmetric positive-definite matrix.
    lower : bool
        Whether to return lower or upper triangular decomposition. Default
        `False` which returns upper.

    Returns
    -------
    L : numpy.ndarray
        Upper/lower triangular Cholesky decomposition.
    """
    eps = np.spacing(np.max(np.diag(A)))

    L = np.zeros(A.shape)
    for i in range(A.shape[0]):
        for j in range(i):
            L[i, j] = (A[i, j] - L[i, :]@L[j, :].T) / L[j, j]
        val = A[i, i] - L[i, :]@L[i, :].T
        L[i, i] = np.sqrt(val) if val > eps else np.sqrt(eps)

    if lower:
        return L
    else:
        return L.T


def jacobian(fun, x, **kwargs):
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
    delta = 1e8*np.spacing(x.state_vector.astype(np.float64).ravel())
    # But at least 1e-8
    # TODO: Is this needed? If not, note special case at zero.
    delta[delta < 1e-8] = 1e-8

    x2 = copy.copy(x)  # Create a clone of the input
    x2.state_vector = np.tile(x.state_vector, ndim+1) + np.eye(ndim, ndim+1)*delta[:, np.newaxis]
    x2.state_vector = x2.state_vector.view(StateVectors)

    F = fun(x2, **kwargs)

    jac = np.divide(F[:, :ndim] - F[:, -1:], delta)
    return jac.astype(np.float64)


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
        2 is optimal if the state is normally distributed.
        (default is 2)
    kappa : float, optional
        Secondary spread scaling parameter
        (default is calculated as `3-Ns`)

    Returns
    -------
    : :class:`~.State` with state vector of shape (`Ns`, `2*Ns+1`)
        An State containing the locations of the sigma points.
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
    try:
        sqrt_sigma = np.linalg.cholesky(state.covar)
    except np.linalg.LinAlgError as e:
        warnings.warn(repr(e))
        sqrt_sigma = cholesky_eps(state.covar)

    # Calculate scaling factor for all off-center points
    alpha2 = np.power(alpha, 2)
    lamda = alpha2 * (ndim_state + kappa) - ndim_state
    c = ndim_state + lamda

    # Calculate sigma point locations
    sigma_points = StateVectors([state.state_vector for _ in range(2 * ndim_state + 1)])

    # Cast dtype from int to float to avoid rounding errors
    if np.issubdtype(sigma_points.dtype, np.integer):
        sigma_points = sigma_points.astype(float)

    # Can't use in place addition/subtraction as casting issues may arise when mixing float/int
    sigma_points[:, 1:(ndim_state + 1)] = \
        sigma_points[:, 1:(ndim_state + 1)] + sqrt_sigma*np.sqrt(c)
    sigma_points[:, (ndim_state + 1):] = \
        sigma_points[:, (ndim_state + 1):] - sqrt_sigma*np.sqrt(c)

    # Put these sigma points into s State object list
    sigma_points_states = copy.copy(state)
    sigma_points_states.state_vector = sigma_points

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
    sigma_points : :class:`~.StateVectors` of shape `(Ns, 2*Ns+1)`
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
    : :class:`~.StateVector` of shape `(Ns, 1)`
        Calculated mean
    : :class:`~.CovarianceMatrix` of shape `(Ns, Ns)`
        Calculated covariance
    """

    mean = np.average(sigma_points, axis=1, weights=mean_weights)

    points_diff = sigma_points - mean

    covar = points_diff @ np.diag(covar_weights) @ (points_diff.T)
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
    sigma_points_states : :class:`~.State` with state vector of shape `(Ns, 2*Ns+1)`
        A state containing the locations of the sigma points
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
    : :class:`~.StateVector` of shape `(Ns, 1)`
        Transformed mean
    : :class:`~.CovarianceMatrix` of shape `(Ns, Ns)`
        Transformed covariance
    : :class:`~.CovarianceMatrix` of shape `(Ns,Nm)`
        Calculated cross-covariance matrix
    : :class:`~.StateVectors` of shape `(Ns, 2*Ns+1)`
        An array containing the locations of the transformed sigma points
    : :class:`numpy.ndarray` of shape `(2*Ns+1,)`
        An array containing the transformed sigma point mean weights
    : :class:`numpy.ndarray` of shape `(2*Ns+1,)`
        An array containing the transformed sigma point covariance weights
    """
    # Reconstruct the sigma_points matrix
    sigma_points = sigma_points_states.state_vector

    # Transform points through f
    sigma_points_t = fun(sigma_points_states, points_noise)

    # Calculate mean and covariance approximation
    mean, covar = sigma2gauss(sigma_points_t, mean_weights, covar_weights, covar_noise)

    # Calculate cross-covariance
    cross_covar = (
        (sigma_points-sigma_points[:, 0:1]) @ np.diag(covar_weights) @ (sigma_points_t-mean).T
    ).view(CovarianceMatrix)

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
    theta = np.arcsin(z / rho)
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


def cart2az_el_rg(x, y, z):
    """Convert Cartesian to azimuth (phi), elevation(theta), and range(rho)

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
        A tuple of the form `(phi, theta, rho)`
    """
    rho = np.sqrt(x**2 + y**2 + z**2)
    phi = np.arcsin(x / rho)
    theta = np.arcsin(y / rho)
    return phi, theta, rho


def az_el_rg2cart(phi, theta, rho):
    """Convert azimuth (phi), elevation(theta), and range(rho) to Cartesian

    Parameters
    ----------
    phi: float
        azimuth, expressed in radians
    theta: float
        Elevation expressed in radians, measured from x, y plane
    rho: float
        Range(a.k.a. radial distance)

    Returns
    -------
    (float, float, float)
        A tuple of the form `(phi, theta, rho)`
    """
    x = rho * np.sin(phi)
    y = rho * np.sin(theta)
    z = rho * np.sqrt(1.0 - np.sin(theta)**2 - np.sin(phi)**2)
    return x, y, z


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


def gm_sample(means, covars, size, weights=None, random_state=None):
    """Sample from a mixture of multi-variate Gaussians

    Parameters
    ----------
    means : :class:`~.StateVector`, :class:`~.StateVectors`, :class:`np.ndarray` of shape \
    (num_dims, num_components)
        The means of GM components
    covars : :class:`np.ndarray` of shape (num_components, num_dims, num_dims) or list of \
    :class:`np.ndarray` of shape (num_dims, num_dims)
        Covariance matrices of the GM components
    size : int
        Number of samples to return.
    weights : :class:`np.ndarray` of shape (num_components, ), optional
        The weights of the GM components. If not defined, assumed equal.

    Returns
    -------
    : :class:`~.StateVectors` of shape (num_dims, :attr:`size`)"""

    if isinstance(means, np.ndarray):
        if len(means.shape) == 1:
            means = StateVectors(np.array([means]).T)
        else:
            means = StateVectors(means)

    if isinstance(means, StateVector):
        means = means.view(StateVectors)

    if isinstance(means, StateVectors) and weights is None:
        weights = np.array([1 / means.shape[1]] * means.shape[1])
    elif weights is None:
        weights = np.array([1 / len(means)] * len(means))

    rng = random_state if random_state is not None else np.random

    n_samples = rng.multinomial(size, weights)
    samples = np.vstack([rng.multivariate_normal(mean.ravel(), covar, sample)
                         for (mean, covar, sample) in zip(means, covars, n_samples)]).T

    return StateVectors(samples)


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
    weights = weights / Probability.sum(weights)

    # Cast means as a StateVectors, so this works with ndarray types
    means = means.view(StateVectors)

    # Calculate mean
    mean = np.average(means, axis=1, weights=weights)

    # Calculate covar
    delta_means = means - mean
    covar = np.sum(covars*weights, axis=2, dtype=np.float64) + weights*delta_means@delta_means.T

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
    x = np.asarray(x) % (2*np.pi)  # limit to 2*pi
    N = x // (np.pi / 2)  # Count # of 90 deg multiples

    x = np.where(N == 1, np.pi - x, x)
    x = np.where(N == 2, np.pi - x, x)
    x = np.where(N == 3, x - 2.0 * np.pi, x)
    x = np.where(N == 4, 0.0, x)  # handle the edge case

    return x


def build_rotation_matrix(angle_vector: np.ndarray):
    """
    Calculates and returns the (3D) axis rotation matrix given a vector of
    three angles:
    [roll, pitch/elevation, yaw/azimuth]
    Order of rotations is in reverse: yaw, pitch, roll (z, y, x)
    This is the rotation matrix that implements the rotations that convert the input
    angle_vector to match the x-axis.

    Parameters
    ----------
        angle_vector : :class:`numpy.ndarray` of shape (3, 1): the rotations
        about the :math:'x, y, z' axes.
        In aircraft/radar terms these correspond to
        [roll, pitch/elevation, yaw/azimuth]

    Returns
    -------
        :class:`numpy.ndarray` of shape (3, 3)
            The model (3D) rotation matrix.
    """
    return _build_rotation_matrix(angle_vector[0, 0], angle_vector[1, 0], angle_vector[2, 0])


@lru_cache()
def _build_rotation_matrix(theta_x, theta_y, theta_z):
    return rotx(-theta_x) @ roty(theta_y) @ rotz(-theta_z)


def build_rotation_matrix_xyz(angle_vector: np.ndarray):
    """
    Calculates and returns the (3D) axis rotation matrix given a vector of
    three angles:
    [roll, pitch/elevation, yaw/azimuth]
    Order of rotations is roll, pitch, yaw (x, y, z)
    This is the rotation matrix that implements the rotations that convert a vector aligned to the
    x-axis to the input angle_vector.

    Parameters
    ----------
        angle_vector : :class:`numpy.ndarray` of shape (3, 1): the rotations
        about the :math:'x, y, z' axes.
        In aircraft/radar terms these correspond to
        [roll, pitch/elevation, yaw/azimuth]

    Returns
    -------
        :class:`numpy.ndarray` of shape (3, 3)
            The model (3D) rotation matrix.
    """
    return _build_rotation_matrix_xyz(angle_vector[0, 0], angle_vector[1, 0], angle_vector[2, 0])


@lru_cache()
def _build_rotation_matrix_xyz(theta_x, theta_y, theta_z):
    return rotz(-theta_z) @ roty(theta_y) @ rotx(-theta_x)


def dotproduct(a, b):
    r"""Returns the dot (or scalar) product of two StateVectors or two sets of StateVectors.

    The result for vectors of length :math:`n` is :math:`\Sigma_i^n a_i b_i`.

    Parameters
    ----------
    a : StateVector, StateVectors
        A (set of) state vector(s)
    b : StateVector, StateVectors
        A state vector(s) object of equal dimension to :math:`a`

    Returns
    -------
    : float, numpy.array
        A (set of) scalar value(s) representing the dot product of the vectors.
    """

    if np.shape(a) != np.shape(b):
        raise ValueError("Inputs must be (a collection of) column vectors of the same dimension")

    # Decide whether this is a StateVector or a StateVectors
    if type(a) is StateVector and type(b) is StateVector:
        return np.sum(a * b)
    elif type(a) is StateVectors and type(b) is StateVectors:
        return np.atleast_2d(np.asarray(np.sum(a * b, axis=0)))
    else:
        raise ValueError("Inputs must be `StateVector` or `StateVectors` and of the same type")


def sde_euler_maruyama_integration(fun, t_values, state_x0):
    """Perform SDE Euler Maruyama Integration

    Performs Stochastic Differential Equation Integration using the Euler
    Maruyama method.

    Parameters
    ----------
    fun : callable
        Function to integrate.
    t_values : list of :class:`float`
        Time values to integrate over
    state_x0 : :class:`~.State`
        Initial state for time in first value in :obj:`t_values`.

    Returns
    -------
    : :class:`~.StateVector`
        Final value for the time in last value in :obj:`t_values`
    """
    state_x = copy.deepcopy(state_x0)
    for t, next_t in zip(t_values[:-1], t_values[1:]):
        delta_t = next_t - t
        delta_w = np.random.normal(scale=np.sqrt(delta_t), size=(state_x.ndim, 1))
        a, b = fun(state_x, t)
        state_x.state_vector = state_x.state_vector + a*delta_t + b@delta_w
    return state_x.state_vector


def gauss2cubature(state, alpha=1.0):
    r"""Evaluate the cubature points for an input Gaussian state. This is done under the assumption
    that the input state is :math:`\mathcal{N}(\mathbf{\mu}, \Sigma)` of dimension :math:`n`. We
    calculate the square root of the covariance (via Cholesky factorization), and find the cubature
    points, :math:`X`, as,

    .. math::

        \Sigma &= S S^T

        X_i &= S \xi_i + \mathbf{\mu}

    for :math:`i = 1,...,2n`, where :math:`\xi_i = \sqrt{ \alpha n} [\pm \mathbf{1}]_i` and
    :math:`[\pm \mathbf{1}]_i` are the positive and negative unit vectors in each dimension. We
    include a scaling parameter :math:`\alpha` to allow the selection of cubature points closer to
    the mean or more in the tails, as a potentially useful free parameter.

    Parameters
    ----------
    state : :class:`~.GaussianState`
        A Gaussian state with mean and covariance
    alpha : float, optional
        scaling parameter allowing the selection of cubature points closer to the mean (lower
        values) or further from the mean (higher values)

    Returns
    -------
     : :class:`~.StateVectors`
        Cubature points (as a :class:`~.StateVectors` of dimension :math:`n \times 2n`)

    """
    ndim_state = np.shape(state.state_vector)[0]

    sqrt_covar = np.linalg.cholesky(state.covar)
    cuba_points = np.sqrt(alpha*ndim_state) * np.hstack((np.identity(ndim_state),
                                                         -np.identity(ndim_state)))

    if np.issubdtype(cuba_points.dtype, np.integer):
        cuba_points = cuba_points.astype(float)

    cuba_points = sqrt_covar@cuba_points + state.mean

    return StateVectors(cuba_points)


def cubature2gauss(cubature_points, covar_noise=None, alpha=1.0):
    r"""Get the predicted Gaussian mean and covariance from the cubature points. For dimension
    :math:`n` there are :math:`m = 2n` cubature points. The mean is,

    .. math::

        \mu = \frac{1}{m} \sum\limits_{i=1}^{m} X_i

    and the covariance

    .. math::

        \Sigma = \frac{1}{\alpha}\left(\frac{1}{m} \sum\limits_{i=1}^{m} X_i X_i^T -
        \mathbf{\mu}\mathbf{\mu}^T\right) + Q

    where :math:`Q` is an optional additive noise matrix. The scaling parameter :math:`\alpha`
    allow the for cubature points closer to the mean or more in the tails,

    Parameters
    ----------
    cubature_points : :class:`~.StateVectors`
        Cubature points (as a :class:`~.StateVectors` of dimension :math:`n \times 2n`)
    covar_noise : :class:`~.CovarianceMatrix` of shape `(Ns, Ns)`, optional
        Additive noise covariance matrix
        (default is `None`)
    alpha : float, optional
        scaling parameter allowing the nomination of cubature points closer to the mean (lower
        values) or further from the mean (higher values)

    Returns
    -------
     : :class:`~.GaussianState`
        A Gaussian state with mean and covariance

    """

    m = np.shape(cubature_points)[1]
    mean = np.average(cubature_points, axis=1)
    sigma_mult = cubature_points @ cubature_points.T
    mean_mult = mean @ mean.T
    covar = (1/alpha)*((1/m)*sigma_mult - mean_mult)

    if covar_noise is not None:
        covar = covar + covar_noise

    return mean.view(StateVector), covar.view(CovarianceMatrix)


def cubature_transform(state, fun, points_noise=None, covar_noise=None, alpha=1.0):
    r"""Undertakes the cubature transform as described in [#f]_

    Given a Gaussian distribution, calculates the set of cubature points using
    :meth:`gauss2cubature`, then passes these through the given function and reconstructs the
    Gaussian using :meth:`cubature2gauss`. Returns the mean, covariance, cross covariance and
    transformed cubature points. This instance includes a scaling parameter :math:`\alpha`, not
    included in the reference detailed above, which allows for the selection of cubature points
    closer to, or further from, tne mean.

    Parameters
    ----------
    state : :class:`~.GaussianState`
        A Gaussian state with mean and covariance
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
    alpha : float, optional
        scaling parameter allowing the selection of cubature points closer to the mean (lower
        values) or further from the mean (higher values)

    Returns
    -------
    : :class:`~.StateVector` of shape `(Ns, 1)`
        Transformed mean
    : :class:`~.CovarianceMatrix` of shape `(Ns, Ns)`
        Transformed covariance
    : :class:`~.CovarianceMatrix` of shape `(Ns,Nm)`
        Calculated cross-covariance matrix
    : :class:`~.StateVectors` of shape `(Ns, 2*Ns)`
        An array containing the locations of the transformed cubature points

    References
    ----------
    .. [#f] I. Arasaratnam and S. Haykin, “Cubature Kalman Filters,” in IEEE Transactions on
           Automatic Control, vol. 54, no. 6, pp. 1254-1269, June 2009,
           doi: 10.1109/TAC.2009.2019800.

    """
    ndim_state = np.shape(state.state_vector)[0]
    cubature_points = gauss2cubature(state)

    if points_noise is None:
        cubature_points_t = StateVectors([fun(State(cub_point)) for cub_point in cubature_points])
    else:
        cubature_points_t = StateVectors([
            fun(State(cub_point), points_noise)
            for cub_point, point_noise in zip(cubature_points, points_noise)])

    mean, covar = cubature2gauss(cubature_points_t, covar_noise)

    cross_covar = (1/alpha)*((1./(2*ndim_state))*cubature_points@cubature_points_t.T
                             - np.average(cubature_points, axis=1)@mean.T)
    cross_covar = cross_covar.view(CovarianceMatrix)

    return mean, covar, cross_covar, cubature_points_t


def stochastic_cubature_rule_points(nx, order):
    """Computation of cubature points and weights for the stochastic integration.

    Parameters
    ==========
    nx : int
        Number of points, presumably equivalent to state dimension.
    order : int
        Order for stochastic integration. Only orders 1, 3, and 5 are supported.

    Returns
    =======
    (numpy.ndarray, numpy.ndarray)
        Tuple of sigma points and weights
    """

    if order == 1:
        X = np.random.randn(nx, 1)
        SCRSigmaPoints = np.concatenate((X, -X), axis=1)
        weights = np.array([0.5, 0.5])
    elif order == 3:
        CRSigmaPoints = np.concatenate(
            (np.zeros((nx, 1)), np.eye(nx), -np.eye(nx)), axis=1
        )
        rho = np.sqrt(np.random.chisquare(nx + 2))
        Q = ortho_group.rvs(nx)
        SCRSigmaPoints = Q * rho @ CRSigmaPoints
        weights = np.insert(0.5 * np.ones(2 * nx) / rho**2, 0, (1 - nx / rho**2))

    elif order == 5:
        # generating random values
        r = np.sqrt(np.random.chisquare(2 * nx + 7))

        q = np.random.beta(nx + 2, 3 / 2)

        rho = r * np.sin(np.arcsin(q) / 2)
        delta = r * np.cos(np.arcsin(q) / 2)

        # calculating weights
        c1up = nx + 2 - delta**2
        c1do = rho**2 * (rho**2 - delta**2)
        c2up = nx + 2 - rho**2
        c2do = delta**2 * (delta**2 - rho**2)
        cdo = 2 * (nx + 1) ** 2 * (nx + 2)
        c3 = (7 - nx) * nx**2
        c4 = 4 * (nx - 1) ** 2
        coef1 = c1up * c3 / cdo / c1do
        coef2 = c2up * c3 / cdo / c2do
        coef3 = c1up * c4 / cdo / c1do
        coef4 = c2up * c4 / cdo / c2do

        pom = np.concatenate(
            (
                np.ones(2 * nx + 2) * coef1,
                np.ones(2 * nx + 2) * coef2,
                np.ones(nx * (nx + 1)) * coef3,
                np.ones(nx * (nx + 1)) * coef4,
            ),
            axis=0,
        )
        weights = np.insert(
            pom, 0, (1 - nx * (rho**2 + delta**2 - nx - 2) / (rho**2 * delta**2))
        )

        # Calculating sigma points
        Q = ortho_group.rvs(nx)
        v = np.zeros((nx, nx + 1))
        i_vals, j_vals = np.triu_indices(nx + 1, k=1)
        v[i_vals, i_vals] = np.sqrt((nx+1) * (nx-i_vals) / (nx * (nx-i_vals+1)))
        v[i_vals, j_vals] = -np.sqrt((nx+1) / ((nx-i_vals) * nx * (nx-i_vals+1)))
        v = Q @ v
        i_vals, j_vals = np.tril_indices(nx + 1, k=-1)
        comb_v = v[:, i_vals] + v[:, j_vals]
        y = comb_v / np.linalg.norm(comb_v, axis=0)

        SCRSigmaPoints = np.block(
            [
                np.zeros((nx, 1)),
                -rho * v,
                rho * v,
                -delta * v,
                +delta * v,
                -rho * y,
                rho * y,
                -delta * y,
                delta * y,
            ]
        )
    else:
        raise ValueError("This order of SIF is not supported")

    return (SCRSigmaPoints, weights)


def cub_points_and_tf(nx, order, sqrtCov, mean, transFunct, state):
    r""" Calculates cubature points for stochastic integration filter and
    puts them through given function (measurement/dynamics)

    Parameters
    ==========
    nx : int
       Dimension for cubature points, equivalent to state dimension.
    order : int
        Order for Stochastic Integration. Only orders 1, 3, and 5 are supported
    sqrtCov : np.ndarray
        Matrix square root array of shape (nx, nx) of the covariance matrix
    mean : np.ndarray
        An array of shape (nx, 1) of the state mean
    transFunct : Callable
        A function to transfer state vectors
    state : :class:`~.State`
        State object used to save function output to

    Returns
    =======
    points : numpy.ndarray
        Array of shape (nx, number of points) of cubature points
    w : numpy.ndarray
        Array of shape (number of points) of weights
    trsfPoints : numpy.ndarray
        Array of shape (nx, number of points) (based on order and dim) of
        cubature transformed points
    """

    # -- cubature points and weights computation (for standard normal PDF)
    SCRSigmaPoints, w = stochastic_cubature_rule_points(nx, order)

    # -- points transformation for given filtering mean and covariance matrix
    points = StateVectors(sqrtCov@SCRSigmaPoints + mean)

    # -- points transformation through the function
    state_copy = copy.copy(state)
    state_copy.state_vector = points
    trsfPoints = transFunct(state_copy)

    return points, w, trsfPoints
