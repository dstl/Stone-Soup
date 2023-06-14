import pytest
import numpy as np
from numpy import deg2rad
from scipy.linalg import cholesky, LinAlgError
from pytest import approx, raises

from .. import (
    cholesky_eps, jacobian, gm_reduce_single, mod_bearing, mod_elevation, gauss2sigma,
    rotx, roty, rotz, cart2sphere, cart2angles, pol2cart, sphere2cart, dotproduct, gm_sample)
from ...types.array import StateVector, StateVectors, Matrix
from ...types.state import State, GaussianState


def test_cholesky_eps():
    matrix = np.array([[0.4, -0.2, 0.1],
                       [0.3, 0.1, -0.2],
                       [-0.3, 0.0, 0.4]])
    matrix = matrix@matrix.T

    cholesky_matrix = cholesky(matrix)

    assert cholesky_eps(matrix) == approx(cholesky_matrix)
    assert cholesky_eps(matrix, True) == approx(cholesky_matrix.T)


def test_cholesky_eps_bad():
    matrix = np.array(
        [[ 0.05201447,  0.02882126, -0.00569971, -0.00733617],  # noqa: E201
         [ 0.02882126,  0.01642966, -0.00862847, -0.00673035],  # noqa: E201
         [-0.00569971, -0.00862847,  0.06570757,  0.03251551],
         [-0.00733617, -0.00673035,  0.03251551,  0.01648615]])
    with raises(LinAlgError):
        cholesky(matrix)
    cholesky_eps(matrix)


def test_jacobian():
    """ jacobian function test """

    # State related variables
    state_mean = StateVector([[3.0], [1.0]])

    def f(x):
        return np.array([[1, 1], [0, 1]])@x.state_vector

    jac = jacobian(f, State(state_mean))
    assert np.allclose(jac, np.array([[1, 1], [0, 1]]))


def test_jacobian2():
    """ jacobian function test """

    # Sample functions to compute Jacobian on
    def fun(x):
        """ function for testing scalars i.e. scalar input, scalar output"""
        return 2*x.state_vector**2

    def fun1d(ins):
        """ test function with vector input, scalar output"""
        out = 2*ins.state_vector[0, :]+3*ins.state_vector[1, :]
        return np.atleast_2d(out)

    def fun2d(vec):
        """ test function with 2d input and 2d output"""
        out = np.empty(vec.state_vector.shape)
        out[0, :] = 2*vec.state_vector[0, :]**2 + 3*vec.state_vector[1, :]**2
        out[1, :] = 2*vec.state_vector[0, :]+3*vec.state_vector[1, :]
        return out

    x = 3
    jac = jacobian(fun, State(StateVector([[x]])))
    assert np.allclose(jac, 4*x)

    x = StateVector([[1], [2]])
    # Tolerance value to use to test if arrays are equal
    tol = 1.0e-5

    jac = jacobian(fun1d, State(x))
    T = np.array([2.0, 3.0])

    FOM = np.where(np.abs(jac-T) > tol)
    # Check # of array elements bigger than tol
    assert len(FOM[0]) == 0

    jac = jacobian(fun2d, State(x))
    T = np.array([[4.0*x[0], 6*x[1]],
                  [2, 3]])
    FOM = np.where(np.abs(jac - T) > tol)
    # Check # of array elements bigger than tol
    assert len(FOM[0]) == 0


def test_jacobian_param():
    """ jacobian function test """

    # Sample functions to compute Jacobian on
    def fun(x, value=0.0):
        """ function for jabcobian parameter passing"""
        return value*x.state_vector

    x = 4
    value = 2.0
    jac = jacobian(fun, State(StateVector([[x]])), value=value)
    assert np.allclose(value, jac)


def test_jacobian_large_values():
    # State related variables
    state = State(StateVector([[1E10], [1.0]]))

    def f(x):
        return x.state_vector**2

    jac = jacobian(f, state)
    assert np.allclose(jac, np.array([[2e10, 0.0], [0.0, 2.0]]))


def test_gm_reduce_single():

    means = StateVectors([StateVector([1, 2]), StateVector([3, 4]), StateVector([5, 6])])
    covars = np.stack([[[1, 1], [1, 0.7]],
                       [[1.2, 1.4], [1.3, 2]],
                       [[2, 1.4], [1.2, 1.2]]], axis=2)
    weights = np.array([1, 2, 5])

    mean, covar = gm_reduce_single(means, covars, weights)

    assert np.allclose(mean, np.array([[4], [5]]))
    assert np.allclose(covar, np.array([[3.675, 3.35],
                                        [3.2, 3.3375]]))

    # Test handling of means as array instead of StateVectors
    mean, covar = gm_reduce_single(means.view(np.ndarray), covars, weights)

    assert np.allclose(mean, np.array([[4], [5]]))
    assert np.allclose(covar, np.array([[3.675, 3.35],
                                        [3.2, 3.3375]]))


def test_bearing():
    bearing_in = [10., 170., 190., 260., 280., 350., 705]
    rad_in = deg2rad(bearing_in)

    bearing_out = [10., 170., -170., -100., -80., -10., -15.]
    rad_out = deg2rad(bearing_out)

    for ind, val in enumerate(rad_in):
        assert rad_out[ind] == approx(mod_bearing(val))


def test_elevation():
    elev_in = [10., 80., 110., 170., 190., 260., 280]
    rad_in = deg2rad(elev_in)

    elev_out = [10., 80., 70., 10., -10., -80., -80.]
    rad_out = deg2rad(elev_out)

    for ind, val in enumerate(rad_in):
        assert rad_out[ind] == approx(mod_elevation(val))


@pytest.mark.parametrize(
    "mean",
    [
        1,      # int
        1.0     # float
    ]
)
def test_gauss2sigma(mean):
    covar = 2.0
    state = GaussianState([[mean]], [[covar]])

    sigma_points_states, mean_weights, covar_weights = gauss2sigma(state, kappa=0)

    for n, sigma_point_state in zip((0, 1, -1), sigma_points_states):
        assert sigma_point_state.state_vector[0, 0] == approx(mean + n*covar**0.5)


@pytest.mark.parametrize(
    "angle",
    [
        (
            np.array([np.pi]),  # angle
            np.array([np.pi / 2]),
            np.array([-np.pi]),
            np.array([-np.pi / 2]),
            np.array([np.pi / 4]),
            np.array([-np.pi / 4]),
            np.array([np.pi / 8]),
            np.array([-np.pi / 8]),
        )
    ]
)
def test_rotations(angle):

    c, s = np.cos(angle), np.sin(angle)
    zero = np.zeros_like(angle)
    one = np.ones_like(angle)

    assert np.array_equal(rotx(angle), np.array([[one, zero, zero],
                                                 [zero, c, -s],
                                                 [zero, s, c]]))
    assert np.array_equal(roty(angle), np.array([[c, zero, s],
                                                 [zero, one, zero],
                                                 [-s, zero, c]]))
    assert np.array_equal(rotz(angle), np.array([[c, -s, zero],
                                                 [s, c, zero],
                                                 [zero, zero, one]]))


@pytest.mark.parametrize(
    "x, y, z",
    [  # Cartesian values
        (1., 0., 0.),
        (0., 1., 0.),
        (0., 0., 1.),
        (1., 1., 0.),
        (1., 0., 1.),
        (0., 1., 1.),
        (1., 1., 1.)
    ]
)
def test_cart_sphere_inversions(x, y, z):

    rho, phi, theta = cart2sphere(x, y, z)

    # Check sphere2cart(cart2sphere(cart)) == cart
    assert np.allclose(np.array([x, y, z]), sphere2cart(rho, phi, theta))

    # Check cart2angle == cart2sphere for angles
    assert np.allclose(np.array([phi, theta]), cart2angles(x, y, z))

    # Check that pol2cart(cart2angle(cart)) == cart
    #   note, this only works correctly when z==0
    if z == 0:
        assert np.allclose(np.array([x, y]), pol2cart(rho, phi))


@pytest.mark.parametrize(
    "state_vector1, state_vector2",
    [  # Cartesian values
        (StateVector([2, 4]), StateVector([2, 1])),
        (StateVector([-1, 1, -4]), StateVector([-2, 1, 1])),
        (StateVector([-2, 0, 3, -1]), StateVector([1, 0, -1, 4])),
        (StateVector([-1, 0]), StateVector([1, -2, 3])),
        (Matrix([[1, 0], [0, 1]]), Matrix([[3, 1], [1, -3]])),
        (StateVectors([[1, 0], [0, 1]]), StateVectors([[3, 1], [1, -3]])),
        (StateVectors([[1, 0], [0, 1]]), StateVector([3, 1]))
     ]
)
def test_dotproduct(state_vector1, state_vector2):

    # Test that they raise the right error if not 1d, i.e. vectors
    if type(state_vector1) != type(state_vector2):
        with pytest.raises(ValueError):
            dotproduct(state_vector1, state_vector2)
    elif type(state_vector1) != StateVectors and type(state_vector2) != StateVectors and \
            type(state_vector2) != StateVector and type(state_vector1) != StateVector:
        with pytest.raises(ValueError):
            dotproduct(state_vector1, state_vector2)
    else:
        if len(state_vector1) != len(state_vector2):
            # If they're different lengths check that the correct error is thrown
            with pytest.raises(ValueError):
                dotproduct(state_vector1, state_vector2)
        else:
            # This is what the dotproduct function actually does
            out = 0
            for a_i, b_i in zip(state_vector1, state_vector2):
                out += a_i * b_i

            assert np.allclose(dotproduct(state_vector1, state_vector2),
                               np.reshape(out, np.shape(dotproduct(state_vector1, state_vector2))))


@pytest.mark.parametrize(
    "means, covars, weights, size",
    [
        (
            [np.array([10, 10]), np.array([20, 20]), np.array([30, 30])],  # means
            [np.eye(2), np.eye(2), np.eye(2)],  # covars
            np.array([1/3]*3),  # weights
            20  # size
        ), (
            StateVectors(np.array([[20, 30, 40, 50], [20, 30, 40, 50]])),  # means
            [np.eye(2), np.eye(2), np.eye(2), np.eye(2)],  # covars
            np.array([1/4]*4),  # weights
            20  # size
        ), (
            [np.array([10, 10]), np.array([20, 20]), np.array([30, 30])],  # means
            np.array([np.eye(2), np.eye(2), np.eye(2)]),  # covars
            np.array([1/3]*3),  # weights
            20  # size
        ), (
            [StateVector(np.array([10, 10])), StateVector(np.array([20, 20])),
             StateVector(np.array([30, 30]))],  # means
            [np.eye(2), np.eye(2), np.eye(2)],  # covars
            np.array([1/3]*3),  # weights
            20  # size
        ), (
            StateVector(np.array([10, 10])),  # means
            [np.eye(2)],  # covars
            np.array([1]),  # weights
            20  # size
        ), (
            np.array([10, 10]),  # means
            [np.eye(2)],  # covars
            np.array([1]),  # weights
            20  # size
        ), (
            [np.array([10, 10]), np.array([20, 20]), np.array([30, 30])],  # means
            [np.eye(2), np.eye(2), np.eye(2)],  # covars
            None,  # weights
            20  # size
        ), (
            StateVectors(np.array([[20, 30, 40, 50], [20, 30, 40, 50]])),  # means
            [np.eye(2), np.eye(2), np.eye(2), np.eye(2)],  # covars
            None,  # weights
            20  # size
        )
    ], ids=["mean_list", "mean_statevectors", "3d_covar_array", "mean_statevector_list",
            "single_statevector_mean", "single_ndarray_mean", "no_weight_mean_list",
            "no_weight_mean_statevectors"]
)
def test_gm_sample(means, covars, weights, size):
    samples = gm_sample(means, covars, size, weights=weights)

    # check orientation and size of samples
    assert samples.shape[1] == size
    # check number of dimensions
    if isinstance(means, list):
        assert samples.shape[0] == means[0].shape[0]
    else:
        assert samples.shape[0] == means.shape[0]
