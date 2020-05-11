import pytest
import numpy as np
from numpy import deg2rad
from pytest import approx

from ..functions import (
    jacobian, gm_reduce_single, mod_bearing, mod_elevation, gauss2sigma,
    rotx, roty, rotz, cart2sphere, cart2angles, pol2cart, sphere2cart)
from ..types.array import StateVector, StateVectors
from ..types.state import State, GaussianState


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
        out = 2*ins.state_vector[0]+3*ins.state_vector[1]
        return out

    def fun2d(vec):
        """ test function with 2d input and 2d output"""
        out = np.empty((2, 1))
        out[0] = 2*vec.state_vector[0]**2 + 3*vec.state_vector[1]**2
        out[1] = 2*vec.state_vector[0]+3*vec.state_vector[1]
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


def test_gauss2sigma_float():
    mean = 1.0
    covar = 2.0
    state = GaussianState([[mean]], [[covar]])

    sigma_points_states, mean_weights, covar_weights = gauss2sigma(state, kappa=0)

    for n, sigma_point_state in zip((0, 1, -1), sigma_points_states):
        assert sigma_point_state.state_vector[0, 0] == approx(mean + n*covar**0.5)


def test_gauss2sigma_int():
    mean = 1
    covar = 2.0
    state = GaussianState([[mean]], [[covar]])

    sigma_points_states, mean_weights, covar_weights = gauss2sigma(state, kappa=0)

    for n, sigma_point_state in zip((0, 1, -1), sigma_points_states):
        # Resultant sigma points are still ints
        assert sigma_point_state.state_vector[0, 0] == int(mean + n*covar**0.5)
        assert isinstance(sigma_point_state.state_vector[0, 0], np.integer)


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
