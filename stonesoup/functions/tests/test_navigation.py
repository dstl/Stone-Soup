import pytest
import numpy as np

from stonesoup.types.state import StateVector
from stonesoup.functions.navigation import earthSpeedFlatSq, earthSpeedSq, earthTurnRateVector, \
    getGravityVector, getEulersAngles, getForceVector, getAngularRotationVector, \
    euler2rotationVector


@pytest.mark.parametrize(
    "x, y, z",
    [  # Cartesian values
        (1., 0., 0.),
        (0., 1., 0.),
        (0., 0., 1.)
    ]
)
def test_EarthSpeed(x, y, z):
    """Speed respect the Earth tests"""
    speed3D = np.power(x, 2) + np.power(y, 2) + np.power(z, 2)
    speed2D = np.power(x, 2) + np.power(y, 2)

    # check that the 2D speed is the same as calculated with EarthSpeedSq
    assert np.allclose(speed3D, earthSpeedSq(x, y, z))

    # check that the 3D speed is the same as calculated with EarthSpeed
    assert np.allclose(speed2D, earthSpeedFlatSq(x, y))


@pytest.mark.parametrize(
    "latitude",
    [
        np.array([40]),
        np.array([55]),
        np.array([60]),
        np.array([10]),
        np.array([35])
    ]
)
def test_earthTurnRateVector(latitude):
    """earthTurnRateVector test"""

    turn_rate = 7.292115e-5

    assert np.allclose(earthTurnRateVector(latitude),
                       turn_rate*np.array([
                           np.cos(np.radians(latitude)),
                           np.zeros_like(latitude),
                           -np.sin(np.radians(latitude))]
                       ))


@pytest.mark.parametrize(
    "latitude, altitude, gv",
    [
        (np.array([55.0018]), np.array([1000]),
         np.array([[-7.59254272e-06], [0.00000000e+00], [9.81199039e+00]])),
    ]
)
def test_getGravityVector(latitude, altitude, gv):
    """getGravityVector test"""
    assert np.allclose(getGravityVector(latitude, altitude), gv, rtol=1e-4)


def test_functions_using_states():
    """A unique routine to test various
        functions using a unique 15 dimension state
        and check the results
    """

    state_test = StateVector(
        [10, 20, 1,  # xyz
         5, -5, 1,   # v, xyz
         1, 1, 1,    # a, xyz
         np.radians(100), np.radians(2),     # psi dpsi
         np.radians(50), np.radians(5),      # theta, dtheta
         np.radians(0), np.radians(1)])      # phi, dpi

    # index state positions
    speed_idx = [1, 4, 7]
    acc_idx = [2, 5, 8]
    ang_idx = [9, 11, 13]
    vang_idx = [10, 12, 14]

    # latitude, longitude, altitude
    reference = np.array([55, 0, 0])

    # set of results
    angular_rotation = np.array([[0.02151771], [-0.00417471], [-0.08787658]])

    force_vector = np.array([[-6.11065211], [-6.50757451], [0.13454552]])

    euler_rotation = np.array([[0.02153659], [-0.00410534], [-0.08788881]])

    euler_angles = (np.array([0.,  -0.04846913, -0.24497866]),
                    np.array([0., -0.04668526, 0.05882353]))

    assert np.allclose(getAngularRotationVector(state_test, reference), angular_rotation)
    assert np.allclose(getForceVector(state_test, reference), force_vector)
    assert np.allclose(euler2rotationVector(state_test[ang_idx], state_test[vang_idx]),
                       euler_rotation)
    assert np.allclose(getEulersAngles(state_test[speed_idx], state_test[acc_idx]), euler_angles)
