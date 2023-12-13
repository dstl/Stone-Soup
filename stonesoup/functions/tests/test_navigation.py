import pytest
import numpy as np

from stonesoup.types.state import StateVector
from stonesoup.functions.navigation import earthSpeedFlatSq, earthSpeedSq, earthTurnRateVector, \
    getGravityVector, rotate3Ddeg, getEulersAngles, getForceVector, getAngularRotationVector, \
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
        40,
        55,
        60,
        10,
        35
    ]
)
def test_earthTurnRateVector(latitude):
    """earthTurnRateVector test"""

    turn_rate = 7.292115e-5

    assert np.allclose(earthTurnRateVector(latitude),
                       turn_rate*np.array([
                           np.cos(np.radians(latitude)),
                           0,
                           -np.sin(np.radians(latitude))]
                       ).reshape(1, -1))


@pytest.mark.parametrize(
    "latitude, altitude, gv",
    [
        (55.0018, 1000, np.array([-7.59254272e-06, 0.00000000e+00, 9.81199039e+00])),
    ]
)
def test_getGravityVector(latitude, altitude, gv):
    """getGravityVector test"""

    assert np.allclose(getGravityVector(latitude, altitude), gv, rtol=1e-4)


@pytest.mark.parametrize(
    "psi, theta, phi",
    [
        (90, 0, 0),
        (0, 90, 0),
        (0, 0, 90),
        (30, 60, 0),
        (0, 30, 60),
        (60, 0, 30),
        (45, 0, 0),
        (0, 45, 0),
        (0, 0, 45)
    ]
)
def test_rotate3Ddeg(psi, theta, phi):
    """rotate3Ddeg test"""

    arr0 = np.array([
        (np.cos(np.radians(psi)), np.sin(np.radians(psi)), 0),
        (-np.sin(np.radians(psi)), np.cos(np.radians(psi)), 0),
        (0, 0, 1)
    ])

    arr1 = np.array([
        (np.cos(np.radians(theta)), 0, -np.sin(np.radians(theta))),
        (0, 1, 0),
        (np.sin(np.radians(theta)), 0, np.cos(np.radians(theta)))
    ])

    arr2 = np.array([
        (1, 0, 0),
        (0, np.cos(np.radians(phi)), np.sin(np.radians(phi))),
        (0, -np.sin(np.radians(phi)), np.cos(np.radians(phi)))
    ])

    assert np.array_equal(rotate3Ddeg(psi, theta, phi), arr2@arr1@arr0)


def test_functions_using_states():
    """A unique routine to test various
        functions using a unique 15 dimension state
        and check the results
    """

    state_test = StateVector(
        [10, 20, 1,  # xyz
         5, -5, 1,   # v, xyz
         1, 1, 1,    # a, xyz
         100, 2,     # psi dpsi
         50, 5,      # theta, dtheta
         0, 1])      # phi, dpi

    # index state positions
    pos_idx = [0, 3, 6]
    speed_idx = [1, 4, 7]
    acc_idx = [2, 5, 8]
    ang_idx = [9, 11, 13]
    vang_idx = [10, 12, 14]

    # latitude, longitude, altitude
    reference = np.array([55, 0, 0])

    # set of results
    angular_rotation = np.array([1.23399665, 4.99995881, 0.64274365]).reshape(-1, 1)

    force_vector = np.array([7.27296026, -1.1574382, -5.04688849]).reshape(-1, 1)

    euler_rotation = np.array([1.23395556, 5., 0.64278761]).reshape(-1, 1)

    euler_angles = (np.array([-14.03624347,  -2.7770768, 0.]),
                    np.array([3.37033997, -2.67486843, 0.]))

    assert np.allclose(getAngularRotationVector(state_test, reference),
                       angular_rotation)
    assert np.allclose(getForceVector(state_test, reference),
                       force_vector)
    assert np.allclose(euler2rotationVector(state_test[ang_idx], state_test[vang_idx]),
                       euler_rotation)
    assert np.allclose(getEulersAngles(state_test[speed_idx], state_test[acc_idx]),
                       euler_angles)
