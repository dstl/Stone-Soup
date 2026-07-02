import datetime

import pytest
import numpy as np
from scipy.linalg import expm

from ..nonlinear import SimpleHarmonicMotion
from ....types.state import State
from ....types.array import Matrix, StateVector, StateVectors


@pytest.mark.parametrize("noise_stdev_mag, omega", [
    (0.001, 2*np.pi),   # Standard case, period of 1 second and low noise
    (0.01, 4*np.pi),    # Higher noise and frequency, period of 0.5 seconds
    (0.01, 0.1*np.pi),  # Higher noise and lower frequency, period of 20 seconds
])
def test_shmmodel(noise_stdev_mag: float, omega: float):
    """ SimpleHarmonicMotion Transition Model test """
    # State related variables
    state = State(np.array([[0.0], [1.0]]))
    old_timestamp = datetime.datetime.now()
    timediff = 1  # 1sec
    new_timestamp = old_timestamp + datetime.timedelta(seconds=timediff)
    time_interval = new_timestamp - old_timestamp

    # Model-related components
    F = np.array([[np.cos(omega*timediff), np.sin(omega*timediff)/omega],
                  [-omega*np.sin(omega*timediff), np.cos(omega*timediff)]])

    # Construct the process noise covariance matrix Q using Van Loan's method
    Mdt = Matrix([[0, -timediff, 0, 0],
                  [timediff*omega**2, 0, 0, noise_stdev_mag**2],
                  [0, 0, 0, -timediff*omega**2],
                  [0, 0, timediff, 0]])

    expMdt = expm(Mdt)
    F2 = expMdt[2:4, 2:4]
    G = expMdt[0:2, 2:4]
    Q = F2.T @ G

    # Check the Van Loan method yields the correct state transition matrix F and that Q is
    # symmetric
    assert np.allclose(F, F2.T)
    assert np.allclose(Q, Q.T)

    # Create and a SimpleHarmonicMotion model object
    shm = SimpleHarmonicMotion(omega=omega, noise_stdev_mag=noise_stdev_mag)

    # Ensure ```shm.transfer_function(time_interval)``` returns F
    assert np.allclose(F, shm.jacobian(state, time_interval=time_interval))

    # Ensure ```shm.covar(time_interval)``` returns Q
    assert np.array_equal(Q, shm.covar(time_interval=time_interval))

    # Propagate a state vector through the model (without noise) for one period.
    period = 2.0*np.pi/omega  # 1 period of the motion
    period_timestamp = old_timestamp + datetime.timedelta(seconds=period)
    time_period = period_timestamp - old_timestamp

    new_state_vec_wo_noise = shm.function(
        state,
        time_interval=time_period, noise=False)
    # Assert that the new state vector is approximately equal to the original state vector after
    # one period (since it's a periodic motion)
    assert np.allclose(new_state_vec_wo_noise, state.state_vector)

    # In a quarter of a period, the position will be at a maximum and the velocity will be zero.
    period = period/4.0  # Quarter a period of the motion
    quarter_period_timestamp = old_timestamp + datetime.timedelta(seconds=period)
    time_quarter_period = quarter_period_timestamp - old_timestamp
    new_state_vec_wo_noise = shm.function(
        state,
        time_interval=time_quarter_period, noise=False)
    # Assert that the new state vector is approximately equal to the original state vector with
    # incremented position and velocity after quarter a period.
    assert np.allclose(new_state_vec_wo_noise, np.array([[1.0/omega], [0]]))


# Do a test with StateVectors to check that the model can handle higher-dimensional state vectors
@pytest.mark.parametrize("noise_stdev_mag, omega", [
    (0.001, 2*np.pi),   # Standard case, period of 1 second and low noise
    (0.01, 4*np.pi),    # Higher noise and frequency, period of 0.5 seconds
    (0.01, 0.1*np.pi),  # Higher noise and lower frequency, period of 20 seconds
])
def test_shmmodel_higher_dim(noise_stdev_mag: float, omega: float):
    """ SimpleHarmonicMotion Transition Model test with higher-dimensional state vector """
    # State vector
    sv = StateVector(np.array([0.0, 1.0]))
    # A StateVectors object with five state vectors, separated by a quarter period.
    svs = StateVectors([sv,
                        sv + StateVector([1/omega, -1.0]),
                        -sv,
                        sv - StateVector([1/omega, 1.0]),
                        sv])

    state = State(svs)  # Creates the state.
    old_timestamp = datetime.datetime.now()

    # Create and a SimpleHarmonicMotion model object
    shm = SimpleHarmonicMotion(noise_stdev_mag=noise_stdev_mag, omega=omega)

    # In a quarter of a period, the position and velocities will have incremented by one unit.
    period = 2.0*np.pi/omega
    period = period/4.0  # Quarter a period of the motion
    quarter_period_timestamp = old_timestamp + datetime.timedelta(seconds=period)
    time_quarter_period = quarter_period_timestamp - old_timestamp
    new_state_vec_wo_noise = shm.function(
        state,
        time_interval=time_quarter_period, noise=False)

    # Assert that the new state vector is approximately equal to the original state vector
    # incremented by a quarter period
    for newsv, originalsv in zip(new_state_vec_wo_noise[:, :4], state.state_vector[:, 1:5]):
        assert np.allclose(newsv, originalsv)
