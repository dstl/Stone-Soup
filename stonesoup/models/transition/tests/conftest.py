import pytest
import numpy as np
from scipy.linalg import block_diag


class CT_helper:
    @staticmethod
    def function(state, **kwargs):
        time_interval_sec = kwargs['time_interval'].total_seconds()
        sv1 = state.state_vector
        turn_rate = sv1[4, :]
        # Avoid divide by zero in the function evaluation
        turn_rate[turn_rate == 0.] = np.finfo(float).eps
        dAngle = turn_rate * time_interval_sec
        sv2 = np.array(
            [sv1[0] + np.sin(dAngle)/turn_rate * sv1[1] - sv1[3]/turn_rate*(1. - np.cos(dAngle)),
             sv1[1] * np.cos(dAngle) - sv1[3] * np.sin(dAngle),
             sv1[1]/turn_rate * (1. - np.cos(dAngle)) + sv1[2] + sv1[3]*np.sin(dAngle)/turn_rate,
             sv1[1] * np.sin(dAngle) + sv1[3] * np.cos(dAngle),
             turn_rate])

        return sv2

    @staticmethod
    def covar(linear_noise_coeffs, turn_noise_coeff, time_interval):
        """Returns the transition model noise covariance matrix.

        Returns
        -------
        : :class:`stonesoup.types.state.CovarianceMatrix` of shape\
        (:py:attr:`~ndim_state`, :py:attr:`~ndim_state`)
            The process noise covariance.
        """
        q_x, q_y = linear_noise_coeffs
        q = turn_noise_coeff
        dt = time_interval.total_seconds()

        Q = np.array([[dt**3 / 3., dt**2 / 2.],
                      [dt**2 / 2., dt]])
        C = block_diag(Q*q_x**2, Q*q_y**2, q**2/dt)

        return C


@pytest.fixture
def CT_model():
    return CT_helper
