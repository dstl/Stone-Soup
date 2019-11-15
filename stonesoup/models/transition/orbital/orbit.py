# -*- coding: utf-8 -*-

import copy
import numpy as np
import scipy as sp

from ....base import Property
from ..base import TransitionModel
from ....astro_functions import calculate_itr_eccentric_anomaly


class OrbitalTransitionModel(TransitionModel):

    r""" This class will execute a simple transition model on orbital
    elements. Input is an :class:~`OrbitalState`. How the transition
    occurs is dependent upon the underlying :class:`OrbitalState` type.

    For the :class:~`KeplerianOrbitalState`, transition proceeds as,

        .. math::

            X_{t_1} = X_{t_0} + [0, 0, 0, 0, 0, n \delta t]^{T}\\


    at Epoch :attr:`OrbitalState.timestamp` :math:`t_0` for :attr:`OrbitalElements.state_vector` state :math:`X_{t_0}` and where
    :math:`n` is the mean motion, computed as:

        .. math::

           n = \sqrt{ \frac{\mu}{a^3} }

    and

        .. math::

            X_{t_0} = [e, a, i, \Omega, \omega, M]^{T} \\

    where :math:`e` is the orbital eccentricity (unitless), :math:`a` the semi-major axis (m), :math:`i` the
    inclination (rad), :math:`\Omega` is the longitude of the ascending node (rad), :math:`\omega` the argument of
    periapsis (rad), and :math:`M` the mean anomaly (rad) and :math:`\mu` is the product of the gravitational constant
    and the mass of the primary.


    """

    transition_matrix = Property(
        sp.ndarray, default=None, doc="Transition matrix :math:`\\mathbf{F}`")
    covariance_matrix = Property(
        sp.ndarray, default=None, doc="Transition noise covariance matrix :math:`\\mathbf{Q}`")
    control_matrix = Property(
        sp.ndarray, default=None, doc="Control matrix :math:`\\mathbf{B}`")

    def rvs(self, num_samples):
        """

        :param num_samples: Number of samples, :math:`N`
        :return: N random samples drawn from multivariate normal distribution
        defined by the covariance matrix

        """

        noise = np.array(sp.stats.multivariate_normal.rvs(
            sp.zeros(self.ndim_state()), self.covariance_matrix, num_samples)).transpose()

        return noise

    def pdf(self,o_e,t_i):
        """

        :param o_e: Orbital element state vector
        :param t_i: Time interval
        :return: Not sure, this is a transition model, so p(x_{t+t_i}|x_t)?

        """
        print("No pdf function at present")

    def ndim_state(self):
        return 6

    def covar(self):
        """Construct the covariance matrix"""
        return self.covariance_matrix

    '''def matrix(self):

        """
        :param time_interval:
        :return:


        Parameters
        ----------
        time_interval:: class: `datetime.timedelta`
            A time interval: math:`dt`

        Returns
        -------
        :class:`stonesoup.types.state.CovarianceMatrix` of shape\
        (:py:attr:`~ndim_state`, :py:attr:`~ndim_state`)
            The process noise covariance.

        """
        print("Designed to return the transition matrix at present. TB improved to create the appropriate matrix for time "
              "interval.")

              '''

    def transition(self, orbit, time_interval):

        """

        Parameters
        ----------
        orbital_elements: :attr:`OrbitalElements`
        time_interval: :math:`dt` :attr:`datetime.timedelta`

        """
        orbit_out = copy.deepcopy(orbit)
        if type(orbit) == "KeplerianOrbitState":
            mean_motion = 2*np.pi/orbit.period()
            new_mean_anomaly = orbit.mean_anomaly() + mean_motion*time_interval.total_seconds()
            # Use the mean motion to find the new mean anomaly
            # Get the new eccentric anomaly from the new mean anomaly
            eccentric_anomaly = calculate_itr_eccentric_anomaly(new_mean_anomaly, orbit.eccentricity())
            # And use that to find the new true anomaly
            new_true_anomaly = 2 * np.arctan(np.sqrt((1+orbit.eccentricity()) /
                                                    (1-orbit.eccentricity()))*np.tan(eccentric_anomaly/2))
            # Put the true anomaly into the new state vector
            orbit_out.state_vector[5] = new_true_anomaly
        elif type(orbit) == "TLEOrbitState":
            new_mean_anomaly = orbit.mean_anomaly() + orbit.mean_motion()*time_interval.total_seconds()
            orbit_out.state_vector[5] = new_mean_anomaly
        elif type(orbit) == "EquinoctialOrbitState":
            mean_motion = 2*np.pi/orbit.period()
            new_mean_anomaly = orbit.mean_anomaly() + mean_motion*time_interval.total_seconds()
            orbit_out.state_vector[5] = new_mean_anomaly
        elif type(orbit) == "CartesianOrbitState":
            raise NotImplementedError
        else:
            raise ValueError("Orbit format not recognised")
        orbit_out.timestamp = orbit.timestamp + time_interval

        # Add some noise, if that's what you want
        # if self.covariance_matrix is not None:
        #     orbit_out = self.sample(orbit_out)

        return orbit_out

    def sample(self, o_e):

        """

        Sample from the un-transited state in a multi-variate normal sense according to the covariance matrix.

        Problem is no quantities are actually mvn distributed. So how to guard against unphysical values?

        """
        # return OrbitalState(np.array([np.random.multivariate_normal(o_e.state_vector.flatten(),
        #                                                               self.covariance_matrix)]).transpose(),
        #                        timestamp=o_e.timestamp)
