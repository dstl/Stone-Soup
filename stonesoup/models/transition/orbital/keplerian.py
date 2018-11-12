# -*- coding: utf-8 -*-

import copy
import numpy as np
import scipy as sp
import datetime as datetime

from ....base import Property
from ....types.orbitalelements import OrbitalElements
from .base import OrbitalModel
from ..linear import LinearGaussianTransitionModel


#class KeplerianTransitionModel(OrbitalModel, LinearGaussianTransitionModel):
class KeplerianTransitionModel(OrbitalModel):

    r""" This class will execute a simple Keplerian transition model on orbital elements. Input is an
    :attr:`OrbitalElements` type.

    Transition proceeds as,

        .. math::

            X_{t_1} = X_{t_0} + [0, 0, 0, 0, 0, n \delta t]^{T}\\


    at Epoch :attr:`OrbitalElements.timestamp` :math:`t_0` for :attr:`OrbitalElements` state :math:`X_{t_0}` and where
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

    def function(self, o_e, t_i):
        print("There shouldn't be function called function. Use :attr:`transition` instead.")
        self.transition(orbital_elements=o_e, time_interval=t_i)

    def rvs(self, num_samples):

        noise = np.array(sp.stats.multivariate_normal.rvs(
            sp.zeros(self.ndim_state()), self.covariance_matrix, num_samples)).transpose()

        return noise

    def pdf(self,o_e,t_i):
        print("No pdf function at present")

    def ndim_state(self):
        return 6

    def covar(self):
        """Construct the covariance matrix"""
        return self.covariance_matrix

    def matrix(self):

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
        print("Returns the covariance matrix at present. TB improved to create the appropriate matrix for time "
              "interval.")

    def transition(self, orbital_elements, time_interval):

        """

        Parameters
        ----------
        orbital_elements: :attr:`OrbitalElements`
        time_interval: :math:`dt` :attr:`datetime.timedelta`

        """

        # New container for orbital elements. Consider deepcopy?
        #orbital_elements_out = OrbitalElements(orbital_elements.state_vector, timestamp=orbital_elements.timestamp)
        orbital_elements_out = copy.deepcopy(orbital_elements)

        # Use the mean motion to find the new mean anomaly
        mean_motion = 2*np.pi/orbital_elements.period()
        new_mean_anomaly = orbital_elements.mean_anomaly() + mean_motion*time_interval.total_seconds()
        # Get the new true anomaly from the new mean anomaly
        new_tru_anomaly = self.itr_eccentric_anomaly(new_mean_anomaly, orbital_elements.eccentricity())

        # Put in the true anomaly
        orbital_elements_out.state_vector[5] = new_tru_anomaly
        orbital_elements_out.timestamp = orbital_elements.timestamp + time_interval

        # Add some noise, if that's what you want
        if self.covariance_matrix is not None:
            orbital_elements_out = self.sample(orbital_elements_out)

        return orbital_elements_out

    def sample(self, o_e):

        """

        Sample from the un-transited state in a multi-variate normal sense according to the covariance matrix.

        Problem is no quantities are actually mvn distributed. So how to guard against unphysical values?

        """

        return OrbitalElements(np.array([np.random.multivariate_normal(o_e.state_vector.flatten(),
                                                                      self.covariance_matrix)]).transpose(),
                               timestamp=o_e.timestamp)

    def itr_eccentric_anomaly(self, mean_anomaly, eccentricity, tolerance=1e-8):
        r"""

        Solve the transcendental equation :math:`E - e sin E = M_e` for E. This is an iterative process using
        Newton's method.

        :param mean_anomaly: Current mean anomaly
        :param eccentricity: Orbital eccentricity
        :param tolerance:
        :return: the eccentric anomaly
        """
        if mean_anomaly < np.pi:
            ecc_anomaly = mean_anomaly + eccentricity/2
        else:
            ecc_anomaly = mean_anomaly - eccentricity/2

        ratio = 1

        while ratio > tolerance:
            f = ecc_anomaly - eccentricity*np.sin(ecc_anomaly)
            fp = 1 - np.cos(ecc_anomaly)
            ratio = f/fp # Need to check conditioning
            ecc_anomaly = ecc_anomaly - ratio

        return ecc_anomaly
