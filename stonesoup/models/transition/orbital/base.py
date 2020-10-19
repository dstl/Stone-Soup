import numpy as np
from abc import abstractmethod
from datetime import timedelta

from numpy import matlib
from scipy.stats import multivariate_normal

from ..base import TransitionModel
from ....types.array import CovarianceMatrix


class OrbitalTransitionModel(TransitionModel):
    """Orbital Transition Model base class. This class will execute a
    transition model on an orbital element state vector. Input is an
    :class:~`OrbitalState`, and the various daughter classes will
    implement their chosen state transitions."""

    @property
    @abstractmethod
    def ndim_state(self):
        """Number of state dimensions"""
        pass

    def covar(self, time_interval=timedelta(seconds=1)):
        r"""Return the transition covariance matrix

        Parameters
        ----------
        time_interval: :math:`\delta t` :attr:`datetime.timedelta`, optional
            The time interval over which to test the new state (default is 1
            second)

        Returns
        -------
        : CovarianceMatrix
            The transition covariance matrix

        """
        return CovarianceMatrix(self.noise * time_interval.total_seconds())

    def rvs(self, num_samples=1, orbital_state=None,
            time_interval=timedelta(seconds=0)):
        r"""Sample from the transited state. Do this in a fairly simple-minded
        way by way of additive white noise in Cartesian coordinates.

        .. math::

            \mathbf{x}_t = f(\mathbf{x}_{t-1}) + \mathbf{\zeta},
            \mathbf{\zeta} \sim \mathcal{N}(\mathbf{0}, \Sigma)

        where

        Parameters
        ----------
        num_samples : int, optional
            Number of samples, (default is 1)
        orbital_state: :class:`~.OrbitalState`, optional
            The orbital state class (default is None, in which case a
            Gauss-distributed samples are generated at Cartesian
            :math:`[0,0,0,0,0,0]^T`)
        time_interval : :class:`~.datetime.timedelta`, optional
            The time over which the transition occurs, (default is 0)

        Returns
        -------
        : numpy.array, dimension (6, num_samples)
            num_samples random samples of the state vector drawn from
            a normal distribution defined by the transited mean
            anomaly, :math:`M_{t_1}` and the covariances
            :math:`\Sigma`.

        """
        samples = multivariate_normal.rvs(mean=np.zeros(6), cov=self.covar(
            time_interval=time_interval), size=num_samples)

        # multivariate_normal.rvs() does stupid in the case of a single sample
        # so we have to put this back
        if num_samples == 1:
            samples = np.array([samples])

        if orbital_state is None:
            return samples.T
        else:
            new_cstate_vector = self.transition(
                orbital_state, time_interval=time_interval)
            return matlib.repmat(new_cstate_vector, 1, num_samples) + \
                samples.T

    def pdf(self, test_state, orbital_state,
            time_interval=timedelta(seconds=0)):
        r"""Return the value of the pdf at :math:`\delta t` for a given test
        orbital state. Assumes multi-variate normal distribution in Cartesian
        position and velocity coordinates.

        Parameters
        ----------
        test_state: :class:`~.OrbitalState`
            The orbital state vector to test.
        orbital_state: :class:`~.OrbitalState`
            The 'mean' orbital state class. This undergoes the state transition
            before comparison with the test state.
        time_interval: :math:`\delta t` :attr:`datetime.timedelta`
            The time interval over which to test the new state

        Returns
        -------
        : float
            The value of the pdf at 'test_state'


        """

        return multivariate_normal.pdf(test_state.cartesian_state_vector.
                                       ravel(),
                                       mean=self.transition(orbital_state,
                                                            time_interval).
                                       ravel(),
                                       cov=self.covar(
                                           time_interval=time_interval))
        # ravel appears to be necessary here.
