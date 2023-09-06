# -*- coding: utf-8 -*-
from abc import abstractmethod
from datetime import timedelta

import numpy as np

from ..base import TransitionModel, GaussianModel
from ....base import Property
from ....types.array import CovarianceMatrix


class OrbitalTransitionModel(TransitionModel):
    """Orbital transition model base class. This class will execute a transition model on an
    orbital element state vector. Input is an :class:~`OrbitalState`, and the various daughter
    classes will implement their chosen state transitions."""

    @property
    @abstractmethod
    def ndim_state(self):
        """Number of state dimensions"""
        pass


class OrbitalGaussianTransitionModel(OrbitalTransitionModel, GaussianModel):
    """Gaussian version of the orbital transition model base class."""

    process_noise = Property(
        CovarianceMatrix, default=CovarianceMatrix(np.zeros((6, 6))),
        doc=r"Transition (or process) noise covariance :math:`\Sigma` per unit time "
            r"interval (assumed seconds)")

    def _noiseinrightform(self, noise=False, time_interval=timedelta(seconds=1)):
        """Take the noise parameter/matrix and return the noise in the correct form"""

        if isinstance(noise, bool) or noise is None:
            if noise:
                noise = self.rvs(time_interval=time_interval)
            else:
                noise = 0

        return noise

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
        return CovarianceMatrix(self.process_noise * time_interval.total_seconds())
