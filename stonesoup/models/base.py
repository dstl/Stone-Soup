# -*- coding: utf-8 -*-
from abc import abstractmethod

from ..base import Base


class Model(Base):
    """Model type

    Base/Abstract class for all models."""

    @abstractmethod
    def function(self):
        """ Model function"""
        pass

    @abstractmethod
    def rvs(self):
        """Model noise/sample generation method"""
        pass

    @abstractmethod
    def pdf(self):
        """Model pdf/likelihood evaluator method"""
        pass


class LinearModel(Model):
    """LinearModel class

    Base/Abstract class for all linear models"""

    @abstractmethod
    def matrix(self):
        """ Model matrix"""
        pass

    def function(self, state_vector, noise=None, **kwargs):
        """Model function :math:`f(t,x(t),w(t))`

        Parameters
        ----------
        state_vector: :class:`stonesoup.types.state.StateVector`
            An input state vector
        noise: :class:`numpy.ndarray`
            An externally generated random process noise sample (the default in
            `None`, in which case process noise will be generated internally)

        Returns
        -------
        : :class:`numpy.ndarray`
            The model function evaluated.
        """

        if noise is None:
            noise = self.rvs(**kwargs)

        return self.matrix(**kwargs)@state_vector + noise


class NonLinearModel(Model):
    """NonLinearModel class

    Base/Abstract class for all non-linear models"""

    @abstractmethod
    def jacobian(self):
        """ Model jacobian matrix"""
        pass


class TimeVariantModel(Model):
    """TimeVariantModel class

    Base/Abstract class for all time-variant models"""


class TimeInvariantModel(Model):
    """TimeInvariantModel class

    Base/Abstract class for all time-invariant models"""


class GaussianModel(Model):
    """GaussianModel class

    Base/Abstract class for all Gaussian models"""

    @abstractmethod
    def covar(self):
        """Model covariance getter"""
