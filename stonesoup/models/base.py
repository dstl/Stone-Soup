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
