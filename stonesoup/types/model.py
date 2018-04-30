# -*- coding: utf-8 -*-
from .base import Type
from abc import abstractmethod, abstractproperty


class Model(Type):
    """Model type

    Base/Abstract class for all models."""

    @abstractmethod
    def eval(self):
        """Model function evaluator method"""
        pass

    @abstractmethod
    def random(self):
        """Model noise/sample generation method"""
        pass

    @abstractmethod
    def pdf(self):
        """Model pdf/likelihood evaluator method"""
        pass


class LinearModel(Model):
    """LinearModel type

    Base/Abstract class for all linear models"""

    @abstractproperty
    def _transfer_matrix(self):
        """ Model transfer matrix.

        A (private) linear, state-independent and (potentially)
        time-variant model transfer matrix"""


class NonLinearModel(Model):
    """NonLinearModel type

    Base/Abstract class for all non-linear models"""

    @abstractproperty
    def _transfer_function(self):
        """ Model transfer function.

        A (private) non-linear, (potentially) state-dependent and
        time-variant model transfer function"""


class TimeVariantModel(Model):
    """TimeVariantModel type

    Base/Abstract class for all Time-Variant (TV) models"""

    @abstractproperty
    def time_variant(self):
        """Time variant

        All time-variant models should define a time variant property"""


class GaussianModel(Model):
    """GaussianModel type

    Base/Abstract class for all Gaussian models"""

    @abstractproperty
    def _noise_covariance(self):
        """ Model covariance

        A (private) state-independent and (potentially) time-variant noise
        covariance matrix"""

    @abstractmethod
    def covar(self):
        """Model covariance getter"""
