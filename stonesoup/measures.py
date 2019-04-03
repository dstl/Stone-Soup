# -*- coding: utf-8 -*-
from abc import abstractmethod

import numpy as np
from scipy.spatial import distance

from .base import Base, Property


class Measure(Base):
    """Measure base type

    A measure provides a means to assess the seperation between two
    :class:`~.State` objects state1 and state2.
    """
    mapping = Property(
        np.array,
        default=None,
        doc="Mapping array which specifies which elements within the"
            " state vectors are to be assessed as part of the measure"
    )

    @abstractmethod
    def __call__(self, state1, state2):
        r"""
        Compute the distance between a pair of :class:`~.State` objects

        Parameters
        ----------
        state1 : :class:`~.State`
        state2 : :class:`~.State`

        Returns
        -------
        float
            distance measure between a pair of input :class:`~.State` objects

        """
        return NotImplementedError


class Euclidean(Measure):
    r"""Euclidean distance measure

    This measure returns the Euclidean distance between a pair of
    :class:`~.State` objects.

    The Euclidean distance between a pair of state vectors :math:`u` and
    :math:`v` is defined as:

    .. math::
         \sqrt{\sum_{i=1}^{N}{(u_i - v_i)^2}}

    """
    def __call__(self, state1, state2):
        r"""Calculate the Euclidean distance between a pair of state vectors

        Parameters
        ----------
        state1 : :class:`~.State`
        state2 : :class:`~.State`

        Returns
        -------
        float
            Euclidean distance between two input :class:`~.State`

        """
        # Calculate Euclidean distance between two state
        if self.mapping is not None:
            return distance.euclidean(state1.state_vector[self.mapping],
                                      state2.state_vector[self.mapping])
        else:
            return distance.euclidean(state1.state_vector, state2.state_vector)


class EuclideanWeighted(Measure):
    r"""Weighted Euclidean distance measure

    This measure returns the Euclidean distance between a pair of
    :class:`~.State` objects, taking into account a specified weighting.

    The Weighted Euclidean distance between a pair of state vectors :math:`u`
    and :math:`v` with weighting :math:`w` is defined as:

    .. math::
       \sqrt{\sum_{i=1}^{N}{w_i|(u_i - v_i)^2}}

    Note
    ----
    The EuclideanWeighted object has a property called weighting, which
    allows the method to be called on different pairs of states.
    If different weightings need to be used then multiple
    :class:`Measure` objects must be created with the specific weighting

    """
    weighting = Property(
        [np.array],
        doc="Weighting vector for the Euclidean calculation")

    def __call__(self, state1, state2):
        r"""Calculate the weighted Euclidean distance between a pair of state
        objects

        Parameters
        ----------
        state1 : :class:`~.State`
        state2 : :class:`~.State`

        Returns
        -------
        dist : float
            Weighted euclidean distance between two input
            :class:`~.State` objects

        """
        if self.mapping is not None:
            return distance.euclidean(state1.state_vector[self.mapping],
                                      state2.state_vector[self.mapping],
                                      self.weighting)
        else:
            return distance.euclidean(state1.state_vector,
                                      state2.state_vector,
                                      self.weighting)


class Mahalanobis(Measure):
    r"""Mahalanobis distance measure

    This measure returns the Mahalanobis distance between a pair of
    :class:`~.State` objects taking into account the distribution (i.e.
    the :class:`~.CovarianceMatrix`) of the first :class:`.State` object

    The Mahalanobis distance between a distribution with mean :math:`\mu` and
    Covariance matrix :math:`\Sigma` and a point :math:`x` is defined as:

    .. math::
            \sqrt{( {\mu - x})  \Sigma^{-1}  ({\mu - x}^T )}


    """
    def __call__(self, state1, state2):
        r"""Calculate the Mahalanobis distance between a pair of state objects

        Parameters
        ----------
        state1 : :class:`~.State`
        state2 : :class:`~.State`

        Returns
        -------
        float
            Mahalanobis distance between a pair of input :class:`~.State`
            objects

        """
        if self.mapping is not None:
            u = state1.state_vector[self.mapping]
            v = state2.state_vector[self.mapping]
            # extract the mapped covariance data
            rows = np.array(self.mapping, dtype=np.intp)
            columns = np.array(self.mapping, dtype=np.intp)
            cov = state1.covar[rows[:, np.newaxis], columns]
        else:
            u = state1.state_vector
            v = state2.state_vector
            cov = state1.covar

        vi = np.linalg.inv(cov)

        return distance.mahalanobis(u, v, vi)


class SquaredGaussianHellinger(Measure):
    r"""Squared Gaussian Hellinger distance measure

    This measure returns the Squared Hellinger distance between a pair of
    :class:`~.GaussianState` multivariate objects.

    The Squared Hellinger distance between two multivariate normal
    distributions :math:`P \sim N(\mu_1,\Sigma_1)` and
    :math:`Q \sim N(\mu_2,\Sigma_2)` is defined as:

    .. math::
            1 - \sqrt{\frac{det(\Sigma_1)^{1/4}det(\Sigma_2)^{1/4}}
            {det(\Sigma_1+\Sigma_2/2)^{1/2}}}
            exp\bigg(\frac{-1}{8}(\mu_1-\mu_2)^T
            (\frac{\Sigma_1+\Sigma_2}{2})^{-1}(\mu_1-\mu_2)\bigg)

    Note
    ----
    This distance is bounded between 0 and :math:`\sqrt{2}`
    """
    def __call__(self, state1, state2):
        r""" Calculate the Squared Hellinger distance multivariate normal
        distributions

        Parameters
        ----------
        state1 : :class:`~.GaussianState`
        state2 : :class:`~.GaussianState`

        Returns
        -------
        float
            Squared Hellinger distance between two input
            :class:`~.GaussianState`



        """
        if self.mapping is not None:
            mu1 = state1.state_vector[self.mapping]
            mu2 = state2.state_vector[self.mapping]

            # extract the mapped covariance data
            rows = np.array(self.mapping, dtype=np.intp)
            columns = np.array(self.mapping, dtype=np.intp)
            sigma1 = state1.covar[rows[:, np.newaxis], columns]
            sigma2 = state2.covar[rows[:, np.newaxis], columns]
        else:
            mu1 = state1.state_vector
            mu2 = state2.state_vector
            sigma1 = state1.covar
            sigma2 = state2.covar

        sigma1_plus_sigma2 = sigma1 + sigma2
        if np.linalg.det(sigma1_plus_sigma2) > 1e-10:
            mu1_minus_mu2 = mu1 - mu2
            E = mu1_minus_mu2.T*(np.linalg.inv(sigma1_plus_sigma2)
                                 / mu1_minus_mu2)
            epsilon = -0.125*E
            numerator = np.sqrt(np.linalg.det(sigma1*sigma2))
            denominator = np.linalg.det(sigma1_plus_sigma2/2)
            squared_hellinger = 1 - (
                np.sqrt(numerator/denominator)*np.exp(epsilon))
        else:
            mu1_minus_mu2 = mu1 - mu2
            temp = mu1_minus_mu2.T * mu1_minus_mu2
            squared_hellinger = temp
        squared_hellinger = squared_hellinger[0, 0]
        return squared_hellinger


class GaussianHellinger(SquaredGaussianHellinger):
    r"""Gaussian Hellinger distance measure

    This measure returns the Hellinger distance between a pair of
    :class:`~.GaussianState` multivariate objects.

    The Hellinger distance between two multivariate normal distributions
    :math:`P \sim N(\mu_1,\Sigma_1)` and :math:`Q \sim N(\mu_2,\Sigma_2)`
    is defined as:

    .. math::
            \sqrt{1 - \sqrt{\frac{det(\Sigma_1)^{1/4}det(\Sigma_2)^{1/4}}
            {det(\Sigma_1+\Sigma_2/2)^{1/2}}}
            exp\bigg(\frac{-1}{8}(\mu_1-\mu_2)^T
            (\frac{\Sigma_1+\Sigma_2}{2})^{-1}(\mu_1-\mu_2)\bigg)}

    Note
    ----
    This distance is bounded between 0 and 1
    """
    def __call__(self, state1, state2):
        r""" Calculate the Hellinger distance between 2 state elements

        Parameters
        ----------
        state1 : :class:`~.GaussianState`
        state2 : :class:`~.GaussianState`

        Returns
        -------
        float
            Hellinger distance between two input :class:`~.GaussianState`


        """
        return np.sqrt(super().__call__(state1, state2))
