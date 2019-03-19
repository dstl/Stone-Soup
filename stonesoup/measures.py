# -*- coding: utf-8 -*-
from abc import abstractmethod

import numpy as np
from scipy.spatial import distance

from stonesoup.base import Property
from .base import Base


class Measures(Base):
    """Measure base type

    A measure provides a means to assess the seperation between two
    :class:`~.State` objects state1 and state2.
    """
    mapping = Property(
        np.array,
        default=None,
        doc="Mapping array which specifies which elements within the state \
             vectors are to be assessed as part of the measure"
    )

    @abstractmethod
    def __call__(self, state1, state2):
        r"""
        Compute the distance between a pair of :class:`~.State`

        Parameters
        ----------
        state1 : :class:`~.State`
        state2 : :class:`~.State`

        Returns
        -------
        float
            distance measure between input :class:`~.State`

        """
        return NotImplementedError


class Euclidean(Measures):
    r"""Euclidean distance measure\

    This measure returns the euclidean distance between a pair of
    :class:`~StateVector` information within a pair of :class:`~.State`\
    objects. \

    The Euclidean distance is defined as:

    .. math::
         \sqrt{\sum_{n=1}^{N}{(u_i - v_i)^2}}

    """
    def __call__(self, state1, state2):
        r"""Calculate the Euclidean distance between state vector elements
        indicated by the mapping.

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


class EuclideanWeighted(Measures):
    r"""Weighted Euclidean distance measure\

    This measure returns the euclidean distance between\
    the :class:`~StateVector` information within a pair of :class:`~.State`\
    objects, taking into account a specified weighting for the elements under\
    consideration.

    The Weighted Euclidean distance is defined as:

    .. math::
       \sqrt{\sum_{n=1}^{N}{w_i|(u_i - v_i)^2}}

    Note
    ----
    The EuclideanWeighted object has a property called weighting which allows \
    the method to be called repeatably on different pairs of state vectors. If\
    different weightings need to be used then currently multiple objects must\
    be specified

    """
    weighting = Property(
        [np.array],
        doc="Weighting vector for the euclidean calculation")

    def __call__(self, state1, state2):
        r"""Calculate the weighted Euclidean distance between state vector
        elements indicated by the mapping

        Parameters
        ----------
        state1 : :class:`~.State`
        state2 : :class:`~.State`

        Returns
        -------
        dist : float
            Weighted euclidean distance between two input :class:`~.State`

        """
        if self.mapping is not None:
            return distance.euclidean(state1.state_vector[self.mapping],
                                      state2.state_vector[self.mapping],
                                      self.weighting)
        else:
            return distance.euclidean(state1.state_vector,
                                      state2.state_vector,
                                      self.weighting)


class Mahalanobis(Measures):
    r"""Mahalanobis distance measure

    This measure returns the Mahalanobis distance between a pair of
     :class:`~.State` objects taking into account the distribution (i.e. the
     :class:`~.CovarianceMatrix`) of the first :class:`.State`

    The Mahalanobis distance is defined as:

    .. math::
        D_{M}({x}) = \sqrt{( {x - y} * \Sigma^{-1} * {x - y}^T )}

    Note
    ----
    The Covariance used in this calculation is taken from the first state

    """
    def __call__(self, state1, state2):
        r"""Calculate the Mahalanobis distance between 2 state elements

        Parameters
        ----------
        state1 : :class:`~.State`
        state2 : :class:`~.State`

        Returns
        -------
        float
            Mahalanobis distance between two input :class:`~.State`

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


class SquaredHellinger(Measures):
    r"""Squared Hellinger distance measure

    This measure returns the Squared Hellinger distance between a pair of
     :class:`~.GaussianState` multivariate objects.

    The Squared Hellinger distance between two multivariate normal
    distributions :math:`P ~ N(\mu_1,\Sigma_1)` and
    :math:`Q ~ N(\mu_2,\Sigma_2)` is defined as:

    .. math::
        H^2(P,Q) = 1 - \sqrt{det(\Sigma_1)^{1/4}det(\Sigma_2)^{1/4}
        /det(\Sigma_1+\Sigma_2/2)^{1/2}}
        exp{-1/8(\mu_1-\mu_2)^T(\Sigma_1+\Sigma_2/2)^-1(\mu_1-\mu_2)}

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

        Note
        ----
        This distance is bounded between 0 and :math:`\sqrt{2}`

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


class Hellinger(SquaredHellinger):
    r"""Hellinger distance measure

    This measure returns the Hellinger distance between a pair of
     :class:`~.GaussianState` multivariate objects.

    The Hellinger distance between two multivariate normal distributions
    :math:`P ~ N(\mu_1,\Sigma_1)` and :math:`Q ~ N(\mu_2,\Sigma_2)` is defined
    as:

    .. math::
        H(P,Q) = \sqrt{1 - \sqrt{det(\Sigma_1)^{1/4}det(\Sigma_2)^{1/4}
        /det(\Sigma_1+\Sigma_2/2)^{1/2}}
        exp{-1/8(\mu_1-\mu_2)^T(\Sigma_1+\Sigma_2/2)^-1(\mu_1-\mu_2)}}

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

        Note
        ----
        This distance is bounded between 0 and 1

        """
        return np.sqrt(super().__call__(state1, state2))
