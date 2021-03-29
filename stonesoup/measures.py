# -*- coding: utf-8 -*-
from abc import abstractmethod

import numpy as np
from scipy.spatial import distance

from stonesoup.types.state import State
from .base import Base, Property


class Measure(Base):
    """Measure base type

    A measure provides a means to assess the seperation between two
    :class:`~.State` objects state1 and state2.
    """
    mapping: np.ndarray = Property(
        default=None,
        doc="Mapping array which specifies which elements within the"
            " state vectors are to be assessed as part of the measure"
    )
    mapping2: np.ndarray = Property(
        default=None,
        doc="A second mapping for when the states being compared exist "
            "in different parameter spaces. Defaults to the same as the"
            " first mapping"
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.mapping2 is not None and self.mapping is None:
            raise ValueError("Cannot set mapping2 if mapping is None. "
                             "If this is really what you meant to do, then"
                             " set mapping to include all dimensions.")
        if self.mapping2 is None and self.mapping is not None:
            self.mapping2 = self.mapping

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
            return distance.euclidean(state1.state_vector[self.mapping, :],
                                      state2.state_vector[self.mapping2, :])
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
    weighting: np.ndarray = Property(doc="Weighting vector for the Euclidean calculation")

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
            return distance.euclidean(state1.state_vector[self.mapping, :],
                                      state2.state_vector[self.mapping2, :],
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
            u = state1.state_vector[self.mapping, :]
            v = state2.state_vector[self.mapping2, :]
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
            H^{2}(P, Q) &= 1 - {\frac{\det(\Sigma_1)^{\frac{1}{4}}\det(\Sigma_2)^{\frac{1}{4}}}
            {\det\left(\frac{\Sigma_1+\Sigma_2}{2}\right)^{\frac{1}{2}}}}
            \exp\left(-\frac{1}{8}(\mu_1-\mu_2)^T
            \left(\frac{\Sigma_1+\Sigma_2}{2}\right)^{-1}(\mu_1-\mu_2)\right)\\
            &\equiv  1 - \sqrt{\frac{\det(\Sigma_1)^{\frac{1}{2}}\det(\Sigma_2)^{\frac{1}{2}}}
            {\det\left(\frac{\Sigma_1+\Sigma_2}{2}\right)}}
            \exp\left(-\frac{1}{8}(\mu_1-\mu_2)^T
            \left(\frac{\Sigma_1+\Sigma_2}{2}\right)^{-1}(\mu_1-\mu_2)\right)

    Note
    ----
    This distance is bounded between 0 and 1
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
            mu1 = state1.state_vector[self.mapping, :]
            mu2 = state2.state_vector[self.mapping2, :]

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
        mu1_minus_mu2 = mu1 - mu2
        E = mu1_minus_mu2.T @ np.linalg.inv(sigma1_plus_sigma2/2) @ mu1_minus_mu2
        epsilon = -0.125*E
        numerator = np.sqrt(np.linalg.det(sigma1 @ sigma2))
        denominator = np.linalg.det(sigma1_plus_sigma2/2)
        squared_hellinger = 1 - np.sqrt(numerator/denominator)*np.exp(epsilon)
        squared_hellinger = squared_hellinger.item()
        return squared_hellinger


class GaussianHellinger(SquaredGaussianHellinger):
    r"""Gaussian Hellinger distance measure

    This measure returns the Hellinger distance between a pair of
    :class:`~.GaussianState` multivariate objects.

    The Hellinger distance between two multivariate normal distributions
    :math:`P \sim N(\mu_1,\Sigma_1)` and :math:`Q \sim N(\mu_2,\Sigma_2)`
    is defined as:

    .. math::
            H(P,Q) = \sqrt{1 - {\frac{\det(\Sigma_1)^{\frac{1}{4}}\det(\Sigma_2)^{\frac{1}{4}}}
            {\det\left(\frac{\Sigma_1+\Sigma_2}{2}\right)^{\frac{1}{2}}}}
            \exp\left(-\frac{1}{8}(\mu_1-\mu_2)^T
            \left(\frac{\Sigma_1+\Sigma_2}{2}\right)^{-1}(\mu_1-\mu_2)\right)}

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


class Accuracy(Measure):
    def __call__(self, state1, state2):

        if isinstance(state1, State):
            s1 = state1.state_vector
        else:
            s1 = state1

        if isinstance(state2, State):
            s2 = state2.state_vector
        else:
            s2 = state2

        mins = [min(s1, s2) for s1, s2 in zip(s1, s2)]
        maxs = [max(s1, s2) for s1, s2 in zip(s1, s2)]
        return np.sum(mins)/np.sum(maxs)
