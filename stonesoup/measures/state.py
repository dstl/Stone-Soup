import copy
from abc import abstractmethod
from functools import lru_cache

import numpy as np
from scipy.spatial import distance

from .base import BaseMeasure
from ..base import Property
from ..types.state import State, ParticleState, GaussianState


class Measure(BaseMeasure):
    """Measure base type

    A measure provides a means to assess the separation between two
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
        state_vector1 = getattr(state1, 'mean', state1.state_vector)
        state_vector2 = getattr(state2, 'mean', state2.state_vector)

        if self.mapping is not None:
            return distance.euclidean(state_vector1[self.mapping, 0],
                                      state_vector2[self.mapping2, 0])
        else:
            return distance.euclidean(state_vector1[:, 0], state_vector2[:, 0])


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
        state_vector1 = getattr(state1, 'mean', state1.state_vector)
        state_vector2 = getattr(state2, 'mean', state2.state_vector)

        if self.mapping is not None:
            return distance.euclidean(state_vector1[self.mapping, 0],
                                      state_vector2[self.mapping2, 0],
                                      self.weighting)
        else:
            return distance.euclidean(state_vector1[:, 0],
                                      state_vector2[:, 0],
                                      self.weighting)


class SquaredMahalanobis(Measure):
    r"""Squared Mahalanobis distance measure

    This measure returns the Squared Mahalanobis distance between a pair of
    :class:`~.State` objects taking into account the distribution (i.e.
    the :class:`~.CovarianceMatrix`) of the first :class:`.State` object

    The Squared Mahalanobis distance between a distribution with mean :math:`\mu`
    and Covariance matrix :math:`\Sigma` and a point :math:`x` is defined as:

    .. math::
            ( {\mu - x})  \Sigma^{-1}  ({\mu - x}^T )


    """
    state_covar_inv_cache_size: int = Property(
        default=128,
        doc="Number of covariance matrix inversions to cache. Setting to `0` will disable the "
            "cache, whilst setting to `None` will not limit the size of the cache. Default is "
            "128.")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.state_covar_inv_cache_size is None or self.state_covar_inv_cache_size > 0:
            self._inv_cov = lru_cache(maxsize=self.state_covar_inv_cache_size)(self._inv_cov)

    def __getstate__(self):
        result = copy.copy(self.__dict__)
        result["_inv_cov"] = None
        return result

    def __setstate__(self, state):
        self.__dict__ = state
        if self.state_covar_inv_cache_size is None or self.state_covar_inv_cache_size > 0:
            self._inv_cov = lru_cache(maxsize=self.state_covar_inv_cache_size)(type(self)._inv_cov)
        else:
            self._inv_cov = type(self)._inv_cov

    def __call__(self, state1, state2):
        r"""Calculate the Squared Mahalanobis distance between a pair of state objects

        Parameters
        ----------
        state1 : :class:`~.State`
        state2 : :class:`~.State`

        Returns
        -------
        float
            Squared Mahalanobis distance between a pair of input :class:`~.State`
            objects

        """
        state_vector1 = getattr(state1, 'mean', state1.state_vector)
        state_vector2 = getattr(state2, 'mean', state2.state_vector)

        if self.mapping is not None:
            u = state_vector1[self.mapping, 0]
            v = state_vector2[self.mapping2, 0]
            # extract the mapped covariance data
            vi = self._inv_cov(state1, tuple(self.mapping))
        else:
            u = state_vector1[:, 0]
            v = state_vector2[:, 0]
            vi = self._inv_cov(state1)

        delta = u - v

        return np.dot(np.dot(delta, vi), delta)

    @staticmethod
    def _inv_cov(state, mapping=None):
        if mapping:
            rows = np.array(mapping, dtype=np.intp)
            columns = np.array(mapping, dtype=np.intp)
            covar = state.covar[rows[:, np.newaxis], columns]
        else:
            covar = state.covar

        return np.linalg.inv(covar)


class Mahalanobis(SquaredMahalanobis):
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
        return np.sqrt(super().__call__(state1, state2))


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
        if hasattr(state1, 'mean'):
            state_vector1 = state1.mean
        else:
            state_vector1 = state1.state_vector

        if hasattr(state2, 'mean'):
            state_vector2 = state2.mean
        else:
            state_vector2 = state2.state_vector

        if self.mapping is not None:
            mu1 = state_vector1[self.mapping, :]
            mu2 = state_vector2[self.mapping2, :]

            # extract the mapped covariance data
            rows = np.array(self.mapping, dtype=np.intp)
            columns = np.array(self.mapping, dtype=np.intp)
            sigma1 = state1.covar[rows[:, np.newaxis], columns]
            sigma2 = state2.covar[rows[:, np.newaxis], columns]
        else:
            mu1 = state_vector1
            mu2 = state_vector2
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

        if -1e-10 < squared_hellinger < 0.0:
            squared_hellinger = 0.0
        elif squared_hellinger < 0.0:  # pragma: no cover
            raise ValueError("Measure shouldn't be less than 0")  # this should be impossible

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


class ObservationAccuracy(Measure):
    r"""Accuracy measure

    This measure evaluates the accuracy of a categorical distribution with respect to another."""

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


class KLDivergence(Measure):
    r"""Kullback-Leibler divergence between two distributions

    Kullback-Leibler divergence, also referred to as relative entropy, is a
    statistical distance. It describes how a probability distribution is
    different from another. The expression for Kullback-Leibler divergence
    is given by [1]_

    .. math::
        D_{KL}(P\Vert Q) = \sum_x P(x)\log \frac{P(x)}{Q(x)},

    where :math:`P(x)` is the first distribution, or ``state1`` and :math:`Q(x)`
    is the second distribution or, ``state2``. It is worth noting that Kullback-Leibler
    divergence is not symmetric under interchange of :math:`P(x)` and :math:`Q(x)`. The
    implementation here uses natural log meaning the returned divergence has units in nats.
    This implementation assumes a discrete probability space and currently only accepts
    :class:`~.ParticleState`.

    References
    ----------
    .. [1] MacKay, David J. C. 2003. Information Theory, Inference and Learning
       Algorithms, 1st Ed. Cambridge University Press, """

    def __call__(self, state1, state2):
        r"""
        Computes the Kullback–Leibler divergence from ``state1`` to ``state2``

        Parameters
        ----------
        state1 : :class:`~.ParticleState`
        state2 : :class:`~.ParticleState`

        Returns
        -------
        float
            Kullback–Leibler divergence from ``state1`` to ``state2``

        """
        if isinstance(state1, ParticleState) and isinstance(state2, ParticleState):
            if len(state1) == len(state2):

                log_term = np.zeros(state1.log_weight.shape)

                invalid_indx = (np.isinf(state1.log_weight) | np.isnan(state1.log_weight)
                                | np.isinf(state2.log_weight) | np.isnan(state2.log_weight))

                # Do not consider NANs and inf in the subtraction
                log_term[~invalid_indx] = state1.log_weight[~invalid_indx] \
                    - state2.log_weight[~invalid_indx]

                kld = np.sum(np.exp(state1.log_weight)*log_term)
            else:
                raise ValueError(f'The input sizes are not compatible '
                                 f'({len(state1)} != {len(state2)})')

        elif isinstance(state1, GaussianState) and isinstance(state2, GaussianState):

            state1 = copy.copy(state1)
            state2 = copy.copy(state2)

            if self.mapping is not None:
                state1.state_vector = state1.state_vector[self.mapping, :]
                state2.state_vector = state2.state_vector[self.mapping2, :]

                rows = np.array(self.mapping, dtype=np.intp)
                columns = np.array(self.mapping, dtype=np.intp)
                state1.covar = state1.covar[rows[:, np.newaxis], columns]

                rows2 = np.array(self.mapping2, dtype=np.intp)
                columns2 = np.array(self.mapping2, dtype=np.intp)
                state2.covar = state2.covar[rows2[:, np.newaxis], columns2]

            if state1.ndim == state2.ndim:

                log_term = np.log(np.linalg.det(state2.covar) / np.linalg.det(state1.covar))

                n_dims = state1.ndim

                inv_state2_covar = np.linalg.inv(state2.covar)
                trace_term = np.trace(inv_state2_covar@state1.covar)

                delta = state2.state_vector - state1.state_vector
                mahalanobis_term = delta.T @ inv_state2_covar @ delta

                kld = float(0.5*(log_term - n_dims + trace_term + mahalanobis_term))

            else:
                raise ValueError(f'The state dimensions are not compatible '
                                 f'({state1.ndim} != {state2.ndim}')

        else:
            raise NotImplementedError('This measure is currently only compatible with '
                                      'ParticleState or GaussianState types')

        return kld
