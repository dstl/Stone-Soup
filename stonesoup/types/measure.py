# -*- coding: utf-8 -*-
import numpy as np

from ..base import Property
from .base import Type


class Measure(Type):
    """Measure base type

    A measure provides a means to assess the seperation between two
    :class:'~.State' objects X and Y.
    """
    threshold = Property(
        float,
        default=None,
        doc="Threshold to be applied to distance measure")
    # TODO determine if threshold is a generic property or measure specific
    mapping = Property(
        [np.array],
        default=None,
        doc="Mapping array which specifies which elements within the state \
             vectors are to be assessed as part of the measure"
    )

    def get_dist(self, x, y):
        """
        Compute the distance between a pair of :class:'~.State'

        Parameters
        ----------
        x : :class:'~.State'
        y : :class:'~.State'

        Returns
        -------
            distance measure

        """
        return NotImplemented


class Euclidean(Measure):
    r"""Euclidean distance measure\

    This measure provides the means to measure the euclidean distance between\
    the :class:'~StateVector' information within a pair of :class:'~.State'\
    objects.\

    The Euclidean distance is defined as:
    .. math::

        d = \sqrt{\sum{\mathcal{x-y}^2}}
    """
    def get_dist(self, x, y):
        """Calculate the Euclidean distance between state vector elements
        indicated by the mapping.

        :param x::class:'~.State'
        :param y::class:'~.State'
        :return: distance
        """
        # Calculate Euclidean distance between two state
        if self.mapping is None:
            dist = np.linalg.norm(x.state_vector[self.mapping] -
                                  y.state_vector[self.mapping])
        else:
            dist = np.linalg.norm(x.state_vector - y.state_vector)

        if (self.threshold is None) or (dist < self.threshold):
            return dist
        else:
            return None


class Mahalanobis(Measure):
    r"""Mahalanobis distance measure

    This measure returns the Mahalanobis distance between a pair of
     :class:'~.State' objects taking into account the distribution (i.e. the
     :class:'~.CovarianceMatrix') of the first :class:'~.State'

    The Mahalanobis distance is defined as:
    .. math::
        d = \sqrt{{\mathcal{x-y}^T} * \Sigma^-1 * {\mathcal{x-y}}}

    Note
    ----
    The Covariance used in this calculation is taken from the first state
    """
    def get_dist(self, x, y):
        """ Calculate the Mahalanobis distance between 2 state elements

        :param x::class:'~.State'
        :param y::class:'~.State'
        :return: distance
        """
        if self.mapping is None:
            d = x.state_vector[self.mapping] - y.state_vector[self.mapping]
            # TODO modify covariance to reflect mapping
            cov = x.covar
        else:
            d = x.state_vector - y.state_vector
            cov = x.covar

        cov_inv = np.linalg.inv(x.covar)
        dist = np.sqrt(np.einsum('nj,jk,nk->n', d, cov_inv, d))

        return dist


