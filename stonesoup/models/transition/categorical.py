# -*- coding: utf-8 -*-

from datetime import timedelta

import numpy as np
from scipy.stats import multinomial

from .base import Property
from ...models.transition import TransitionModel
from ...types.array import Matrix, StateVector


class MarkovianTransitionModel(TransitionModel):
    r"""The transition model for categorical states

    This is a time invariant, transition model of a Markov process.

    A state space vector takes the form :math:`\alpha_t^i = P(\phi_t^i)`, representing a
    categorical distribution over a discrete, finite set of possible categories
    :math:`\Phi = \{\phi^m|m\in \mathbf{N}, m\le M\}` (for some finite :math:`M`).

    Models the transition from one category to another.

    Intended to be used in conjunction with the :class:`~.CategoricalState` type.
    """
    transition_matrix: Matrix = Property(
        doc=r"Stochastic matrix :math:`F_t^{ij} = F^{ij} = P(\phi_t^i|\phi_{t-1}^j)` determining "
            r"the conditional probability that an object is category :math:`\phi^i` at 'time' "
            r":math:`t` given that it was category :math:`\phi^j` at 'time' :math:`t-1`. "
            r"Columns are normalised.")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Normalise matrix columns
        self.transition_matrix = self.transition_matrix / np.sum(self.transition_matrix, axis=0)

    def function(self, state, time_interval: timedelta = None, noise: bool = False, **kwargs):
        r"""Applies the linear transformation:

        .. math::
            F^{ij}\alpha_{t-1}^j = P(\phi_t^i|\phi_{t-1}^j)P(\phi_t^j)

        The resultant vector is normalised.

        Though this model is time-invariant, a check is made to see whether the time-interval given
        is 0. In this instance, no transformation is applied.

        Parameters
        ----------
        state: :class:`~.CategoricalState`
            The state to be transitioned.
        time_interval: datetime.timedelta
            Duration to transition state for.
        noise: bool
            Indicates whether transitioned vector is sampled from and the resultant category
            returned instead. This is a discrete category instead of a distribution
            over the state space. It is represented by an M-tuples, with all components
            equal to 0, except at an index corresponding to the relevant category.
            For example :math:`e^k` indicates that the category is :math:`\phi^k`.
            If `False`, the resultant distribution is returned.

        Returns
        -------
        state_vector: :class:`stonesoup.types.array.StateVector`
            of shape (:py:attr:`~ndim_state, 1`). The resultant state vector of the transition.
        """

        if time_interval is None or time_interval.total_seconds() == 0:
            return state.state_vector

        new_vector = self.transition_matrix @ state.state_vector
        new_vector = new_vector / np.sum(new_vector)  # normalise

        if noise:
            rv = multinomial(n=1, p=new_vector.flatten())
            return StateVector(rv.rvs(size=1, random_state=None))
        else:
            return StateVector(new_vector)

    @property
    def ndim_state(self):
        return self.transition_matrix.shape[1]

    def rvs(self):
        raise NotImplementedError

    def pdf(self):
        raise NotImplementedError
