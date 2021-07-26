# -*- coding: utf-8 -*-
from typing import Union

import numpy as np

from .base import Property
from ...models.transition import TransitionModel
from ...types.array import Matrix, StateVector, StateVectors, CovarianceMatrix


class CategoricalTransitionModel(TransitionModel):
    r"""The transition model for categorical data
    This is a time invariant model for the transition of a state that can take one of a finite
    number of categories :math:`\{\phi_m|m\in\Z_{\gt0}\}`, where a state space vector takes the
    form :math:`x_{k_i} = P(\phi_i, k)`, i.e. the :math:`i`-th state vector component is the
    probability that the state is of category :math:`\phi_i` at 'time' :math:`k`.
    """
    transition_matrix: Matrix = Property(
        doc=r"Stochastic matrix :math:`(F_{k+1})_{ij} = P(\phi_i, k+1 | \phi_j, k)` determining "
            r"the conditional probability that an object is category :math:`\phi_i` at 'time'"
            r":math:`k + 1` given that it was category :math:`\phi_j` at 'time' :math:`k`."
            r"Columns must sum to 1.")
    transition_covariance: CovarianceMatrix = Property(
        default=None,
        doc="Transition covariance, used in noise generation.")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for i, col in enumerate(self.transition_matrix.T):
            if not np.isclose(np.sum(col), 1):
                raise ValueError(f"Column {i} of transition matrix does not sum to 1")

    @property
    def ndim_state(self):
        """Number of model state dimensions/categories."""
        return self.transition_matrix.shape[0]

    def function(self, state, noise: bool = False, **kwargs) -> StateVector:
        r"""Applies the linear transformation:

        .. math::
            (F_{k+1}\mathbf{x}_k)_{i} = P(\phi_i, k+1 | \phi_j, k)P(\phi_j, k)

        The resultant vector is transformed to the interval :math:`[-\infty, \infty]` via a logit
        function, in order to add noise.
        This is then transformed back to the interval :math:`[0, 1]` and normalised.

        Parameters
        ----------
        state : :class:`~.CategoricalState`
            The state to be transitioned according to the models in :py:attr:`~model_list`.
        noise: bool
            If 'True', additive log noise (generated via random sampling) will be included prior
            to transformation back to the interval :math:`[0, 1]`. Noise vectors cannot be defined
            beforehand. Only a boolean value is valid.

        Returns
        -------
        state_vector: :class:`stonesoup.types.array.StateVector`
            of shape (:py:attr:`~ndim_state, 1`). The resultant state vector of the transition.

        Notes
        -----
        For p=0 or p=1 infinities will be generated. We manage that by using the `numpy.finfo()`
        function.
        """
        if isinstance(noise, bool) and noise:
            noise = self.rvs()
        elif not noise:
            noise = 0
        else:
            raise ValueError("Noise is generated via random sampling, and defined noise is not "
                             "implemented")

        # what to do if p=1
        fp = self.transition_matrix @ state.state_vector

        with np.errstate(divide='ignore', over='ignore', under='ignore'):
            if any(fp == 1):
                y = fp.astype(float)
                y[fp == 1] = np.finfo(np.float64).max
                y[fp == 0] = np.finfo(np.float64).min
            else:
                y = np.log(fp / (1 - fp))

            y += noise

            p = 1 / (1 + np.exp(-y))
            return p / np.sum(p)

    def rvs(self, num_samples=1, **kwargs) -> Union[StateVector, StateVectors]:
        """Create noise samples. This just samples from a multivariate normal distribution with
        mean=0 and covariance defined by the transition covariance matrix. This is additive log
        noise, so multiplicative but with its effect largest at p=0.5 (logit(p)=0) and nil when
        p=0 (logit(p)=-inf, or p=1 (logit(p)=+inf. One could 'normalise' this such that the effect
        is more constant across p but it's impossible to escape the points that map to infinity in
        the logit function, so p = 1 + noise will still be p = 1 (+etc), unless a different form
        is used.
        """
        omega = np.random.multivariate_normal(np.zeros(self.ndim_state),
                                              self.transition_covariance,
                                              size=num_samples)
        return StateVectors(omega).T

    def pdf(self, state1, state2, **kwargs):
        """Assumes that state 1 is binary and this returns the probability that the transited
        state2 is state1"""
        Fx = self.transition_matrix @ state2.state_vector
        return Fx.T @ state1.state_vector
