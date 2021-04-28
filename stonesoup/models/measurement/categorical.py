# -*- coding: utf-8 -*-
from typing import Union, Sequence

import numpy as np
from scipy.stats import multinomial

from .base import MeasurementModel
from ...base import Property
from ...types.array import Matrix, StateVector, StateVectors, CovarianceMatrix


class CategoricalMeasurement(MeasurementModel):
    r"""Measurement model which returns a category.
    This is a time invariant model for simple observations of a state.
    A measurement can take one of a finite number of observation categories
    :math:`\{y_k|k\in\Z_{\gt0}\}` and a measurement vector :math:`(z_k)_i = P(y_i, k)` will define
    a categorical distribution over these categories. Measurements are generated via random
    sampling.
    """
    emission_matrix: Matrix = Property(
        doc=r"The emission matrix :math:`(E_k)_{ij} = P(z_{i}, k | \phi_{j}, k)`, defining the "
            r"probability of getting an observation "
            r":math:`z` from state :math:`x_{k_i} = P(\phi_i, k)`. "
            r"Rows of the matrix must sum to 1")
    emission_covariance: CovarianceMatrix = Property(default=None, doc="Emission covariance")
    mapping: Sequence[int] = Property(default=None,
                                      doc="Mapping between measurement and state dims")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        for i, row in enumerate(self.emission_matrix):
            if sum(row) != 1:
                raise ValueError(f"Row {i} of emission matrix does not sum to 1")

        if self.mapping is None:
            self.mapping = np.arange(self.ndim_state)
        elif len(self.mapping) != np.shape(self.emission_matrix)[0]:
            raise ValueError(f"Emission matrix maps from {np.shape(self.emission_matrix)[0]} "
                             f"elements of the state space, but the mapping is length "
                             f"{len(self.mapping)}")

    @property
    def ndim_meas(self):
        """Number of observation dimensions/categories."""
        return self.emission_matrix.shape[1]

    def _cond_prob_emission(self, state, noise=False, **kwargs):
        """This function returns the probability of each observation category conditioned on the
        input state, :math:`(p(z_j|x_i) p(x_i)` (this should come out normalised).
        Noise is additive."""

        if type(noise) is bool and noise:
            noise = self.rvs()
        elif not noise:
            noise = 0

        prenormalised = self.emission_matrix.T @ state.state_vector[self.mapping] + noise
        return prenormalised / np.sum(prenormalised)

    def function(self, state, noise=False, **kwargs):
        """Observation function

        Parameters
        ----------
        state: :class:`~.State`
            An input (hidden class) state, where the state vector defines a categorical
            distribution over the set of possible hidden states :math:`\{\phi_m|m\in\Z_{\gt0}\}`
        noise: bool
            If 'True', additive noise (generated via random sampling) will be included. Noise
            vectors cannot be defined beforehand. Only a boolean value is valid.

        Returns
        -------
        :class:`numpy.ndarray` of shape (:py:attr:`~ndim_meas`, 1)
            The observer function evaluated. The resultant vector represents a categorical
            distribution over the set of possible measurement categories. A definitive measurement
            category is chosen, hence the vector will be binary.
        """

        rv = multinomial(n=1, p=self._cond_prob_emission(state, noise=False, **kwargs).flatten())
        return StateVector(rv.rvs(size=1, random_state=None))

    def rvs(self, num_samples=1, **kwargs) -> Union[StateVector, StateVectors]:
        """Additive noise."""
        omega = np.random.multivariate_normal(np.zeros(self.ndim),
                                              self.emission_covariance,
                                              size=num_samples)
        return StateVectors(omega).T

    def pdf(self, state1, state2, **kwargs):
        """Assumes that state 1 is a binary measurement state and this returns the probability
        that the emission of state2 is state1"""
        Hx = self._cond_prob_emission(state2)
        return Hx.T @ state1.state_vector
