# -*- coding: utf-8 -*-
from typing import Union, Sequence

import numpy as np
from scipy.stats import multinomial

from .base import MeasurementModel
from ...base import Property
from ...types.array import Matrix, StateVector, StateVectors, CovarianceMatrix


class CategoricalMeasurementModel(MeasurementModel):
    r"""Measurement model which returns a category.
    This is a time invariant model for simple observations of a state.
    A measurement can take one of a finite number of observation categories
    :math:`Y = \{y_k|k\in Z_{\gt0}\}` and a measurement vector :math:`(z_k)_i = P(y_i, k)` will
    define a categorical distribution over these categories. Measurements are generated via random
    sampling.

    Intended to be used in conjunction with the :class:`~.CategoricalState` type.
    """
    emission_matrix: Matrix = Property(
        doc=r"The emission matrix defining emission probability "
            r":math:`(E_k)_{ij} = P(z_{j}, k | \phi_{i}, k)` (the probability of receiving an "
            r"observation :math:`z` from state :math:`x_{k_j} = P(\phi_j, k)`. "
            r"Rows of the matrix must sum to 1.")
    emission_covariance: CovarianceMatrix = Property(doc="Emission covariance.")
    mapping: Sequence[int] = Property(default=None,
                                      doc="Mapping between measurement and state dims.")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        for i, row in enumerate(self.emission_matrix):
            if not np.isclose(np.sum(row), 1):
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
        Noise is additive, and used by transforming resultant vectors using a logit function."""

        if type(noise) is bool and noise:
            noise = self.rvs()
        elif not noise:
            noise = 0
        else:
            raise ValueError("Noise is generated via random sampling, and defined noise is not "
                             "implemented")

        hp = self.emission_matrix.T @ state.state_vector[self.mapping]

        with np.errstate(divide='ignore', over='ignore', under='ignore'):
            if any(hp == 1):
                y = hp.astype(float)
                y[hp == 1] = np.finfo(np.float64).max
                y[hp == 0] = np.finfo(np.float64).min
            else:
                y = np.log(hp / (1 - hp))

            y += noise

            p = 1 / (1 + np.exp(-y))
            return p / np.sum(p)

    def function(self, state, noise=False, **kwargs):
        r"""Observation function

        Parameters
        ----------
        state: :class:`~.CategoricalState`
            An input (hidden class) state, where the state vector defines a categorical
            distribution over the set of possible hidden states
            :math:`\Phi = \{\phi_m|m\in Z_{\gt0}\}`.
        noise: bool
            If 'True', additive noise (generated via random sampling) will be included. Noise
            vectors cannot be defined beforehand. Only a boolean value is valid.

        Returns
        -------
        state_vector: :class:`stonesoup.types.array.StateVector`
            of shape (:py:attr:`~ndim_meas, 1`). The resultant measurement vector.
            The resultant vector represents a categorical distribution over the set of possible
            measurement categories. A definitive measurement category is chosen, hence the vector
            will be binary.
        """
        cond_prob_emission = self._cond_prob_emission(state, noise=noise, **kwargs)
        rv = multinomial(n=1, p=cond_prob_emission.flatten())
        if noise:
            return StateVector(rv.rvs(size=1, random_state=None))
        else:
            return StateVector(cond_prob_emission)

    def rvs(self, num_samples=1, **kwargs) -> Union[StateVector, StateVectors]:
        """Additive noise."""
        omega = np.random.multivariate_normal(np.zeros(self.ndim),
                                              self.emission_covariance,
                                              size=num_samples)
        return StateVectors(omega).T

    def pdf(self, state1, state2, **kwargs):
        """Assumes that state 1 is a binary measurement state (i.e. one vector element is 1 and
        the rest are zeroes).
        Returns the probability that the emission of state 2 is state 1."""
        Hx = self._cond_prob_emission(state2)
        return Hx.T @ state1.state_vector
