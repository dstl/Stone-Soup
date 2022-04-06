# -*- coding: utf-8 -*-
from typing import Sequence

import numpy as np

from ..measurement import MeasurementModel
from ...base import Property
from ...types.array import Matrix, StateVector


class MarkovianMeasurementModel(MeasurementModel):
    r"""The measurement model for categorical states

    This is a time invariant, measurement model of a hidden Markov process.

    A measurement can take one of a finite number of observable categories
    :math:`Z = \{\zeta^n|n\in \mathbf{N}, n\le N\}` (for some finite :math:`N`). A measurement
    vector represents a categorical distribution over :math:`Z`.

    .. math::
        \mathbf{y}_t^i = P(\zeta_t^i)

    A state space vector takes the form :math:`\alpha_t^i = P(\phi_t^i)`, representing a
    categorical distribution over a discrete, finite set of possible categories
    :math:`\Phi = \{\phi^m|m\in \mathbf{N}, m\le M\}` (for some finite :math:`M`).

    It is assumed that a measurement is independent of everything but the true state of a target.

    Intended to be used in conjunction with the :class:`~.CategoricalState` type.
    """
    emission_matrix: Matrix = Property(
        doc=r"Matrix of emission/output probabilities "
            r":math:`E_t^{ij} = E^{ij} = P(\zeta_t^i | \phi_t^j)`, determining the conditional "
            r"probability that a measurement is category :math:`\zeta^i` at 'time' :math:`t` "
            r"given that the true state category is :math:`\phi^j` at 'time' :math:`t`. "
            r"Columns will be normalised.")
    measurement_categories: Sequence[str] = Property(doc="Sequence of measurement category names. "
                                                         "Defaults to a list of integers",
                                                     default=None)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Normalise matrix columns
        self.emission_matrix = self.emission_matrix / np.sum(self.emission_matrix, axis=0)

        if self.measurement_categories is None:
            self.measurement_categories = list(map(str, range(self.ndim_meas)))

        if len(self.measurement_categories) != self.ndim_meas:
            raise ValueError(
                f"ndim_meas of {self.ndim_meas} does not match number of measurement categories "
                f"{len(self.measurement_categories)}"
            )

    def function(self, state, **kwargs):
        r"""Applies the linear transformation:

        .. math::
            E^{ij}\alpha_{t-1}^j = P(\zeta_t^i|\phi_t^j)P(\phi_t^j)

        The resultant vector is normalised.

        Parameters
        ----------
        state: :class:`~.CategoricalState`
            The state to be measured.

        Returns
        -------
        state_vector: :class:`stonesoup.types.array.StateVector`
            of shape (:py:attr:`~ndim_meas, 1`). The resultant measurement vector.
        """

        meas_vector = self.emission_matrix @ state.state_vector
        meas_vector = meas_vector / np.sum(meas_vector)  # normalise

        return StateVector(meas_vector)

    @property
    def ndim_state(self):
        return self.emission_matrix.shape[1]

    @property
    def ndim_meas(self):
        return self.emission_matrix.shape[0]

    @property
    def mapping(self):
        """Assumes that all elements of the state space are considered."""
        return np.arange(self.ndim_state)

    def rvs(self):
        raise NotImplementedError

    def pdf(self):
        raise NotImplementedError
