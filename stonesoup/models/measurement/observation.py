# -*- coding: utf-8 -*-
from typing import Union

import numpy as np
import scipy

from stonesoup.measures import ObservationAccuracy
from .base import MeasurementModel
from ...base import Property
from ...models.base import ReversibleModel
from ...types.array import Matrix, StateVector, StateVectors


class BasicTimeInvariantObservationModel(MeasurementModel, ReversibleModel):
    r"""This class models a simple observation of a hidden class that is time invariant.
    It is assumed that the entire state vector in the state space is observed, and that this
    vector defines a categorical distribution on the hidden class of the target being observed.
    I.e. Each component of a state space state vector represents the probability that a target is
    a particular class.
    Output measurements are state vectors in the measurement space, defining categorical
    distributions over measurement classes.

    Notes:
        All properties of the model can be defined by its :attr:`emission_matrix`.
        As all components of the state space are considered, no :attr:`mapping` is necessary.
    """
    emission_matrix: Matrix = Property(
        doc=r"Matrix defining emissions from measurement classes. In essence, it defines the "
            r"probability an observed target is a particular hidden class :math:`\phi_{i}`, given "
            r"it has been observed to be measured class :math:`z_{j}`. "
            r":math:`E_{ij} = P(\phi_{i} | z_{j})`. This can be defined without normalisation, "
            r"whereby on instantiation, the matrix will have its columns normalised to sum to 1.")
    reverse_emission: Matrix = Property(
        default=None,
        doc=r"Matrix utilised in generating observations. Defines the probability a target of "
            r"hidden class :math:`\phi_{j}` will be observed as measurement class :math:`z_{i}`. "
            r":math:`K_{ij} = P(z_{i} | \phi_{j})`. If undefined, will default to the "
            r"transpose of the emission matrix with columns normalised to sum to 1.")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Normalise emission rows
        self.emission_matrix = \
            self.emission_matrix / self.emission_matrix.sum(axis=0)[np.newaxis, :]

        # Default reverse emission is normalised emission transpose
        if self.reverse_emission is None:
            self.reverse_emission = \
                self.emission_matrix.T / self.emission_matrix.T.sum(axis=0)[np.newaxis, :]

    @property
    def mapping(self):
        return np.arange(self.ndim_state)

    @property
    def ndim_meas(self):
        """ndim_meas getter method

        Returns
        -------
        :class:`int`
            The number of observation dimensions
        """

        return np.shape(self.reverse_emission)[0]

    @property
    def ndim_state(self):
        """Number of state dimensions"""
        return np.shape(self.emission_matrix)[0]

    def function(self, state, noise: bool = False, **kwargs):
        """Observer function :math:`HX_{t}`

        Parameters
        ----------
        state: :class:`~.State`
            An input (hidden class) state
        noise: bool
            If 'True', the resultant multinomial distribution is sampled from, and a definitive
            measurement class is returned. This can be interpreted as there being no ambiguity as
            to what the returned class is. To represent this, a 1 is placed at the index of the
            observed measurement class in the measurement vector, and 0's elsewhere.
            If 'False', the resultant multinomial is returned.

        Returns
        -------
        :class:`numpy.ndarray` of shape (:py:attr:`~ndim_meas`, 1)
            The observer function evaluated and resultant categorical distribution sampled from to
            represent a determinate measurement class.
        """

        y = self.reverse_emission @ state.state_vector

        y = y / np.sum(y)

        if noise:
            y = self._sample(y.flatten())

        return StateVector(y)

    def inverse_function(self, detection, **kwargs) -> StateVector:
        return self.emission_matrix @ detection.state_vector

    def jacobian(self, state, **kwargs):
        raise NotImplementedError("Jacobian for observation measurement model is not defined.")

    def _sample(self, row):
        rv = scipy.stats.multinomial(n=1, p=row)
        return rv.rvs(size=1, random_state=None)

    def pdf(self, state1, state2, **kwargs):
        measure = ObservationAccuracy()
        return measure(state1, state2)

    def rvs(self, num_samples=1, **kwargs) -> Union[StateVector, StateVectors]:
        raise NotImplementedError("Noise generation for observation-based measurements is not "
                                  "implemented")
