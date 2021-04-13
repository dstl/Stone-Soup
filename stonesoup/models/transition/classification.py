# -*- coding: utf-8 -*-
from typing import Union

import numpy as np

from ...measures import ObservationAccuracy
from ...base import Property
from ...models.transition import TransitionModel
from ...types.array import Matrix, StateVector, StateVectors


class BasicTimeInvariantClassificationTransitionModel(TransitionModel):
    r"""Time invariant model of a classification transition

    The assumption is that an object can be classified as finitely many discrete classes
    :math:`\{\phi_k|k\in\Z_{\ge0}\}`, with a state space defined by state vectors representing
    multinomial distributions over these classes :math:`x_{t_i} = P(\phi_i, t)`,
    with constant probability :math:`P(\phi_i, t + \Delta t | \phi_j, t)` of transitioning from
    any class :math:`\phi_j` to class :math:`\phi_i` in any given time-step :math:`\Delta t > 0`.
    This is modelled by the stochastic matrix :attr:`transition_matrix`.
    """
    transition_matrix: Matrix = Property(
        doc=r"Matrix :math:`F_{ij} = P(\phi_i, t + \Delta t | \phi_j, t)` determining the "
            r"probability that an object is class :math:`\phi^i` at time :math:`t + \Delta t` "
            r"given that it was class :math:`\phi^{j}` at time "
            r":math:`t \hspace \forall \Delta t > 0`.")
    transition_noise: Matrix = Property(
        default=None,
        doc=r"Matrix :math:`\omega_{ij}` defining additive noise to class transition. "
            r"Noise added is given by :math:`noise_{i} = \omega_{ij}F_{jk}x_{k}`")

    @property
    def ndim_state(self):
        """ndim_state getter method

        Returns
        -------
        : :class:`int`
            The number of model state dimensions.
        """

        return self.transition_matrix.shape[0]

    def function(self, state, noise: bool = False, **kwargs) -> StateVector:
        r"""Applies transformation

        .. math::
            \mathbf{x}_{t + \Delta t} &= f(\mathbf{x}_{t + \Delta t}, t)\\
                                      &= f(\mathbf{x}_{t + \Delta t})\\
                                      &= \mathbf{Fx}_t + \boldsymbol{\Omega}\\
                                      &= \mathbf{Fx}_t + \boldsymbol{\omega}\mathbf{Fx}_t\\
                                      &= (I + \boldsymbol{\omega})\mathbf{Fx}_t

        .. math::
            (\mathbf{x}_{t + \Delta t})_i =
            (\delta_{ij} + \omega_{ij})P(\phi_j, t + \Delta t | \phi_k, t)P(\phi_k, t)

        (note that this is then normalised).

        Parameters
        ----------
        state : :class:`stonesoup.state.State`
            The state to be transitioned according to the models in :py:attr:`~model_list`.

        Returns
        -------
        state_vector: :class:`stonesoup.types.array.StateVector`
            of shape (:py:attr:`~ndim_state, 1`). The resultant state vector of the transition,
            representing a multinomial distribution across the space of possible classes the
            transitioned object can take.
        """
        x = self.transition_matrix @ state.state_vector

        if noise:
            if self.transition_noise is None:
                raise AttributeError("Require a defined transition noise matrix to generate noise")

            row = self.transition_noise @ x

            x = x + StateVector(row)

        x = x / np.sum(x)  # normalise

        return x

    def pdf(self, state1, state2, **kwargs):
        measure = ObservationAccuracy()
        return measure(state1, state2)

    def rvs(self, num_samples=1, **kwargs) -> Union[StateVector, StateVectors]:
        raise NotImplementedError("Noise generation for classification-based state transitions is "
                                  "not implemented")
