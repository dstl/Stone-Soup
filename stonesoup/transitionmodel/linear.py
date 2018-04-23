# -*- coding: utf-8 -*-
import numpy as np

from ..base import Property
from .base import TransitionModel


class LinearTransitionModel(TransitionModel):
    """Linear Transition Model

    Transitions state using a transition matrix, and optionally accepts a
    control matrix and vector to apply to the new state.

    .. math::

        \hat{\mathbf{x}}_{k\mid k-1} &=
            \mathbf{F}_k \hat{\mathbf{x}}_{k-1\mid k-1} +
            \mathbf{B}_k \mathbf{u}_k

    """

    transition_matrix = Property(
        np.ndarray, doc="Linear transition matrix :math:`\mathbf{F}_k`.")
    control_matrix = Property(
        np.ndarray, default=None, doc="Control matrix :math:`\mathbf{B}_k`.")

    def transition(self, state_vector, control_vector=None):
        """Transition state

        Parameters
        ----------
        state_vector : StateVector
            State vector :math:`\hat{\mathbf{x}}_{k-1|k-1}`.
        control_vector : StateVector, optional
            Control vector :math:`\mathbf{u}_k`. Default is None in which case
            no control vector is applied.

        Returns
        -------
        StateVector
            New state vector :math:`\hat{\mathbf{x}}_{k\mid k-1}`.
        """
        new_state_vector = self.transition_matrix @ state_vector
        if control_vector is not None:
            new_state_vector += self.control_matrix @ control_vector
        return new_state_vector
