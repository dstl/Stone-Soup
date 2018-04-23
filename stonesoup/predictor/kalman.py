# -*- coding: utf-8 -*-
import numpy as np

from ..base import Property
from .base import Predictor
from ..transitionmodel import LinearTransitionModel
from ..types import CovarianceMatrix, GaussianState


class KalmanPredictor(Predictor):
    """Kalman Predictor

    Kalman predictor utilising a provided transition model to estimate the state
    vector :math:`\hat{\mathbf{x}}_{k\mid k-1}` and an estimated covariance
    :math:`\mathbf{P}_{k\mid k-1}`

    .. math::

        \mathbf{P}_{k\mid k-1}=
        \mathbf{F}_{k} \mathbf{P}_{k-1\mid k-1} \mathbf{F _{k}^{\mathrm{T}}+
        \mathbf{Q}_{k}}

    """

    transition_model = Property(
        LinearTransitionModel,
        doc="Linear transition model for predicting the state estimate")
    process_noise_covar = Property(
        CovarianceMatrix, doc="Process noise :math:`\mathbf{Q}_k`")

    def predict(self, state):
        """Predict state

        Parameters
        ----------
        state : GaussianState
            State which to predict from. :math:`\hat{\mathbf{x}}_{k-1|k-1}` and
            :math:`\mathbf{P}_{k-1\mid k-1}`

        Returns
        -------
        GaussianState
            Predicted state :math:`\hat{\mathbf{x}}_{k-1\mid k}` and
            :math:`\mathbf{P}_{k-1\mid k}`
        """
        trans_matrix = self.transition_model.transition_matrix
        covar = getattr(state, 'covar', np.zeros((state.ndim, state.ndim)))
        return GaussianState(
            self.transition_model.transition(state.state_vector),
            trans_matrix @ covar @ trans_matrix.T + self.process_noise_covar
        )
