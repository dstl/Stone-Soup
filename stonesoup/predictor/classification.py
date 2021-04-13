# -*- coding: utf-8 -*-

from ..base import Property
from ..models.transition import TransitionModel
from ..predictor._utils import predict_lru_cache
from ..predictor.kalman import KalmanPredictor
from ..types.prediction import Prediction


class ClassificationPredictor(KalmanPredictor):
    r"""A simplification of the Kalman filter prediction process. Removing calculation of
    resultant covariances, under the assumption that the state space is comprised of a finite set
    of discrete classifications that a state object can take, whereby a state vector would
    represent a multinomial distribution over this finite set.
    Here,

    .. math::

        \mathbf{x}_{t + \Delta t} = (I + \boldsymbol{\omega})\mathbf{Fx}_t

    Notes
    -----
    It is assumed that a transition model similar to that of
    :class:`~.BasicTimeInvariantClassificationTransitionModel` is used.
    """

    transition_model: TransitionModel = Property(doc="The transition model to be used")

    def _transition_function(self, prior, **kwargs):
        return self.transition_model.function(prior, **kwargs)

    @predict_lru_cache()
    def predict(self, prior, timestamp=None, **kwargs):
        r"""The predict function

        Parameters
        ----------
        prior : :class:`~.State`
            :math:`\mathbf{x}_{t}` representing a state
        timestamp : :class:`datetime.datetime`, optional
            :math:`t + \Delta t`, where :math:`\Delta t > 0`.
        **kwargs :
            These are passed to the :meth:`transition_model.function` method.

        Returns
        -------
        : :class:`~.StatePrediction`
            :math:`\mathbf{x}_{t + \Delta t|t}`, the predicted state.
        """
        predict_over_interval = self._predict_over_interval(prior, timestamp)

        x_pred = self._transition_function(prior, time_interval=predict_over_interval, **kwargs)

        return Prediction.from_state(prior, x_pred, timestamp=timestamp)
