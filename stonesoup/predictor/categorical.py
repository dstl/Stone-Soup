# -*- coding: utf-8 -*-

from ..base import Property
from ..models.transition.categorical import MarkovianTransitionModel
from ..predictor import Predictor
from ..predictor._utils import predict_lru_cache
from ..types.prediction import Prediction


class HMMPredictor(Predictor):
    r"""Hidden Markov model predictor

    Assumes transition model is time-invariant, and therefore care should be taken when predicting
    forward to the same time."""

    transition_model: MarkovianTransitionModel = Property(
        doc="The transition model used to predict states forward in `time`."
    )

    @predict_lru_cache()
    def predict(self, prior, timestamp=None, **kwargs):
        r"""Predicts a :class:`~.CategoricalState` forward using the :attr:`transition_model`.

        Parameters
        ----------
        prior : :class:`~.CategoricalState`
            :math:`\alpha_{t-1}`
        timestamp : :class:`datetime.datetime`, optional
            :math:`t`
        **kwargs :
            These are passed to the :meth:`transition_model.function` method.

        Returns
        -------
        : :class:`~.CategoricalStatePrediction`
            The predicted state.

        Notes
        -----
        The Markovian transition model is time-invariant and the evaluated `time_interval` can be
        `None`.
        """

        predict_over_interval = self._predict_over_interval(prior, timestamp)

        prediction_vector = self.transition_model.function(prior,
                                                           time_interval=predict_over_interval,
                                                           **kwargs)

        return Prediction.from_state(prior, prediction_vector, timestamp=timestamp,
                                     transition_model=self.transition_model)

    def _predict_over_interval(self, prior, timestamp):
        """Private method to get the prediction interval (or None)

        Parameters
        ----------
        prior : :class:`~.State`
            The prior state
        timestamp : :class:`datetime.datetime`, optional
            The (current) timestamp

        Returns
        -------
        : :class:`datetime.timedelta`
            time interval to predict over
        """

        # Deal with undefined timestamps
        if timestamp is None or prior.timestamp is None:
            predict_over_interval = None
        else:
            predict_over_interval = timestamp - prior.timestamp

        return predict_over_interval
