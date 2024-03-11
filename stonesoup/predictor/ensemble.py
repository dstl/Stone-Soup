from .base import Property
from ..models.transition import TransitionModel
from .kalman import KalmanPredictor
from ..types.prediction import Prediction


class EnsemblePredictor(KalmanPredictor):
    r"""Ensemble Kalman Filter Predictor class

    The EnKF predicts the state by treating each column of the ensemble matrix
    as a state vector. The state is propagated through time by applying the
    transition function to each member (vector) of the ensemble.

    .. math::

        \hat{X}_k = [f(x_1), f(x_2), ..., f(x_M)]

    """
    transition_model: TransitionModel = Property(doc="The transition model to be used.")

    def predict(self, prior, timestamp=None, **kwargs):
        """Ensemble Kalman Filter prediction step

        Parameters
        ----------
        prior : :class:`~.EnsembleState`
            A prior state object
        control_input : :class:`~.State`, optional
            The control input. It will only have an effect if
            :attr:`control_model` is not `None` (the default is `None`)
        timestamp: :class:`datetime.datetime`, optional
            A timestamp signifying when the prediction is performed
            (the default is `None`)

        Returns
        -------
        : :class:`~.EnsembleStatePrediction`
            The predicted state
        """

        # Compute time_interval
        time_interval = self._predict_over_interval(prior, timestamp)
        # This block of code propagates each column through the transition model.
        pred_ensemble = self.transition_model.function(
            prior, noise=True, time_interval=time_interval)

        return Prediction.from_state(prior, pred_ensemble, timestamp=timestamp,
                                     transition_model=self.transition_model, prior=prior)
