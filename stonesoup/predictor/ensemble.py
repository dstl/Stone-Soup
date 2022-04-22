from .base import Property
from ..models.transition import TransitionModel
from .kalman import KalmanPredictor
from ..types.array import StateVectors
from ..types.state import State
from ..types.prediction import Prediction


class EnsemblePredictor(KalmanPredictor):
    """Ensemble Kalman Filter Predictor class

    The EnKF is a hybrid of the Kalman updating scheme and the
    Monte Carlo aproach of the the particle filter.
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
        pred_ensemble = StateVectors(
            [self.transition_model.function(State(state_vector=ensemble_member),
                                            noise=True, time_interval=time_interval)
             for ensemble_member in prior.ensemble.T])

        return Prediction.from_state(prior, pred_ensemble, timestamp=timestamp,
                                     transition_model=self.transition_model)
