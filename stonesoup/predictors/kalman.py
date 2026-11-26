from ..predictor.kalman import KalmanPredictor

from .base import Predictors


class KalmanPredictors(Predictors):
    """Predictor container for Kalman transition models.

    This class specialises :class:`~.Predictors` by configuring
    :class:`~.KalmanPredictor` as the predictor class to be instantiated for
    each provided transition model.
    """
    predictor_class = KalmanPredictor
