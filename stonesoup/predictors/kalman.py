from ..predictor.kalman import KalmanPredictor

from .base import Predictors


class KalmanPredictors(Predictors):
    predictor_class = KalmanPredictor
