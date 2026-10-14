from ...predictor.kalman import KalmanPredictor
from ...models.transition.linear import ConstantVelocity

from ..base import Predictors
from ..kalman import KalmanPredictors


def test_kalman_predictors():
    predictor_class = KalmanPredictor
    cv = ConstantVelocity(1)
    transitions = [cv, cv]
    predictor_list = [predictor_class(model) for model in transitions]
    F3KP = KalmanPredictors(transition_models=transitions)
    F3P = Predictors(predictor_class=predictor_class, transition_models=transitions)
    for x, y, z in zip(F3KP.predictors, F3P.predictors, predictor_list):
        assert x.transition_model == cv
        assert y.transition_model == cv
        assert z.transition_model == cv
