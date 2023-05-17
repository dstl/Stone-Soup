from stonesoup.custom.types.state import TwoStateGaussianState
from stonesoup.types.prediction import Prediction


class TwoStateGaussianStatePrediction(Prediction, TwoStateGaussianState):
    """ A Gaussian state object representing the predicted distribution
    :math:`p(x_{k+T}, x_{k} | Y)` """