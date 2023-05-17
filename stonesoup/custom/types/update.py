from stonesoup.custom.types.state import TwoStateGaussianState
from stonesoup.types.update import Update


class TwoStateGaussianStateUpdate(Update, TwoStateGaussianState):
    """ A Gaussian state object representing the predicted distribution
    :math:`p(x_{k+T}, x_{k} | Y)` """