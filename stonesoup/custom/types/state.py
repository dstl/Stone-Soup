import datetime

from stonesoup.base import Property
from stonesoup.types.numeric import Probability
from stonesoup.types.state import GaussianState


class TwoStateGaussianState(GaussianState):
    """ A Gaussian state object representing the distribution :math:`p(x_{k+T}, x_{k} | Y)` """
    start_time: datetime.datetime = Property(doc='Timestamp at t_k')
    end_time: datetime.datetime = Property(doc='Timestamp at t_{k+T}')
    weight: Probability = Property(default=0, doc="Weight of the Gaussian State.")
    tag: str = Property(default=None, doc="Unique tag of the Gaussian State.")
    # scan_id: int = Property(doc='The scan id')

    @property
    def timestamp(self):
        return self.end_time