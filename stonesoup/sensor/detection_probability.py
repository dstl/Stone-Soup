from abc import abstractmethod
import numpy as np

from stonesoup.base import Base, Property
from stonesoup.types.array import StateVector


class DetectionProbability(Base):
    """Base class to define sensor detection probability"""
    @abstractmethod
    def __call__(self, state_vector: StateVector) -> float:
        """
        Method to determine the probability of detection based on the location of the target in
        bearing, range space. Note for a 2D sensor the state vector will be [bearing, range] and
        for a 3D sensor the state vector will be [elevation, bearing, range]

        Parameters
        ----------
        state_vector : StateVector
            location of the target in bearing range space

        Returns
        -------
        float
            The resulting probability of detection
        """


class ConstantDetectionProbability(DetectionProbability):
    """:class:`.DetectionProbability` class to implement a constant probability of detection"""
    p_d: float = Property(default=1., doc="The probaiblity of detection")

    def __call__(self, *args, **kwargs) -> float:
        return self.p_d


class ExponentialDecayDetectionProbability(DetectionProbability):
    """:class:`.DetectionProbability` class to implement a probability of detection of decays as
    the range to the target increases"""
    decay_rate: float = Property(default=np.inf, doc="The characteristic range of the "
                                                     "exponential decay")

    def __call__(self, state_vector, *args, **kwargs) -> float:
        range = state_vector[-1, :].astype(float)
        return np.exp(-range / self.decay_rate)


class SigmoidDetectionProbability(DetectionProbability):
    """:class:`.DetectionProbability` class to implement a probability of detection which observes
    a sigmoid function"""
    decay_rate: float = Property(doc="The rate at which the Sigmoid function decays")
    midpoint: float = Property(doc="The range at which the probability of detection is 50%")

    def __call__(self, state_vector, *args, **kwargs):
        range = state_vector[-1, :].astype(float)
        return 1 - 1/(1 + np.exp(-(range-self.midpoint)/self.decay_rate))
