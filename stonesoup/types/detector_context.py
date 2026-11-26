from abc import ABC, abstractmethod
from collections.abc import Callable

from .detection import Detection
from .hypothesis import SingleHypothesis
from .numeric import Probability


class DetectorContext(ABC):
    """Detection probability and clutter intensity for a detector step."""

    @abstractmethod
    def prob_detection(self, hypothesis: SingleHypothesis) -> Probability:
        """Return the detection probability for a hypothesis."""
        raise NotImplementedError

    @abstractmethod
    def clutter_spatial_density(self, detection: Detection) -> float:
        """Return the clutter spatial density for a detection."""
        raise NotImplementedError


class SimpleDetectorContext(DetectorContext):
    """Detector context backed by scalar values or callables."""

    def __init__(
            self,
            prob_detection: Probability | float | Callable[[SingleHypothesis], Probability]
            = Probability(1),
            clutter_spatial_density: float | Callable[[Detection], float] = 1e-26):
        self._prob_detection = prob_detection
        self._clutter_spatial_density = clutter_spatial_density

    def prob_detection(self, hypothesis):
        if callable(self._prob_detection):
            return self._prob_detection(hypothesis)
        return Probability(self._prob_detection)

    def clutter_spatial_density(self, detection):
        if callable(self._clutter_spatial_density):
            return self._clutter_spatial_density(detection)
        return self._clutter_spatial_density
