import collections
from abc import abstractmethod

from .base import Initiator
from ..base import Property
from ..models.base import ReversibleModel
from ..types.detection import Detection


class StatesLengthLimiter(Initiator):
    """Wrapper that defines the length of track history stored in memory

    By default Stone Soup stores the track history for all tracks in memory.  If
    running Stone Soup on very large data your application may run out of memory and
    the process terminated - often by the operating system.

    This wrapper converts the states space list to a collections.deque data type
    with the maximum length specified.

    .. code-block:: python

        from stonesoup.initiator.wrapper import StatesLengthLimiter

        initiator = StatesLengthLimiter(<initiator model>, max_length)

    """
    initiator: Initiator = Property(doc="Stone Soup Initiator")
    max_length: int = Property(doc="Length of track history to be stored in memory")

    def initiate(self, *args, **kwargs):
        tracks = self.initiator.initiate(*args, **kwargs)
        for track in tracks:
            track.states = collections.deque(track.states, self.max_length)
            track.metadatas = collections.deque(track.metadatas, self.max_length)
        return tracks


class InitiatorWithFilteredDetections(Initiator):

    """
    This wrapper lets you filter the detections going into an initiator
    """

    initiator: Initiator = Property(doc="Stone Soup Initiator")

    def initiate(self, detections, *args, **kwargs):
        filtered_detections = {detection for detection in detections
                               if self.filter_detection(detection)}
        return self.initiator.initiate(filtered_detections, *args, **kwargs)

    @abstractmethod
    def filter_detection(self, detection: Detection) -> bool:
        """This function determines if a detection is passed to an initiator"""
        raise NotImplementedError


class InitiatorWithOnlyReversibleDetections(InitiatorWithFilteredDetections):
    def filter_detection(self, detection: Detection) -> bool:
        return isinstance(detection.measurement_model, ReversibleModel)
