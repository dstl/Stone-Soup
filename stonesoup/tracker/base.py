import datetime
from abc import abstractmethod, ABC
from typing import Tuple, Set

from ..base import Base, Property
from ..reader import DetectionReader
from ..types.detection import Detection
from ..types.track import Track


class Tracker(Base):
    """Tracker base class"""

    @property
    @abstractmethod
    def tracks(self) -> Set[Track]:
        raise NotImplementedError

    def __iter__(self):
        return self

    @abstractmethod
    def __next__(self) -> Tuple[datetime.datetime, Set[Track]]:
        """
        Returns
        -------
        : :class:`datetime.datetime`
            Datetime of current time step
        : set of :class:`~.Track`
            Tracks existing in the time step
        """
        raise NotImplementedError


class TrackerWithDetector(Tracker, ABC):
    detector: DetectionReader = Property(doc="Detector used to generate detection objects.")

    def __iter__(self):
        self.detector_iter = iter(self.detector)
        return super().__iter__()

    def __next__(self) -> Tuple[datetime.datetime, Set[Track]]:
        time, detections = next(self.detector_iter)
        return self.update_tracker(time, detections)

    def update_tracker(self, time: datetime.datetime, detections: Set[Detection]) \
            -> Tuple[datetime.datetime, Set[Track]]:
        """
        This function updates the tracker with detections and outputs the time and tracks
        """
        raise NotImplementedError
