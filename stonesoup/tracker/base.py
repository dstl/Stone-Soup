import datetime
from abc import abstractmethod
from typing import Tuple, Set, Iterator

from ..base import Base, Property
from ..reader import DetectionReader
from ..types.detection import Detection
from ..types.track import Track


class Tracker(Base):
    """Tracker base class"""

    detector: DetectionReader = Property(default=None,
                                         doc="Detector used to generate detection objects.")

    @property
    @abstractmethod
    def tracks(self) -> Set[Track]:
        raise NotImplementedError

    def __iter__(self) -> Iterator[Tuple[datetime.datetime, Set[Track]]]:
        if self.detector is None:
            raise ValueError("Detector has not been set")

        for time, detections in self.detector:
            yield self.update_tracker(time, detections)

    @abstractmethod
    def update_tracker(self, time: datetime.datetime, detections: Set[Detection]) \
            -> Tuple[datetime.datetime, Set[Track]]:
        """
        This function updates the tracker with detections and outputs the time and tracks

        Returns
        -------
        : :class:`datetime.datetime`
            Datetime of current time step
        : set of :class:`~.Track`
            Tracks existing in the time step
        """
        raise NotImplementedError
