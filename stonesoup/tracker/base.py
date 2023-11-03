import datetime
from abc import abstractmethod
from typing import Tuple, Set, Iterator, Iterable

from ..base import Base
from ..types.detection import Detection
from ..types.track import Track


class Tracker(Base):
    """Tracker base class"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.detector_iter = None

    @property
    def detector(self) -> Iterable[Tuple[datetime.datetime, Set[Detection]]]:
        """ Input into the tracker"""
        raise TypeError("Detector is not present in this tracker")

    def __iter__(self) -> Iterator[Tuple[datetime.datetime, Set[Track]]]:
        if self.detector is None:
            raise ValueError("Detector has not been set")

        if self.detector_iter is None:
            self.detector_iter = iter(self.detector)

        return self

    def __next__(self) -> Tuple[datetime.datetime, Set[Track]]:
        time, detections = next(self.detector_iter)
        return self.update_tracker(time, detections)

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

    @property
    @abstractmethod
    def tracks(self) -> Set[Track]:
        raise NotImplementedError
