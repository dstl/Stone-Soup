import datetime
from abc import abstractmethod
from typing import Iterator, Set, Tuple

from ..base import Base
from ..types.detection import Detection
from ..types.track import Track


class Tracker(Base):
    """Tracker base class"""

    @property
    @abstractmethod
    def tracks(self) -> Set[Track]:
        raise NotImplementedError

    def __iter__(self) -> Iterator[Tuple[datetime.datetime, Set[Track]]]:
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


class _TrackerMixInBase(Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.detector_iter = None

    def __iter__(self) -> Iterator[Tuple[datetime.datetime, Set[Track]]]:
        if self.detector is None:
            raise AttributeError("Detector has not been set. A detector attribute is required to "
                                 "iterate over a tracker.")
        if self.detector_iter is None:
            self.detector_iter = iter(self.detector)

        return super().__iter__()


class _TrackerMixInNext(_TrackerMixInBase):
    """ The tracking logic is contained within the __next__ method."""

    @abstractmethod
    def __next__(self) -> Tuple[datetime.datetime, Set[Track]]:
        ...

    def update_tracker(self, time: datetime.datetime, detections: Set[Detection]) \
            -> Tuple[datetime.datetime, Set[Track]]:

        placeholder_detector_iter = self.detector_iter
        self.detector_iter = iter([(time, detections)])
        tracker_output = next(self)
        self.detector_iter = placeholder_detector_iter
        return tracker_output


class _TrackerMixInUpdate(_TrackerMixInBase):
    """ The tracking logic is contained within the update_tracker function."""

    def __next__(self) -> Tuple[datetime.datetime, Set[Track]]:
        time, detections = next(self.detector_iter)
        return self.update_tracker(time, detections)

    @abstractmethod
    def update_tracker(self, time: datetime.datetime, detections: Set[Detection]) \
            -> Tuple[datetime.datetime, Set[Track]]:
        ...
