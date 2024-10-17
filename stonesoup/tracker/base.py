import datetime
from abc import abstractmethod
from collections.abc import Iterator

from ..base import Base
from ..types.detection import Detection
from ..types.track import Track


class Tracker(Base):
    """Tracker base class"""

    @property
    @abstractmethod
    def tracks(self) -> set[Track]:
        raise NotImplementedError

    def __iter__(self) -> Iterator[tuple[datetime.datetime, set[Track]]]:
        return self

    @abstractmethod
    def __next__(self) -> tuple[datetime.datetime, set[Track]]:
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

    def __iter__(self) -> Iterator[tuple[datetime.datetime, set[Track]]]:
        if self.detector is None:
            raise AttributeError("Detector has not been set. A detector attribute is required to "
                                 "iterate over a tracker.")
        if self.detector_iter is None:
            self.detector_iter = iter(self.detector)

        return super().__iter__()


class _TrackerMixInNext(_TrackerMixInBase):
    """ The tracking logic is contained within the __next__ method."""

    @abstractmethod
    def __next__(self) -> tuple[datetime.datetime, set[Track]]:
        """Pull detections from the detector (`detector_iter`). Act on them to create tracks."""

    def update_tracker(self, time: datetime.datetime, detections: set[Detection]) \
            -> tuple[datetime.datetime, set[Track]]:

        placeholder_detector_iter = self.detector_iter
        self.detector_iter = iter([(time, detections)])
        tracker_output = next(self)
        self.detector_iter = placeholder_detector_iter
        return tracker_output


class _TrackerMixInUpdate(_TrackerMixInBase):
    """ The tracking logic is contained within the update_tracker function."""

    def __next__(self) -> tuple[datetime.datetime, set[Track]]:
        time, detections = next(self.detector_iter)
        return self.update_tracker(time, detections)

    @abstractmethod
    def update_tracker(self, time: datetime.datetime, detections: set[Detection]) \
            -> tuple[datetime.datetime, set[Track]]:
        """Use `time` and `detections` to create tracks."""
