"""Base classes for Stone Soup feeder"""
import datetime
from abc import abstractmethod
from typing import Iterator, Tuple, Set, Sequence

from ..base import Base
from ..base import Property
from ..buffered_generator import BufferedGenerator
from ..reader import Reader, DetectionReader, GroundTruthReader
from ..types.detection import Detection
from ..types.track import Track

DetectionOutput = Tuple[datetime.datetime, Set[Detection]]
TrackOutput = Tuple[datetime.datetime, Set[Track]]


class Feeder(Reader):
    """Feeder base class

    Feeder consumes and outputs :class:`.State` data and can be used to
    modify the sequence, duplicate or drop data.
    """

    reader: Reader = Property(doc="Source of detections")

    @abstractmethod
    @BufferedGenerator.generator_method
    def data_gen(self):
        raise NotImplementedError


class DetectionFeeder(Feeder, DetectionReader):
    """Detection feeder base class

    Feeder consumes and outputs :class:`.Detection` data and can be used to
    modify the sequence, duplicate or drop data.
    """

    @BufferedGenerator.generator_method
    def detections_gen(self):
        yield from self.data_gen()


class GroundTruthFeeder(Feeder, GroundTruthReader):
    """Ground truth feeder base class

    Feeder consumes and outputs :class:`.GroundTruthPath` data and can be used to
    modify the sequence, duplicate or drop data.
    """

    @BufferedGenerator.generator_method
    def groundtruth_paths_gen(self):
        yield from self.data_gen()


class TrackFeeder(Base):
    """
    Something that produces (time, Set[Track])
    """

    @abstractmethod
    def __iter__(self) -> Iterator[TrackOutput]:
        raise NotImplementedError


class MultipleTrackFeeder(Feeder):
    """Multiple track feeder base class

    Feeds tracks from multiple sources
    It is important that the order of the Tracks is the same as the order of the readers
    """

    reader: Reader = Property(default=None, doc="Overrides parent `Reader` class, is not needed")
    readers: Sequence[TrackFeeder] = Property(doc="Source of tracks")

    @abstractmethod
    def __iter__(self) -> Iterator[Tuple[datetime.datetime, Sequence[Set[Track]]]]:
        raise NotImplementedError

    def data_gen(self):
        raise NameError
