import datetime
from abc import abstractmethod
from collections import deque
from typing import Tuple

from .base import DetectionFeeder, GroundTruthFeeder
from ..base import Property
from ..buffered_generator import BufferedGenerator


class IterFeeder(DetectionFeeder, GroundTruthFeeder):
    """Base class for other iterating detection feeders

    """
    def __iter__(self):
        return self

    @abstractmethod
    def __next__(self) -> Tuple[datetime.datetime, set]:
        """Abstract method for going to the next item

        This makes it a requirement for any future __next__
        to output something that is of type DetectionOutput.

        """
        ...

    @BufferedGenerator.generator_method
    def data_gen(self):
        """allows object to be an iterable

        will yield the detection time and detection for each
        detection in the Feeder
        """
        detection_iter = iter(self)
        for time, detections in detection_iter:
            yield time, detections


class QueueFeeder(IterFeeder):
    reader: deque = Property(default=None)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.reader is None:
            self.reader = deque()
        elif isinstance(self.reader, deque):
            pass
        else:
            self.reader = deque(self.reader)

    def __next__(self):
        if len(self.reader) > 0:
            return self.reader.popleft()
        else:
            raise StopIteration

    def append(self, value):
        self.reader.append(value)
