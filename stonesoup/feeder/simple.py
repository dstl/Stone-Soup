import copy
import datetime
from abc import abstractmethod
from typing import Iterable, List, Tuple

from ..base import Property
from ..buffered_generator import BufferedGenerator
from ..feeder import Feeder
from .base import DetectionFeeder, GroundTruthFeeder, DetectionOutput, Feeder


class SimpleFeeder(Feeder):
    """
    A wrapper around an iterable. This is wrapper is used if you only want a feeder to be used once
    """
    reader: Iterable = Property(doc="Source of states")
    # reader is of 'iterable' type - something you can iterate over

    @BufferedGenerator.generator_method
    def data_gen(self):
        """
        Buffered generator takes this function and gives SimpleFeeder an iterable call.
        Allows user to use instantiation of class as an iterable
        ie: for item in simpleFeeder: ...
        Calling this function will return every item in reader.
        """
        for item in self.reader:
            yield item


class IterFeeder(DetectionFeeder, GroundTruthFeeder):
    """Base class for other iterating detection feeders

    """
    def __iter__(self):
        return self

    @abstractmethod
    def __next__(self) -> Tuple[datetime.datetime, set]:
        """Abstract method for going to the next item

        This makes it a requirement for any future __next__ to output something
        that is of type Tuple[datetime.datetime, set].
        """
        ...

    @BufferedGenerator.generator_method
    def data_gen(self):
        """allows object to be an iterable

        will yield the detection time and detection for each
        detection in the Feeder
        """
        feeder_iter = iter(self)
        for time, items in feeder_iter:
            yield time, items


class IterDetectionFeeder(IterFeeder):

    @abstractmethod
    def __next__(self) -> DetectionOutput:
        ...
