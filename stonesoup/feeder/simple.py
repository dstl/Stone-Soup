import datetime
import warnings
from abc import abstractmethod
from collections import defaultdict, deque
from typing import List, Iterable, Collection

from .base import DetectionFeeder, GroundTruthFeeder, DetectionOutput, Feeder
from ..base import Property
from ..buffered_generator import BufferedGenerator
from ..types.detection import Detection


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
    def __next__(self) -> DetectionOutput:
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


class IterDetectionFeeder(IterFeeder):

    @abstractmethod
    def __next__(self) -> DetectionOutput:
        ...


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


class QueueDetectionFeeder(IterDetectionFeeder):
    """
    Allows user to change what comes out of the detection feeder
    while it's being iterated through
    """

    reader: List[DetectionOutput] = Property(default=None,
                                             doc="Input data to the feeder."
                                                 " Should be iterable data type.")

    def __init__(self, *args, **kwargs):
        """
        If no reader is given, reader will be made an empty list.
        Otherwise, reader will be made to a list.
        """
        super().__init__(*args, **kwargs)
        if self.reader is None:
            self.reader = []
        else:
            self.reader = list(self.reader)

    def __next__(self) -> DetectionOutput:
        """
        If reader still exists, return the first element and remove that
        element from the list. Otherwise, stop iterating.
        """
        if len(self.reader) > 0:
            return self.reader.pop(0)
        else:
            raise StopIteration

    def append_detection(self, detection: Detection):
        """
        Add a detection to the reader.
        """
        self.reader.append((detection.timestamp, {detection}))


class UnorderedDetectionsToDetectionFeeder(IterDetectionFeeder):
    """
    Takes out-of-order detections as input, groups together detections from the same time step,
    and sorts them into time order. Enables detections to be iterated over.
    """
    reader: Iterable[Detection] = Property(doc="Source of detections")
    times: Collection[datetime.datetime] = Property(
        default=None,
        doc="If none, find earliest time of detections")

    def __iter__(self):
        """
        Creates a dictionary in which the keys are the union of detection timesteps in 'reader'
        and timesteps in 'times', and the values are sets of detections associated with those
        times. The times are then sorted and made into an iterable, which will be used by __next__.
        """
        self.detection_dict = defaultdict(set)

        if self.times is not None:
            for time in self.times:
                self.detection_dict[time] = set()

        for det in self.reader:
            self.detection_dict[det.timestamp].add(det)
        self.times = sorted(self.detection_dict.keys())
        self.times_iter = iter(self.times)  # creates iterable version of times.
        return self

    def __next__(self) -> DetectionOutput:
        """
        Iterate once and output next timestamp and set of detections
        associated with that timestamp.
        """
        time = next(self.times_iter)
        return time, self.detection_dict[time]


class OrderedDetectionsToDetectionFeeder(DetectionFeeder):
    """
    Converts collection of ordered detections to an iterable that
    yields a timestamp and a set of detections associated with that timestamp.

    Required because by default, the reader will only return one detection
    and one timestamp, rather than the set of detections associated
    with one timestamp."
    """
    reader: Iterable[Detection] = Property(doc="Source of detections")

    @BufferedGenerator.generator_method
    def data_gen(self) -> DetectionOutput:
        """
        Implementation of class description.
        """
        time = datetime.datetime.min
        detection_buffer = set()

        for idx, detection in enumerate(self.reader):
            if idx == 0:  # first detection
                time = detection.timestamp
                detection_buffer = {detection}

            elif detection.timestamp == time:  # if current det is at same time as previous det
                detection_buffer.add(detection)  # add detection to current set

            elif detection.timestamp > time:  # if current detection happens after previous one
                yield time, detection_buffer  # output these

                time = detection.timestamp  # update time and detection buffer to new timestep
                detection_buffer = {detection}
            else:
                # Detections are unordered
                warnings.warn("Detections are unordered. Detections have been thrown away."
                              "Please use UnorderedDetectionsToDetectionFeeder.")

        # No more new detections: yield remaining buffer
        yield time, detection_buffer  # Needed otherwise will not output final timestep
