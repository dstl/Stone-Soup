import copy
from datetime import datetime

from stonesoup.base import Property
from ..buffered_generator import BufferedGenerator
from .base import DetectionReader
from stonesoup.types.detection import Detection


class CustomDetectionFeeder(DetectionReader):
    """A custom runtime dynamic detection feeder


    This detection feeder allows the detections to be dynamically altered during runtime.
    At each iteration the detection feeder will remove a detection from the front of the list 'available_detections' and
    package in a format to match other detection feeders: a tuple containing a timestamp and set
    (containing detections). The timestamp will be drawn from the timestamp in the detection. If the detection
    statevector is None, then there is an empty set with the timestamp from the detection

    """
    available_detections = Property(list, default=None, doc='These detections are returned one at time when the class '
                                                            'is iterated over.')

    def __iter__(self):
        return self

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.available_detections = []

    def __next__(self):
        if len(self.available_detections) > 0:
            a_detection = self.available_detections.pop(0)
            if a_detection is not None:
                if isinstance(a_detection, Detection):
                    output = (a_detection.timestamp, {a_detection})
                    if a_detection.state_vector is None:
                        output = (a_detection.timestamp, set())
                    return output
                elif isinstance(a_detection, tuple) and isinstance(a_detection[0], datetime) and \
                        isinstance(a_detection[1], set) and \
                        all(isinstance(list_item, Detection) for list_item in a_detection[1]):
                    return a_detection
                else:
                    raise SyntaxError
        else:
            raise StopIteration

    @BufferedGenerator.generator_method
    def detections_gen(self):
        detection_iter = iter(self)
        for time, detections in detection_iter:
            yield (time, detections)