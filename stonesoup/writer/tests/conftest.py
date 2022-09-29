import datetime

import pytest

from ...buffered_generator import BufferedGenerator
from ...reader import DetectionReader, GroundTruthReader
from ...tracker import Tracker
from ...types.array import StateVector
from ...types.detection import Detection
from ...types.groundtruth import GroundTruthState, GroundTruthPath
from ...types.state import State
from ...types.track import Track


@pytest.fixture()
def detection_reader():
    class TestDetectionReader(DetectionReader):
        @BufferedGenerator.generator_method
        def detections_gen(self):
            time = datetime.datetime(2018, 1, 1, 14)
            state_vector = StateVector([[0]])
            for i in range(3):
                detections = {
                    Detection(state_vector + i,  timestamp=time)
                    for _ in range(i)}
                yield time, detections
                time += datetime.timedelta(minutes=1)
    return TestDetectionReader()


@pytest.fixture()
def groundtruth_reader():
    class TestGroundTruthReader(GroundTruthReader):
        @BufferedGenerator.generator_method
        def groundtruth_paths_gen(self):
            time = datetime.datetime(2018, 1, 1, 14)
            state_vector = StateVector([[0]])
            for i in range(2):
                groundtruth_paths = {
                    GroundTruthPath(
                        [GroundTruthState(
                            state_vector + i + 10*j, timestamp=time)
                         for j in range(i)],
                        str(k))
                    for k in range(i)}
                yield time, groundtruth_paths
                time += datetime.timedelta(minutes=1)
    return TestGroundTruthReader()


@pytest.fixture()
def tracker():
    class TestTracker(Tracker):
        @property
        def tracks(self):
            return self._tracks

        def __iter__(self):
            self.iter = iter(range(2))
            self.time = datetime.datetime(2018, 1, 1, 13, 59)
            return super().__iter__()

        def __next__(self):
            i = next(self.iter)
            state_vector = StateVector([[0]])
            self.time += datetime.timedelta(minutes=1)
            self._tracks = {
                Track(
                    [State(
                        state_vector + i + 10*j, timestamp=self.time)
                        for j in range(i)],
                    str(k))
                for k in range(i)}
            return self.time, self.tracks
    return TestTracker()
