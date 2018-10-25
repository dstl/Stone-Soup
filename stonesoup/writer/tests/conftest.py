# -*- coding: utf-8 -*-
import datetime

import pytest

from ...types.array import StateVector
from ...types.detection import Detection
from ...types.groundtruth import GroundTruthState, GroundTruthPath
from ...types.state import State
from ...types.track import Track


@pytest.fixture()
def detection_reader():
    class TestDetectionReader():
        def __init__(self):
            self.detections = set()

        def detections_gen(self):
            time = datetime.datetime(2018, 1, 1, 14)
            state_vector = StateVector([[0]])
            for i in range(3):
                self.detections = {
                    Detection(state_vector + i,  timestamp=time)
                    for _ in range(i)}
                yield time, self.detections
                time += datetime.timedelta(minutes=1)
    return TestDetectionReader()


@pytest.fixture()
def groundtruth_reader():
    class TestGroundTruthReader():
        def __init__(self):
            self.groundtruth_paths = set()

        def groundtruth_paths_gen(self):
            time = datetime.datetime(2018, 1, 1, 14)
            state_vector = StateVector([[0]])
            for i in range(3):
                self.groundtruth_paths = {
                    GroundTruthPath(
                        [GroundTruthState(
                            state_vector + i + 10*j, timestamp=time)
                         for j in range(i)])
                    for _ in range(i)}
                yield time, self.groundtruth_paths
                time += datetime.timedelta(minutes=1)
    return TestGroundTruthReader()


@pytest.fixture()
def tracker():
    class TestTracker():
        def __init__(self):
            self.tracks = set()

        def tracks_gen(self):
            time = datetime.datetime(2018, 1, 1, 14)
            state_vector = StateVector([[0]])
            for i in range(2):
                self.tracks = {
                    Track(
                        [State(
                            state_vector + i + 10*j, timestamp=time)
                            for j in range(i)],
                        str(k))
                    for k in range(i)}
                yield time, self.tracks
                time += datetime.timedelta(minutes=1)
    return TestTracker()
