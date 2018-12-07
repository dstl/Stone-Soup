# -*- coding: utf-8 -*-
import datetime

import numpy as np
import pytest

from ...types.array import StateVector
from ...types.detection import Detection
from ...types.hypothesis import SingleMeasurementDistanceHypothesis
from ...types.prediction import StatePrediction, StateMeasurementPrediction
from ...types.state import State
from ...types.track import Track


@pytest.fixture()
def initiator():
    class TestInitiator:
        def initiate(self, unassociated_detections):
            return {Track([State(detection.state_vector)])
                    for detection in unassociated_detections}
    return TestInitiator()


@pytest.fixture()
def deleter():
    class TestDeleter:
        def delete_tracks(self, tracks):
            return {track
                    for track in tracks
                    if len(track.states) > 10}
    return TestDeleter()


@pytest.fixture()
def detector():
    class TestDetector:
        def detections_gen(self):
            time = datetime.datetime(2018, 1, 1, 14)
            for step in range(20):
                yield time, {Detection(StateVector([[step + 10*i]]))
                             for i in range(3)
                             if (step - i) % 5}
                time += datetime.timedelta(minutes=1)
            for step in range(3):
                yield time, set()
                time += datetime.timedelta(minutes=1)
    return TestDetector()


@pytest.fixture()
def data_associator():
    class TestDataAssociator:
        def associate(self, tracks, detections, time):
            associations = {}
            for track in tracks:
                prediction = StatePrediction(track.state_vector + 1)
                measurement_prediction = StateMeasurementPrediction(
                    prediction.state_vector)
                for detection in detections:
                    if np.array_equal(measurement_prediction.state_vector,
                                      detection.state_vector):
                        associations[track] = \
                            SingleMeasurementDistanceHypothesis(
                            prediction, detection, 0, measurement_prediction)
                        break
                else:
                    associations[track] = SingleMeasurementDistanceHypothesis(
                        prediction, None, None, 1)
            return associations
    return TestDataAssociator()


@pytest.fixture()
def updater():
    class TestUpdater:
        def update(self, hypothesis):
            return State(hypothesis.measurement.state_vector)
    return TestUpdater()
