# -*- coding: utf-8 -*-
import datetime

import numpy as np
import pytest

from ...types.array import StateVector
from ...types.detection import MissedDetection, GaussianDetection
from ...types.hypothesis import SingleDistanceHypothesis, \
    SingleProbabilityHypothesis
from ...types.multihypothesis import MultipleHypothesis
from ...types.prediction import StateMeasurementPrediction, \
    GaussianStatePrediction
from ...types.track import Track
from ...types.update import GaussianStateUpdate


@pytest.fixture()
def initiator():
    class TestInitiator:
        def initiate(self, unassociated_detections):
            return {Track([detection])
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
                yield time, {GaussianDetection(
                    StateVector([[step + 10*i]]), [[10]], timestamp=time)
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
                prediction = GaussianStatePrediction(track.state_vector + 1,
                                                     [[5]], time)
                measurement_prediction = StateMeasurementPrediction(
                    prediction.state_vector)
                for detection in detections:
                    if np.array_equal(measurement_prediction.state_vector,
                                      detection.state_vector):
                        associations[track] = \
                            SingleDistanceHypothesis(
                            prediction, detection, 0, measurement_prediction)
                        break
                else:
                    associations[track] = SingleDistanceHypothesis(
                        prediction, None, None, 1)
            return associations
    return TestDataAssociator()


@pytest.fixture()
def data_mixture_associator():
    class TestDataMixtureAssociator:
        def associate(self, tracks, detections, time):
            associations = {}
            for track in tracks:
                prediction = GaussianStatePrediction(track.state_vector + 1,
                                                     [[5]], time)
                measurement_prediction = StateMeasurementPrediction(
                    prediction.state_vector)
                multihypothesis = []
                for detection in detections:
                    if np.array_equal(measurement_prediction.state_vector,
                                      detection.state_vector):
                        multihypothesis.append(
                            SingleProbabilityHypothesis(
                                prediction, detection,
                                measurement_prediction=measurement_prediction,
                                probability=0.9
                            ))
                        multihypothesis.append(
                            SingleProbabilityHypothesis(
                                prediction, MissedDetection(timestamp=time),
                                measurement_prediction=measurement_prediction,
                                probability=0.1
                            ))
                        break
                else:
                    multihypothesis.append(
                        SingleProbabilityHypothesis(
                            prediction, MissedDetection(timestamp=time),
                            measurement_prediction=measurement_prediction,
                            probability=0.1
                        ))
                associations[track] = MultipleHypothesis(multihypothesis)

            return associations
    return TestDataMixtureAssociator()


@pytest.fixture()
def updater():
    class TestUpdater:
        def update(self, hypothesis):
            return GaussianStateUpdate(hypothesis.measurement.state_vector,
                                       hypothesis.measurement.covar,
                                       hypothesis,
                                       0)
    return TestUpdater()
