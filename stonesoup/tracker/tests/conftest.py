import datetime

import numpy as np
import pytest
from scipy.stats import multivariate_normal

from ...buffered_generator import BufferedGenerator
from ...reader import DetectionReader
from ...types.array import StateVector, StateVectors
from ...types.detection import MissedDetection, GaussianDetection
from ...types.hypothesis import SingleDistanceHypothesis, \
    SingleProbabilityHypothesis
from ...types.multihypothesis import MultipleHypothesis
from ...types.numeric import Probability
from ...types.prediction import StateMeasurementPrediction, \
    GaussianStatePrediction, GaussianMeasurementPrediction, ParticleStatePrediction, \
    ParticleMeasurementPrediction
from ...types.state import ParticleState
from ...types.track import Track
from ...types.update import GaussianStateUpdate, ParticleStateUpdate


@pytest.fixture()
def initiator():
    class TestInitiator:
        def initiate(self, detections, timestamp):
            return {Track([detection])
                    for detection in detections}
    return TestInitiator()


@pytest.fixture()
def particle_initiator(initiator):
    class TestParticleInitiator:
        def initiate(self, detections, timestamp):
            tracks = set()
            for detection in detections:
                samples = multivariate_normal.rvs(detection.state_vector.ravel(),
                                                  np.eye(detection.state_vector.shape[0])*0.01,
                                                  size=100)
                state_vector = StateVectors(np.atleast_2d(samples))
                state = ParticleState(state_vector, weight=np.ones(100)/100,
                                      timestamp=detection.timestamp)
                tracks.add(Track([state]))
            return tracks
    return TestParticleInitiator()


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
    class TestDetector(DetectionReader):
        @BufferedGenerator.generator_method
        def detections_gen(self):
            time = datetime.datetime(2018, 1, 1, 14)
            for step in range(20):
                yield time, {GaussianDetection(
                    StateVector([[step + (10*i)]]), [[2]], timestamp=time)
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
        def associate(self, tracks, detections, timestamp):
            associations = {}
            for track in tracks:
                prediction = GaussianStatePrediction(track.state_vector + 1,
                                                     [[5]], timestamp)
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
        def associate(self, tracks, detections, timestamp):
            associations = {}
            for track in tracks:
                prediction = GaussianStatePrediction(track.state_vector + 1,
                                                     [[5]], timestamp)
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
                                prediction, MissedDetection(timestamp=timestamp),
                                measurement_prediction=measurement_prediction,
                                probability=0.1
                            ))
                        break
                else:
                    multihypothesis.append(
                        SingleProbabilityHypothesis(
                            prediction, MissedDetection(timestamp=timestamp),
                            measurement_prediction=measurement_prediction,
                            probability=0.1
                        ))
                associations[track] = MultipleHypothesis(multihypothesis)

            return associations
    return TestDataMixtureAssociator()


@pytest.fixture()
def data_particle_associator():
    class TestDataParticleAssociator:
        def associate(self, tracks, detections, timestamp):
            associations = {}
            for track in tracks:
                prediction = ParticleStatePrediction(track.state_vector + 1,
                                                     weight=track.weight,
                                                     timestamp=timestamp)
                measurement_prediction = StateMeasurementPrediction(
                    prediction.mean)
                multihypothesis = []
                for detection in detections:
                    if np.allclose(measurement_prediction.state_vector,
                                   detection.state_vector):
                        multihypothesis.append(
                            SingleProbabilityHypothesis(
                                prediction, detection,
                                measurement_prediction=measurement_prediction,
                                probability=Probability(0.9)
                            ))
                        multihypothesis.append(
                            SingleProbabilityHypothesis(
                                prediction, MissedDetection(timestamp=timestamp),
                                measurement_prediction=measurement_prediction,
                                probability=0.1
                            ))
                        break
                else:
                    multihypothesis.append(
                        SingleProbabilityHypothesis(
                            prediction, MissedDetection(timestamp=timestamp),
                            measurement_prediction=measurement_prediction,
                            probability=0.1
                        ))
                associations[track] = MultipleHypothesis(multihypothesis)

            return associations
    return TestDataParticleAssociator()


@pytest.fixture()
def updater():
    class TestUpdater:
        def update(self, hypothesis):
            return GaussianStateUpdate(hypothesis.measurement.state_vector,
                                       hypothesis.prediction.covar,
                                       hypothesis,
                                       0)

        def predict_measurement(self, state_prediction,
                                measurement_model=None, **kwargs):
            return GaussianMeasurementPrediction(
                    state_prediction.state_vector,
                    state_prediction.covar,
                    state_prediction.timestamp)

    return TestUpdater()


@pytest.fixture()
def particle_updater():
    class TestParticleUpdater:
        resampler = None

        def update(self, hypothesis):
            return ParticleStateUpdate(hypothesis.measurement.state_vector,
                                       weight=hypothesis.prediction.weight,
                                       hypothesis=hypothesis,
                                       timestamp=hypothesis.measurement.timestamp)

        def predict_measurement(self, state_prediction,
                                measurement_model=None, **kwargs):
            return ParticleMeasurementPrediction(
                    state_prediction.state_vector,
                    state_prediction.weight,
                    state_prediction.timestamp)

    return TestParticleUpdater()


@pytest.fixture()
def predictor():
    class TestGaussianPredictor:
        def predict(self, prior, control_input=None, timestamp=None, **kwargs):
            return GaussianStatePrediction(prior.state_vector+1,
                                           prior.covar*2, timestamp)
    return TestGaussianPredictor()
