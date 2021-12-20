import datetime

import numpy as np
import pytest

from ..array import StateVector
from ..detection import Detection
from ..hypothesis import SingleHypothesis
from ..particle import Particle
from ..prediction import (
    GaussianStatePrediction, GaussianMeasurementPrediction,
    StatePrediction, StateMeasurementPrediction,
    ParticleStatePrediction, ParticleMeasurementPrediction)
from ..state import (
    State, GaussianState, SqrtGaussianState, TaggedWeightedGaussianState, ParticleState)
from ..track import Track
from ..update import (
    Update, StateUpdate, GaussianStateUpdate, SqrtGaussianStateUpdate, ParticleStateUpdate,
    TaggedWeightedGaussianStateUpdate)


def test_stateupdate():
    """ StateUpdate test """

    with pytest.raises(TypeError):
        StateUpdate()

    state_vector = np.array([[-1.8513], [0.9994], [0], [0]]) * 1e4
    timestamp = datetime.datetime.now()
    prediction = StatePrediction(state_vector=np.array([[1], [1],
                                                        [0.5], [0.5]]),
                                 timestamp=timestamp)
    measurement_prediction = StateMeasurementPrediction(
        state_vector=np.array([[2], [2], [1.5], [1.5]]),
        timestamp=timestamp)
    measurement = Detection(state_vector=np.array([[5], [7]]),
                            timestamp=timestamp)

    state_update = StateUpdate(
        state_vector,
        SingleHypothesis(
            prediction=prediction,
            measurement=measurement,
            measurement_prediction=measurement_prediction),
        timestamp=timestamp)

    assert np.array_equal(state_vector, state_update.state_vector)
    assert np.array_equal(prediction, state_update.hypothesis.prediction)
    assert np.array_equal(measurement_prediction,
                          state_update.hypothesis.measurement_prediction)
    assert np.array_equal(measurement, state_update.hypothesis.measurement)
    assert timestamp == state_update.timestamp


def test_gaussianstateupdate():
    """GaussianStateUpdate test"""

    with pytest.raises(TypeError):
        GaussianStateUpdate()

    state_vector = np.array([[-1.8513], [0.9994], [0], [0]]) * 1e4
    covar = np.ones([4, 4])
    timestamp = datetime.datetime.now()

    prediction = GaussianStatePrediction(state_vector=state_vector,
                                         covar=covar,
                                         timestamp=timestamp)
    meas_pred = GaussianMeasurementPrediction(
        state_vector=np.array([[2], [2], [1.5], [1.5]]),
        covar=np.ones([4, 4]) * 2,
        timestamp=timestamp)
    measurement = Detection(state_vector=np.array([[5], [7]]),
                            timestamp=timestamp)
    state_update = GaussianStateUpdate(state_vector,
                                       covar,
                                       SingleHypothesis(
                                           prediction=prediction,
                                           measurement=measurement,
                                           measurement_prediction=meas_pred),
                                       timestamp=timestamp)

    assert np.array_equal(state_vector, state_update.state_vector)
    assert np.array_equal(covar, state_update.covar)
    assert np.array_equal(prediction, state_update.hypothesis.prediction)
    assert np.array_equal(meas_pred,
                          state_update.hypothesis.measurement_prediction)
    assert np.array_equal(measurement, state_update.hypothesis.measurement)
    assert np.array_equal(timestamp, state_update.timestamp)


def test_particlestateupdate():
    """ParticleStateUpdate test"""

    particles = [Particle(np.array([[10], [10]]),
                          1 / 9),
                 Particle(np.array([[10], [20]]),
                          1 / 9),
                 Particle(np.array([[10], [30]]),
                          1 / 9),
                 Particle(np.array([[20], [10]]),
                          1 / 9),
                 Particle(np.array([[20], [20]]),
                          1 / 9),
                 Particle(np.array([[20], [30]]),
                          1 / 9),
                 Particle(np.array([[30], [10]]),
                          1 / 9),
                 Particle(np.array([[30], [20]]),
                          1 / 9),
                 Particle(np.array([[30], [30]]),
                          1 / 9),
                 ]
    timestamp = datetime.datetime.now()
    prediction = ParticleStatePrediction(None, particle_list=particles, timestamp=timestamp)
    meas_pred = ParticleMeasurementPrediction(None, particle_list=particles, timestamp=timestamp)
    measurement = Detection(state_vector=np.array([[5], [7]]),
                            timestamp=timestamp)
    state_update = ParticleStateUpdate(None, SingleHypothesis(prediction=prediction,
                                                              measurement=measurement,
                                                              measurement_prediction=meas_pred),
                                       particle_list=particles, timestamp=timestamp)

    eval_mean = np.mean(np.hstack([i.state_vector for i in particles]),
                        axis=1).reshape(2, 1)
    assert np.allclose(eval_mean, state_update.mean)
    assert np.all([particles[i].state_vector ==
                   StateVector(state_update.state_vector[:, i]) for i in range(9)])
    assert prediction == state_update.hypothesis.prediction
    assert meas_pred == state_update.hypothesis.measurement_prediction
    assert measurement == state_update.hypothesis.measurement
    assert timestamp == state_update.timestamp


def test_from_state():
    state = State([[0]], timestamp=datetime.datetime.now())
    update = Update.from_state(state, [[1]], hypothesis=None)
    assert isinstance(update, StateUpdate)
    assert update.timestamp == state.timestamp
    assert update.state_vector[0] == 1

    state = GaussianState([[0]], [[2]], timestamp=datetime.datetime.now())
    update = Update.from_state(state, [[1]], [[3]], hypothesis=None)
    assert isinstance(update, GaussianStateUpdate)
    assert update.timestamp == state.timestamp
    assert update.state_vector[0] == 1
    assert update.covar[0] == 3

    state = SqrtGaussianState([[0]], [[2]], timestamp=datetime.datetime.now())
    update = Update.from_state(state, [[1]], [[np.sqrt(3)]], hypothesis=None)
    assert isinstance(update, SqrtGaussianStateUpdate)
    assert update.timestamp == state.timestamp
    assert update.state_vector[0] == 1
    assert update.covar[0] == pytest.approx(3)

    state = TaggedWeightedGaussianState(
        [[0]], [[2]], weight=0.5, timestamp=datetime.datetime.now())
    update = Update.from_state(state, [[1]], [[3]], hypothesis=None)
    assert isinstance(update, TaggedWeightedGaussianStateUpdate)
    assert update.timestamp == state.timestamp
    assert update.state_vector[0] == 1
    assert update.covar[0] == 3

    state = ParticleState([Particle([[0]], weight=0.5)], timestamp=datetime.datetime.now())
    update = Update.from_state(state, None, particle_list=[Particle([[1]], weight=0.5)],
                               hypothesis=None)
    assert isinstance(update, ParticleStateUpdate)
    assert update.timestamp == state.timestamp
    assert update.state_vector[0] == 1

    with pytest.raises(TypeError, match='Update type not defined for str'):
        Update.from_state("a", state_vector=2)


def test_from_state_sequence():
    sequence = Track([GaussianState([[0]], [[2]], timestamp=datetime.datetime.now())])
    update = Update.from_state(sequence, [[1]], [[3]], hypothesis=None)
    assert isinstance(update, GaussianStateUpdate)
    assert update.timestamp == sequence.timestamp
    assert update.state_vector[0] == 1
    assert update.covar[0] == 3
