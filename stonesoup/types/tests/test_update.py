# -*- coding: utf-8 -*-
import datetime

import numpy as np
import pytest

from ...types.detection import Detection
from ...types.hypothesis import SingleHypothesis
from ...types.particle import Particle
from ...types.prediction import (
    GaussianStatePrediction, GaussianMeasurementPrediction,
    StatePrediction, StateMeasurementPrediction,
    ParticleStatePrediction, ParticleMeasurementPrediction, ASDGaussianStatePrediction,
    ASDGaussianMeasurementPrediction)
from ...types.update import (
    StateUpdate, GaussianStateUpdate, ParticleStateUpdate, ASDGaussianStateUpdate)


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


def test_asdgaussianstateupdate():
    """GaussianStateUpdate test"""

    with pytest.raises(TypeError):
        ASDGaussianStateUpdate()

    state_vector = np.array([[-1.8513], [0.9994], [0], [0]]) * 1e4
    covar = np.ones([4, 4])
    timestamp = datetime.datetime.now()

    prediction = ASDGaussianStatePrediction(multi_state_vector=state_vector,
                                         multi_covar=covar,
                                         timestamps=[timestamp], act_timestamp=timestamp)
    meas_pred = ASDGaussianMeasurementPrediction(
        multi_state_vector=np.array([[2], [2], [1.5], [1.5]]),
        multi_covar=np.ones([4, 4]) * 2,
        timestamps=[timestamp])
    measurement = Detection(state_vector=np.array([[5], [7]]),
                            timestamp=timestamp)
    state_update = ASDGaussianStateUpdate(multi_state_vector=state_vector,
                                       multi_covar=covar,
                                       hypothesis=SingleHypothesis(
                                           prediction=prediction,
                                           measurement=measurement,
                                           measurement_prediction=meas_pred),
                                       timestamps=[timestamp])

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
    prediction = ParticleStatePrediction(particles=particles,
                                         timestamp=timestamp)
    meas_pred = ParticleMeasurementPrediction(
        particles=particles,
        timestamp=timestamp)
    measurement = Detection(state_vector=np.array([[5], [7]]),
                            timestamp=timestamp)
    state_update = ParticleStateUpdate(particles,
                                       SingleHypothesis(
                                           prediction=prediction,
                                           measurement=measurement,
                                           measurement_prediction=meas_pred),
                                       timestamp=timestamp)

    eval_mean = np.mean(np.hstack([i.state_vector for i in particles]),
                        axis=1).reshape(2, 1)
    assert np.allclose(eval_mean, state_update.mean)
    assert np.all([particles[i].state_vector ==
                   state_update.particles[i].state_vector for i in range(9)])
    assert np.array_equal(prediction, state_update.hypothesis.prediction)
    assert np.array_equal(meas_pred,
                          state_update.hypothesis.measurement_prediction)
    assert np.array_equal(measurement, state_update.hypothesis.measurement)
    assert np.array_equal(timestamp, state_update.timestamp)
