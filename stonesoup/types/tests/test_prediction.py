import datetime
import pickle

import numpy as np
import pytest
from scipy.stats import multivariate_normal

from ..hypothesis import SingleHypothesis
from ..prediction import (
    Prediction, MeasurementPrediction,
    StatePrediction, StateMeasurementPrediction,
    GaussianStatePrediction, GaussianMeasurementPrediction,
    SqrtGaussianStatePrediction, TaggedWeightedGaussianStatePrediction,
    ParticleStatePrediction, ParticleMeasurementPrediction,
    RaoBlackwellisedParticleStatePrediction, MultiModelParticleStatePrediction,
    BernoulliParticleStatePrediction,
    ASDGaussianStatePrediction, ASDGaussianMeasurementPrediction, KernelParticleStatePrediction)
from ..state import (
    State, GaussianState, SqrtGaussianState, TaggedWeightedGaussianState, ParticleState)
from ..track import Track
from ..update import StateUpdate
from ..array import StateVectors


def test_stateprediction():
    """ StatePrediction test """

    with pytest.raises(TypeError):
        StatePrediction()

    state_vector = np.array([[-1.8513], [0.9994], [0], [0]]) * 1e4
    timestamp = datetime.datetime.now()

    state_prediction = StatePrediction(state_vector, timestamp)
    assert np.array_equal(state_vector, state_prediction.state_vector)
    assert timestamp == state_prediction.timestamp


def test_prediction_prior():
    state0 = State([[1, 2, 3]])
    pred1 = StatePrediction([[2, 3, 1]], prior=state0)
    update1 = StateUpdate([[2, 3, 1]], hypothesis=SingleHypothesis(pred1, None))
    pred2 = StatePrediction([[3, 1, 2]], prior=update1)

    assert pred1.prior is state0
    assert update1.hypothesis.prediction.prior is state0
    assert pred2.prior is update1

    del state0

    assert pred1.prior is None
    assert update1.hypothesis.prediction.prior is None
    assert pred2.prior is update1

    pickle.dumps(pred1)


@pytest.mark.parametrize(
    'particle_class', [ParticleStatePrediction, MultiModelParticleStatePrediction,
                       RaoBlackwellisedParticleStatePrediction, BernoulliParticleStatePrediction])
def test_particle_parent_parent(particle_class):
    state1 = particle_class([[1, 2, 3]], weight=np.full((3, ), 1/3))
    state2 = particle_class([[2, 3, 1]], weight=np.full((3, ), 1/3), parent=state1)
    state3 = particle_class([[3, 1, 2]], weight=np.full((3, ), 1/3), prior=state2, parent=state2)

    del state1  # All remaining references should be weak

    assert state3.prior.parent is None

    pickle.dumps(state3)


def test_statemeasurementprediction():
    """ MeasurementPrediction test """

    with pytest.raises(TypeError):
        StateMeasurementPrediction()

    state_vector = np.array([[-1.8513], [0.9994], [0], [0]]) * 1e4
    timestamp = datetime.datetime.now()

    measurement_prediction = StateMeasurementPrediction(state_vector,
                                                        timestamp)
    assert np.array_equal(state_vector, measurement_prediction.state_vector)
    assert timestamp == measurement_prediction.timestamp


def test_gaussianstateprediction():
    """ GaussianStatePrediction test """

    with pytest.raises(TypeError):
        GaussianStatePrediction()

    mean = np.array([[-1.8513], [0.9994], [0], [0]]) * 1e4
    covar = np.array([[2.2128, 0, 0, 0],
                      [0.0002, 2.2130, 0, 0],
                      [0.3897, -0.00004, 0.0128, 0],
                      [0, 0.3897, 0.0013, 0.0135]]) * 1e3
    timestamp = datetime.datetime.now()

    with pytest.raises(TypeError):
        GaussianStatePrediction(mean)

    # Test state prediction
    state_prediction = GaussianStatePrediction(mean, covar, timestamp)
    assert np.array_equal(mean, state_prediction.mean)
    assert np.array_equal(covar, state_prediction.covar)
    assert state_prediction.ndim == mean.shape[0]
    assert state_prediction.timestamp == timestamp


def test_gaussianmeasurementprediction():
    """ GaussianMeasurementPrediction test """

    with pytest.raises(TypeError):
        GaussianMeasurementPrediction()

    mean = np.array([[-1.8513], [0.9994], [0], [0]]) * 1e4
    covar = np.array([[2.2128, 0, 0, 0],
                      [0.0002, 2.2130, 0, 0],
                      [0.3897, -0.00004, 0.0128, 0],
                      [0, 0.3897, 0.0013, 0.0135]]) * 1e3
    cross_covar = np.array([[2.2128, 0, 0, 0],
                            [0.0002, 2.2130, 0, 0],
                            [0.3897, -0.00004, 0.0128, 0],
                            [0, 0.3897, 0.0013, 0.0135]]) * 1e3
    wrong_cross_covar = np.array([[2.2128, 0, 0],
                                  [0.0002, 2.2130, 0],
                                  [0.3897, -0.00004, 0.0128],
                                  [0, 0.3897, 0.0013]]) * 1e3
    timestamp = datetime.datetime.now()

    with pytest.raises(TypeError):
        GaussianMeasurementPrediction(mean)

    with pytest.raises(ValueError):
        GaussianMeasurementPrediction(mean, covar,
                                      cross_covar=wrong_cross_covar,
                                      timestamp=timestamp)

    # Test measurement prediction initiation without cross_covar
    measurement_prediction = GaussianMeasurementPrediction(
        mean, covar,
        timestamp=timestamp)
    assert np.array_equal(mean, measurement_prediction.mean)
    assert np.array_equal(covar, measurement_prediction.covar)
    assert measurement_prediction.ndim == mean.shape[0]
    assert measurement_prediction.timestamp == timestamp
    assert measurement_prediction.cross_covar is None

    # Test measurement prediction initiation with cross_covar
    measurement_prediction = GaussianMeasurementPrediction(
        mean, covar,
        cross_covar=cross_covar,
        timestamp=timestamp)
    assert np.array_equal(mean, measurement_prediction.mean)
    assert np.array_equal(covar, measurement_prediction.covar)
    assert np.array_equal(cross_covar, measurement_prediction.cross_covar)
    assert measurement_prediction.ndim == mean.shape[0]
    assert measurement_prediction.timestamp == timestamp


@pytest.mark.parametrize('prediction_type', (Prediction, MeasurementPrediction))
def test_from_state(prediction_type):
    state = State([[0]], timestamp=datetime.datetime.now())
    prediction = prediction_type.from_state(state, [[1]])
    if prediction_type is Prediction:
        assert isinstance(prediction, StatePrediction)
    else:
        assert isinstance(prediction, StateMeasurementPrediction)
    assert prediction.timestamp == state.timestamp
    assert prediction.state_vector[0] == 1

    state = GaussianState([[0]], [[2]], timestamp=datetime.datetime.now())
    prediction = prediction_type.from_state(state, [[1]], [[3]])
    if prediction_type is Prediction:
        assert isinstance(prediction, GaussianStatePrediction)
    else:
        assert isinstance(prediction, GaussianMeasurementPrediction)
    assert prediction.timestamp == state.timestamp
    assert prediction.state_vector[0] == 1
    assert prediction.covar[0] == 3

    state = SqrtGaussianState([[0]], [[2]], timestamp=datetime.datetime.now())
    if prediction_type is Prediction:
        prediction = prediction_type.from_state(state, [[1]], [[np.sqrt(3)]])
        assert isinstance(prediction, SqrtGaussianStatePrediction)
    else:
        prediction = prediction_type.from_state(state, [[1]], [[3]])
        assert isinstance(prediction, GaussianMeasurementPrediction)
    assert prediction.timestamp == state.timestamp
    assert prediction.state_vector[0] == 1
    assert prediction.covar[0] == pytest.approx(3)

    state = TaggedWeightedGaussianState(
        [[0]], [[2]], weight=0.5, timestamp=datetime.datetime.now())
    prediction = prediction_type.from_state(state, [[1]], [[3]])
    if prediction_type is Prediction:
        assert isinstance(prediction, TaggedWeightedGaussianStatePrediction)
    else:
        assert isinstance(prediction, GaussianMeasurementPrediction)
    assert prediction.timestamp == state.timestamp
    assert prediction.state_vector[0] == 1
    assert prediction.covar[0] == 3
    if prediction_type is Prediction:
        assert prediction.weight == 0.5
        assert prediction.tag == state.tag

    state = ParticleState(StateVectors([[1]]), weight=0.5, timestamp=datetime.datetime.now())
    prediction = prediction_type.from_state(state)
    if prediction_type is Prediction:
        assert isinstance(prediction, ParticleStatePrediction)
    else:
        assert isinstance(prediction, ParticleMeasurementPrediction)
    assert prediction.timestamp == state.timestamp
    assert prediction.state_vector[0] == 1

    with pytest.raises(TypeError, match=f'{prediction_type.__name__} type not defined for str'):
        prediction_type.from_state("a", state_vector=2)


@pytest.mark.parametrize('prediction_type', (Prediction, MeasurementPrediction))
def test_from_state_sequence(prediction_type):
    sequence = Track([GaussianState([[0]], [[2]], timestamp=datetime.datetime.now())])
    prediction = prediction_type.from_state(sequence, [[1]], [[3]])
    if prediction_type is Prediction:
        assert isinstance(prediction, GaussianStatePrediction)
    else:
        assert isinstance(prediction, GaussianMeasurementPrediction)
    assert prediction.timestamp == sequence.timestamp
    assert prediction.state_vector[0] == 1
    assert prediction.covar[0] == 3


def test_asdgaussianstateprediction():
    """ GaussianStatePrediction test """

    with pytest.raises(TypeError):
        ASDGaussianStatePrediction()

    mean = np.array([[-1.8513], [0.9994], [0], [0]]) * 1e4
    covar = np.array([[2.2128, 0, 0, 0],
                      [0.0002, 2.2130, 0, 0],
                      [0.3897, -0.00004, 0.0128, 0],
                      [0, 0.3897, 0.0013, 0.0135]]) * 1e3
    timestamp = datetime.datetime.now()

    with pytest.raises(TypeError):
        ASDGaussianStatePrediction(mean)

    # Test state prediction
    state_prediction = ASDGaussianStatePrediction(
        multi_state_vector=mean, multi_covar=covar,
        timestamps=[timestamp], act_timestamp=timestamp)
    assert np.array_equal(mean, state_prediction.mean)
    assert np.array_equal(covar, state_prediction.covar)
    assert state_prediction.ndim == mean.shape[0]
    assert state_prediction.timestamp == timestamp


def test_asdgaussianmeasurementprediction():
    """ GaussianMeasurementPrediction test """

    with pytest.raises(TypeError):
        ASDGaussianMeasurementPrediction()

    mean = np.array([[-1.8513], [0.9994], [0], [0]]) * 1e4
    covar = np.array([[2.2128, 0, 0, 0],
                      [0.0002, 2.2130, 0, 0],
                      [0.3897, -0.00004, 0.0128, 0],
                      [0, 0.3897, 0.0013, 0.0135]]) * 1e3
    cross_covar = np.array([[2.2128, 0, 0, 0],
                            [0.0002, 2.2130, 0, 0],
                            [0.3897, -0.00004, 0.0128, 0],
                            [0, 0.3897, 0.0013, 0.0135]]) * 1e3
    timestamp = datetime.datetime.now()

    with pytest.raises(TypeError):
        ASDGaussianMeasurementPrediction(mean)

    # Test measurement prediction initiation with cross_covar
    measurement_prediction = ASDGaussianMeasurementPrediction(
        mean, multi_covar=covar,
        cross_covar=cross_covar,
        timestamps=timestamp)
    assert np.array_equal(mean, measurement_prediction.mean)
    assert np.array_equal(covar, measurement_prediction.covar)
    assert np.array_equal(cross_covar, measurement_prediction.cross_covar)
    assert measurement_prediction.ndim == mean.shape[0]
    assert measurement_prediction.timestamp == timestamp


def test_kernel_particle_state_prediction():
    number_particles = 4
    rng = np.random.RandomState(50)
    samples = multivariate_normal.rvs([0, 0, 0, 0],
                                      np.diag([0.01, 0.005, 0.1, 0.5]) ** 2,
                                      size=number_particles,
                                      random_state=rng)

    state_vector = StateVectors(samples.T)
    weights = np.array([1 / number_particles] * number_particles)
    timestamp = datetime.datetime.now()

    prediction = KernelParticleStatePrediction(state_vector=state_vector,
                                               weight=weights,
                                               timestamp=timestamp)

    assert np.array_equal(state_vector, prediction.state_vector)
    assert np.array_equal(weights, prediction.weight)
    assert np.array_equal(np.diag(weights), prediction.kernel_covar)
    assert timestamp == prediction.timestamp
