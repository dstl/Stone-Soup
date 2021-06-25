# coding: utf-8
from datetime import datetime, timedelta

import numpy as np
import pytest
from scipy.stats import multivariate_normal

from ...models.transition.categorical import CategoricalTransitionModel
from ...models.transition.linear import RandomWalk, \
    CombinedLinearGaussianTransitionModel
from ...models.transition.tests.test_categorical import create_categorical, \
    create_categorical_matrix
from ...predictor.categorical import HMMPredictor
from ...predictor.composite import CompositePredictor
from ...predictor.kalman import KalmanPredictor, ExtendedKalmanPredictor, \
    UnscentedKalmanPredictor
from ...predictor.particle import ParticlePredictor, ParticleFlowKalmanPredictor
from ...types.array import StateVector
from ...types.array import StateVectors
from ...types.numeric import Probability
from ...types.particle import Particles
from ...types.prediction import CompositePrediction
from ...types.state import ParticleState, CategoricalState
from ...types.state import State, GaussianState, CompositeState


def create_transition_model(gaussian: bool, ndim_state: int):
    """Generate appropriate transition models of particular dimensions"""
    if gaussian:
        models = ndim_state * [RandomWalk(0.1)]
        return CombinedLinearGaussianTransitionModel(models)
    else:
        return CategoricalTransitionModel(create_categorical_matrix(ndim_state, ndim_state).T)


def create_state(gaussian: bool, particles: bool, ndim_state: int, timestamp: datetime):
    """Generate appropriate, random states of particular dimensions"""
    if gaussian:
        # create Gaussian state
        sv = StateVector(np.random.rand(ndim_state))
        cov = np.random.uniform(1, 10) * np.eye(ndim_state)
        if particles:
            # create particle state
            number_particles = 1000
            samples = multivariate_normal.rvs(sv.flatten(), cov, size=number_particles)
            particles = Particles(state_vector=StateVectors(samples.T),
                                  weight=np.array(
                                      [Probability(1 / number_particles)] * number_particles))
            return ParticleState(particles, timestamp=timestamp)
        return GaussianState(sv, cov, timestamp=timestamp)
    else:
        # create multinomial distribution state representative
        return CategoricalState(create_categorical(ndim_state), timestamp=timestamp)


def random_predictor_and_prior(num_predictors, timestamp):
    ndim_states = np.random.randint(2, 5, 6)

    sub_predictors = [
        KalmanPredictor(create_transition_model(True, ndim_states[0])),
        ExtendedKalmanPredictor(create_transition_model(True, ndim_states[1])),
        UnscentedKalmanPredictor(create_transition_model(True, ndim_states[2])),
        ParticlePredictor(create_transition_model(True, ndim_states[3])),
        ParticleFlowKalmanPredictor(create_transition_model(True, ndim_states[4])),
        HMMPredictor(create_transition_model(False, ndim_states[5]))
    ]

    sub_priors = [
        create_state(True, False, ndim_states[0], timestamp),
        create_state(True, False, ndim_states[1], timestamp),
        create_state(True, False, ndim_states[2], timestamp),
        create_state(True, True, ndim_states[3], timestamp),
        create_state(True, True, ndim_states[4], timestamp),
        create_state(False, False, ndim_states[5], timestamp)
    ]

    predictor = CompositePredictor(sub_predictors[:num_predictors])
    prior = CompositeState(sub_priors[:num_predictors])

    return predictor, prior


@pytest.mark.parametrize('num_predictors', [1, 2, 3, 4, 5, 6])
def test_composite_predictor(num_predictors):
    now = datetime.now()
    future = now + timedelta(seconds=5)

    # Test instantiation errors
    with pytest.raises(ValueError, match="sub-predictors must be defined as an ordered list"):
        CompositePredictor({KalmanPredictor(create_transition_model(True, 1))})

    with pytest.raises(ValueError, match="all sub-predictors must be a Predictor type"):
        CompositePredictor([1, 2, 3])

    # create random composite predictor and prior
    predictor, prior = random_predictor_and_prior(num_predictors, now)

    # Test transition model error
    with pytest.raises(NotImplementedError,
                       match="A composition of predictors have no defined transition model"):
        predictor.transition_model

    # Test predict errors
    with pytest.raises(ValueError,
                       match="CompositePredictor can only be used with CompositeState types"):
        predictor.predict(State([0]), future)

    with pytest.raises(ValueError,
                       match="CompositeState must be composed of same number of sub-states as "
                             "sub-predictors"):
        predictor.predict(CompositeState((num_predictors + 1) * [State([0])]))

    # Test predict
    prediction = predictor.predict(prior, timestamp=future)

    assert isinstance(prediction, CompositePrediction)
    assert len(prediction) == len(prior)

    sub_predictors = predictor.sub_predictors

    # Test iter
    for i, exp_sub_predictor in enumerate(predictor):
        assert exp_sub_predictor == sub_predictors[i]

    # Test len
    assert len(predictor) == num_predictors

    # Test get
    for i, expected_predictor in enumerate(sub_predictors):
        assert predictor[i] == expected_predictor

    predictor_slice = predictor[:num_predictors - 1]
    assert isinstance(predictor_slice, CompositePredictor)
    assert len(predictor_slice) == num_predictors - 1
    for i, expected_predictor in enumerate(sub_predictors[:num_predictors - 1]):
        assert predictor_slice[i] == expected_predictor

    # Test contains
    for sub_predictor in sub_predictors:
        assert sub_predictor in predictor

    # Test append
    new_sub_predictor = KalmanPredictor(RandomWalk(200))
    predictor.append(new_sub_predictor)
    assert new_sub_predictor in predictor
    assert len(predictor) == num_predictors + 1
