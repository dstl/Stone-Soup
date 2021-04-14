# coding: utf-8
import datetime

import numpy as np
import pytest
from scipy.stats import multivariate_normal

from ...models.transition.classification import \
    BasicTimeInvariantClassificationTransitionModel
from ...models.transition.linear import RandomWalk, \
    CombinedLinearGaussianTransitionModel
from ...predictor.composite import CompositePredictor
from ...types.array import StateVector
from ...types.array import StateVectors
from ...types.numeric import Probability  # Similar to a float type
from ...types.particle import Particles
from ...types.prediction import CompositePrediction
from ...types.state import ParticleState
from ...predictor.classification import ClassificationPredictor
from ...predictor.kalman import KalmanPredictor, ExtendedKalmanPredictor, \
    UnscentedKalmanPredictor
from ...predictor.particle import ParticlePredictor, ParticleFlowKalmanPredictor
from ...types.state import State, GaussianState, CompositeState


def create_transition_model(gaussian: bool, ndim_state: int):
    """Generate appropriate transition models of particular dimensions"""
    if gaussian:
        models = ndim_state * [RandomWalk(0.1)]
        return CombinedLinearGaussianTransitionModel(models)
    else:
        F = np.random.rand(ndim_state, ndim_state)
        F = F / F.sum(axis=0)[np.newaxis, :]
        return BasicTimeInvariantClassificationTransitionModel(F)


def create_state(gaussian: bool, particles: bool, ndim_state: int):
    """Generate appropriate, random states of particular dimensions"""
    if gaussian:
        # create Gaussian state
        sv = StateVector(np.random.rand(ndim_state))
        cov = np.random.uniform(1, 10) * np.eye(ndim_state)
        if particles:
            # create particle state
            number_particles = 1000
            samples = multivariate_normal.rvs(sv.flatten(), cov, size=number_particles)
            print(sv.flatten(), cov, '\n\n')
            particles = Particles(state_vector=StateVectors(samples.T),
                                  weight=np.array(
                                      [Probability(1 / number_particles)] * number_particles))
            return ParticleState(particles)
        return GaussianState(sv, cov)
    else:
        # create multinomial distribution state representative
        total = 0
        sv = list()
        for i in range(ndim_state - 1):
            x = np.random.uniform(0, 1 - total)
            sv.append(x)
            total += x
        sv.append(1 - total)
        return State(sv)


def get_sub_predictors(num_predictors):
    possible_predictors = [KalmanPredictor, ExtendedKalmanPredictor, UnscentedKalmanPredictor,
                           ParticlePredictor, ParticleFlowKalmanPredictor,
                           ClassificationPredictor]
    sub_predictor_types = possible_predictors[:num_predictors]

    sub_predictors = list()
    for sub_predictor_type in sub_predictor_types:
        ndim_state = np.random.randint(2, 10)

        if sub_predictor_type == ClassificationPredictor:
            gaussian = False
        else:
            gaussian = True

        transition_model = create_transition_model(gaussian, ndim_state)

        sub_predictors.append(sub_predictor_type(transition_model))

    return sub_predictors


@pytest.mark.parametrize('num_predictors', [1, 2, 3, 4, 5, 6])
def test_composite_predictor(num_predictors):
    now = datetime.datetime.now()
    future = now + datetime.timedelta(seconds=5)

    # get random sub-predictors
    sub_predictors = get_sub_predictors(num_predictors)

    # create appropriate priors
    priors = list()
    for sub_predictor in sub_predictors:
        if isinstance(sub_predictor, ClassificationPredictor):
            gaussian = False
            particles = False
        else:
            gaussian = True
            if isinstance(sub_predictor, (ParticlePredictor, ParticleFlowKalmanPredictor)):
                particles = True
            else:
                particles = False

        state = create_state(gaussian, particles, sub_predictor.transition_model.ndim_state)
        state.timestamp = now
        priors.append(state)
    prior = CompositeState(priors)

    # test instantiation errors
    with pytest.raises(ValueError, match="sub-predictors must be defined as an ordered list"):
        CompositePredictor(set(sub_predictors))

    with pytest.raises(ValueError, match="all sub-predictors must be a Predictor type"):
        CompositePredictor(sub_predictors + [1, 2, 3])

    predictor = CompositePredictor(sub_predictors)

    # test transition model error
    with pytest.raises(NotImplementedError,
                       match="A composition of predictors have no defined transition model"):
        predictor.transition_model

    # test predict
    prediction = predictor.predict(prior, future)

    assert isinstance(prediction, CompositePrediction)
    assert len(prediction) == len(prior)

    # test predict errors
    with pytest.raises(ValueError,
                       match="CompositePredictor can only be used with CompositeState types"):
        predictor.predict(State([0]), future)

    with pytest.raises(ValueError,
                       match="CompositeState must be composed of same number of sub-states as "
                             "sub-predictors"):
        predictor.predict(CompositeState((num_predictors + 1) * [State([0])]))

    # test len
    assert len(predictor) == num_predictors

    # test get
    for i, expected_predictor in enumerate(sub_predictors):
        assert predictor[i] == expected_predictor

    new_sub_predictor = KalmanPredictor(RandomWalk(100))

    # test contains
    for sub_predictor in sub_predictors:
        assert sub_predictor in predictor

    # test set
    index = np.random.randint(num_predictors)
    predictor[index] = new_sub_predictor

    assert predictor[index] == new_sub_predictor

    # test del
    del predictor[index]
    assert new_sub_predictor not in predictor

    # test insert
    predictor.insert(index, new_sub_predictor)
    assert predictor[index] == new_sub_predictor
    assert len(predictor) == num_predictors

    # test append
    new_sub_predictor = KalmanPredictor(RandomWalk(200))
    predictor.append(new_sub_predictor)
    assert new_sub_predictor in predictor
    assert len(predictor) == num_predictors + 1
