from datetime import datetime, timedelta

import numpy as np
import pytest
from scipy.stats import multivariate_normal

from ...models.transition.categorical import MarkovianTransitionModel
from ...models.transition.linear import RandomWalk, \
    CombinedLinearGaussianTransitionModel
from ...predictor.categorical import HMMPredictor
from ...predictor.composite import CompositePredictor
from ...predictor.kalman import KalmanPredictor, ExtendedKalmanPredictor, \
    UnscentedKalmanPredictor
from ...predictor.particle import ParticlePredictor, ParticleFlowKalmanPredictor
from ...types.array import StateVector
from ...types.array import StateVectors
from ...types.numeric import Probability
from ...types.prediction import CompositePrediction
from ...types.state import ParticleState, CategoricalState
from ...types.state import State, GaussianState, CompositeState


def create_state(gaussian: bool, particles: bool, ndim_state: int, timestamp: datetime):
    """Create random states of particular dimensions"""

    if gaussian:
        # create Gaussian state
        sv = StateVector(np.random.rand(ndim_state))
        cov = np.random.uniform(1, 10) * np.eye(ndim_state)
        if particles:
            # create particle state
            number_particles = 1000
            samples = multivariate_normal.rvs(sv.flatten(), cov, size=number_particles)
            return ParticleState(state_vector=StateVectors(samples.T),
                                 weight=np.array(
                                     [Probability(1 / number_particles)] * number_particles),
                                 timestamp=timestamp)
        return GaussianState(sv, cov, timestamp=timestamp)
    else:
        # create categorical state
        return CategoricalState(np.random.rand(ndim_state), timestamp=timestamp)


def create_transition_model(gaussian: bool, ndim_state: int):
    """Create appropriate transition models of particular dimensions"""

    if gaussian:
        models = ndim_state * [RandomWalk(0.1)]
        return CombinedLinearGaussianTransitionModel(models)
    else:
        return MarkovianTransitionModel(np.random.rand(ndim_state, ndim_state))


def random_predictor_and_prior(num_predictors, timestamp):
    """Create a particular number of random predictors"""

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
    with pytest.raises(
            ValueError,
            match="Sub-predictors must be defined as an ordered list, not <class 'set'>"):
        CompositePredictor({KalmanPredictor(create_transition_model(True, 1))})

    with pytest.raises(ValueError, match="Cannot create an empty composite predictor"):
        CompositePredictor(list())

    with pytest.raises(ValueError, match="All sub-predictors must be a Predictor type"):
        CompositePredictor([1, 2, 3])

    # create random composite predictor and prior
    predictor, prior = random_predictor_and_prior(num_predictors, now)

    # Test transition model error
    with pytest.raises(NotImplementedError,
                       match="A composition of predictors has no defined transition model"):
        predictor.transition_model

    # Test predict errors
    with pytest.raises(ValueError,
                       match="CompositePredictor can only predict forward CompositeState types"):
        predictor.predict(State([0]), future)

    with pytest.raises(
            ValueError,
            match=f"Mismatch in number of prior sub-states {num_predictors + 1} and number of "
            f"sub-predictors {num_predictors}"):
        predictor.predict(CompositeState((num_predictors + 1) * [State([0])]))

    # Test predict
    prediction = predictor.predict(prior, timestamp=future)

    assert isinstance(prediction, CompositePrediction)
    assert len(prediction) == len(prior)

    sub_predictors = predictor.sub_predictors

    # Test contains
    for sub_predictor in sub_predictors:
        assert sub_predictor in predictor
    assert KalmanPredictor(create_transition_model(True, 5)) not in predictor
    assert 'a' not in predictor

    # Test get
    for i, expected_predictor in enumerate(sub_predictors):
        assert predictor[i] == expected_predictor
    # Test get slice
    if num_predictors > 1:
        predictor_slice = predictor[1:]
        assert isinstance(predictor_slice, CompositePredictor)
        assert predictor_slice.sub_predictors == sub_predictors[1:]

    # Test iter
    for i, exp_sub_predictor in enumerate(predictor):
        assert exp_sub_predictor == sub_predictors[i]

    # Test len
    assert len(predictor) == num_predictors
