import numpy as np

from datetime import datetime, timedelta

from ..nonlinear import DecayTransition
from ....types.state import DecayState
from ....types.array import StateVector, StateVectors, CovarianceMatrix


def test_decaytransition():

    decay_state = DecayState(StateVector([1, 0, 1]), halflife=timedelta(seconds=0.5),
                             timestamp=datetime(2024, 1, 1, 12, 0, 0))
    decay_model = DecayTransition()

    # Test that nothing happens if the time interval is zero
    new_state_vector = decay_model.function(decay_state, time_interval=timedelta(seconds=0))
    assert np.all(new_state_vector == decay_state.state_vector)

    # Test that the decay constant is returned correctly
    assert decay_model.decay_const(decay_state) == np.log(2)/decay_state.halflife.total_seconds()

    # Test that the covariance is returned correctly
    covar = decay_model.prob_decay(decay_state, timedelta(seconds=1)) * \
        (1 - decay_model.prob_decay(decay_state, timedelta(seconds=1)))

    assert np.allclose(CovarianceMatrix(np.diag(decay_state.state_vector.flatten()*covar)),
                       decay_model.covar(decay_state, timedelta(seconds=1)))

    # And test that its shape is correct
    assert decay_model.covar(decay_state, timedelta(seconds=1)).shape == (3, 3)

    # Test the pdf method returns a probability between 0 and 1
    new_state = DecayState(StateVector([1, 0, 0]), halflife=timedelta(seconds=0.5),
                           timestamp=datetime(2024, 1, 1, 12, 0, 1))
    prob = decay_model.pdf(new_state, decay_state, timedelta(seconds=1))
    assert 0 <= prob <= 1

    # Test the rvs method returns a StateVectors object with the correct shape
    random_state = decay_model.rvs(decay_state, timedelta(seconds=1), num_samples=2)
    assert isinstance(random_state, StateVectors)
    assert random_state.shape[1] == 2
