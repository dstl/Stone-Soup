import datetime

import numpy as np

from ...models.transition.linear import ConstantVelocity
from ...predictor.ensemble import EnsemblePredictor
from ...types.state import GaussianState, EnsembleState
from ...types.array import StateVector, CovarianceMatrix


def test_ensemble():
    # Initialise a transition model
    transition_model = ConstantVelocity(noise_diff_coeff=0)

    # Define time related variables
    timestamp = datetime.datetime(2021, 2, 27, 17, 27, 48)
    timediff = 1  # 1 second
    new_timestamp = timestamp + datetime.timedelta(seconds=timediff)
    time_interval = new_timestamp - timestamp

    # Define prior state
    mean = StateVector([[10], [10]])
    covar = CovarianceMatrix(np.eye(2))
    gaussian_state = GaussianState(mean, covar, timestamp)
    num_vectors = 50
    prior_state = EnsembleState.from_gaussian_state(gaussian_state, num_vectors)
    prior_ensemble = prior_state.state_vector

    # Create Predictor object, run prediction
    predictor = EnsemblePredictor(transition_model)
    prediction = predictor.predict(prior_state, timestamp=new_timestamp)

    # Evaluate mean and covariance
    eval_ensemble = transition_model.matrix(timestamp=new_timestamp,
                                            time_interval=time_interval) @ prior_ensemble
    eval_mean = StateVector((np.average(eval_ensemble, axis=1)))
    eval_cov = np.cov(eval_ensemble)

    # Compare evaluated mean and covariance with predictor results
    assert np.allclose(prediction.mean, eval_mean)
    assert np.allclose(prediction.state_vector, eval_ensemble)
    assert np.allclose(prediction.covar, eval_cov)
    assert prediction.timestamp == new_timestamp
