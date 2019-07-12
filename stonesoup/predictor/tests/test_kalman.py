# coding: utf-8
import datetime
import pytest
import numpy as np

from ...models.transition.linear import ConstantVelocity
from ...predictor.kalman import (
    KalmanPredictor, ExtendedKalmanPredictor, UnscentedKalmanPredictor)
from ...types.prediction import GaussianStatePrediction
from ...types.state import GaussianState


@pytest.mark.parametrize(
    "PredictorClass, transition_model, prior_mean, prior_covar",
    [
        (   # Standard Kalman
            KalmanPredictor,
            ConstantVelocity(noise_diff_coeff=0.1),
            np.array([[-6.45], [0.7]]),
            np.array([[4.1123, 0.0013],
                      [0.0013, 0.0365]])
        ),
        (   # Extended Kalman
            ExtendedKalmanPredictor,
            ConstantVelocity(noise_diff_coeff=0.1),
            np.array([[-6.45], [0.7]]),
            np.array([[4.1123, 0.0013],
                      [0.0013, 0.0365]])
        ),
        (   # Unscented Kalman
            UnscentedKalmanPredictor,
            ConstantVelocity(noise_diff_coeff=0.1),
            np.array([[-6.45], [0.7]]),
            np.array([[4.1123, 0.0013],
                      [0.0013, 0.0365]])
        )
    ],
    ids=["standard", "extended", "unscented"]
)
def test_kalman(PredictorClass, transition_model,
                prior_mean, prior_covar):

    # Define time related variables
    timestamp = datetime.datetime.now()
    timediff = 2  # 2sec
    new_timestamp = timestamp + datetime.timedelta(seconds=timediff)
    time_interval = new_timestamp - timestamp

    # Define prior state
    prior = GaussianState(prior_mean,
                          prior_covar,
                          timestamp=timestamp)

    # Calculate evaluation variables
    eval_prediction = GaussianStatePrediction(
        transition_model.matrix(timestamp=new_timestamp,
                                time_interval=time_interval)@prior.mean,
        transition_model.matrix(timestamp=new_timestamp,
                                time_interval=time_interval)
        @prior.covar
        @transition_model.matrix(timestamp=new_timestamp,
                                 time_interval=time_interval).T
        + transition_model.covar(timestamp=new_timestamp,
                                 time_interval=time_interval))

    # Initialise a kalman predictor
    predictor = PredictorClass(transition_model=transition_model)

    # Perform and assert state prediction
    prediction = predictor.predict(prior=prior,
                                   timestamp=new_timestamp)

    assert(np.allclose(prediction.mean,
                       eval_prediction.mean, 0, atol=1.e-14))
    assert(np.allclose(prediction.covar,
                       eval_prediction.covar, 0, atol=1.e-14))
    assert(prediction.timestamp == new_timestamp)

    # TODO: Test with Control Model
