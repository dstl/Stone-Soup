import datetime

import numpy as np

from ..asd import ASDKalmanPredictor
from ...models.transition.linear import ConstantVelocity
from ...types.prediction import ASDGaussianStatePrediction
from ...types.state import ASDGaussianState


def test_asdkalman():
    # simplified. In this case only the normal prediction is tested.
    # There is no testing of the correlations
    transition_model = ConstantVelocity(noise_diff_coeff=0.1)
    prior_mean = np.array([[-6.45], [0.7]])
    prior_covar = np.array([[4.1123, 0.0013],
                            [0.0013, 0.0365]])
    timestamp = datetime.datetime.now()
    timediff = 2  # 2sec
    new_timestamp = timestamp + datetime.timedelta(seconds=timediff)
    time_interval = new_timestamp - timestamp

    # Define prior state
    prior = ASDGaussianState(prior_mean, multi_covar=prior_covar,
                             timestamps=timestamp)

    # Calculate evaluation variables
    eval_prediction = ASDGaussianStatePrediction(
        transition_model.matrix(timestamp=new_timestamp,
                                time_interval=time_interval) @ prior.mean,
        multi_covar=transition_model.matrix(timestamp=new_timestamp,
                                            time_interval=time_interval)
        @ prior.covar
        @ transition_model.matrix(timestamp=new_timestamp,
                                  time_interval=time_interval).T
        + transition_model.covar(timestamp=new_timestamp,
                                 time_interval=time_interval),
        timestamps=[new_timestamp], act_timestamp=new_timestamp)

    # Initialise a kalman predictor
    predictor = ASDKalmanPredictor(transition_model=transition_model)

    # Perform and assert state prediction
    prediction = predictor.predict(prior=prior,
                                   timestamp=new_timestamp)
    assert (np.allclose(prediction.mean,
                        eval_prediction.mean, 0, atol=1.e-14))
    assert (np.allclose(prediction.covar,
                        eval_prediction.covar, 0, atol=1.e-14))
    assert (prediction.timestamp == new_timestamp)
