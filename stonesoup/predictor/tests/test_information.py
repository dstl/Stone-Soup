# coding: utf-8
import datetime
import pytest
import numpy as np

from ...models.transition.linear import ConstantVelocity
from ...predictor.information import InfoFilterPredictor
from ...types.state import InformationState
from ...types.prediction import InformationStatePrediction
from numpy.linalg import inv


@pytest.mark.parametrize(
    "PredictorClass, transition_model, prior_mean, prior_covar",
    [
        (   # Standard Kalman
            InfoFilterPredictor,
            ConstantVelocity(noise_diff_coeff=0.1),
            np.array([[-6.45], [0.7]]),
            np.array([[4.1123, 0.0013],
                      [0.0013, 0.0365]])
        )
    ],
    ids=["standard"]
)
def test_information(PredictorClass, transition_model,
                     prior_mean, prior_covar):

    # Define time related variables
    timestamp = datetime.datetime.now()
    timediff = 2  # 2sec
    new_timestamp = timestamp + datetime.timedelta(seconds=timediff)
    time_interval = new_timestamp - timestamp

    # Define prior state
    # prior = GaussianState(prior_mean,
    #                       prior_covar,
    #                       timestamp=timestamp)

    # Define prior information state
    prior = InformationState(prior_mean, prior_covar, timestamp=timestamp)

    F = transition_model.matrix(timestamp=new_timestamp, time_interval=time_interval)
    print(F)
    M = inv(transition_model.matrix(timestamp=new_timestamp, time_interval=time_interval)).T\
        @prior.info_matrix @ inv(transition_model.matrix(timestamp=new_timestamp,
                                                         time_interval=time_interval))

    inv_Q = inv(transition_model.covar(time_interval=time_interval))
    C = M @ inv(M + inv_Q)
    L = np.identity(len(C)) - C
    info_matrix = L @ M @ L.T + C @ inv_Q @ C.T

    # Calculate evaluation variables
    eval_prediction = InformationStatePrediction(
        L @ inv(transition_model.matrix(
            timestamp=new_timestamp, time_interval=time_interval)).T @ prior_mean,
        info_matrix)

    # Initialise a Information filter predictor
    predictor = PredictorClass(transition_model=transition_model)

    # Perform and assert state prediction
    prediction = predictor.predict(prior=prior,
                                   timestamp=new_timestamp)

    assert(np.allclose(prediction.state_vector,
                       eval_prediction.state_vector, 0, atol=1.e-14))
    assert(np.allclose(prediction.info_matrix,
                       eval_prediction.info_matrix, 0, atol=1.e-14))
    assert(prediction.timestamp == new_timestamp)

    # TODO: Test with Control Model
