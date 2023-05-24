import datetime
import pytest
import numpy as np

from ...models.transition.linear import ConstantVelocity
from ...predictor.information import InformationKalmanPredictor
from ...predictor.kalman import KalmanPredictor
from ...types.state import InformationState, GaussianState
from ...types.array import StateVector, CovarianceMatrix


@pytest.mark.parametrize(
    "PredictorClass, transition_model, prior_mean, prior_covar",
    [
        (   # Standard Kalman
            InformationKalmanPredictor,
            ConstantVelocity(noise_diff_coeff=0.1),
            StateVector([-6.45, 0.7]),
            CovarianceMatrix([[4.1123, 0.0013],
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

    # First do prediction in standard way
    test_state = GaussianState(prior_mean, prior_covar, timestamp=timestamp)
    test_predictor = KalmanPredictor(transition_model)
    test_prediction = test_predictor.predict(test_state, timestamp=new_timestamp)

    # define the precision matrix and information state
    precision_matrix = np.linalg.inv(prior_covar)
    info_state_mean = precision_matrix @ prior_mean

    # Define prior information state
    prior = InformationState(info_state_mean, precision_matrix, timestamp=timestamp)

    # Initialise a Information filter predictor
    predictor = PredictorClass(transition_model=transition_model)

    # Perform and assert state prediction
    prediction = predictor.predict(prior=prior,
                                   timestamp=new_timestamp)

    # reconstruct the state vector and covariance matrix
    pred_covar = np.linalg.inv(prediction.precision)
    pred_mean = pred_covar @ prediction.state_vector

    # And do the tests
    assert np.allclose(predictor._transition_function(prior,
                                                      time_interval=new_timestamp-timestamp),
                       test_prediction.state_vector, 0, atol=1e-14)
    assert np.allclose(pred_mean,
                       test_prediction.state_vector, 0, atol=1.e-14)
    assert np.allclose(pred_covar,
                       test_prediction.covar, 0, atol=1.e-14)
    assert prediction.timestamp == new_timestamp

    # test that we can get to the inverse matrix
    class ConstantVelocitywithInverse(ConstantVelocity):

        def inverse_matrix(self, **kwargs):
            return np.linalg.inv(self.matrix(**kwargs))

    transition_model_winv = ConstantVelocitywithInverse(noise_diff_coeff=0.1)
    predictor_winv = PredictorClass(transition_model_winv)

    # Test this still works
    prediction_from_inv = predictor_winv.predict(prior=prior, timestamp=new_timestamp)

    assert (np.allclose(prediction.state_vector, prediction_from_inv.state_vector, 0, atol=1.e-14))

    # TODO: Test with Control Model
