"""Test for updater.information module"""
import pytest
import numpy as np

from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.types.detection import Detection
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.prediction import GaussianStatePrediction, InformationStatePrediction
from stonesoup.updater.kalman import KalmanUpdater
from stonesoup.updater.information import InformationKalmanUpdater


@pytest.mark.parametrize(
    "UpdaterClass, measurement_model, prediction, measurement",
    [
        (   # Standard Information filter
            InformationKalmanUpdater,
            LinearGaussian(ndim_state=2, mapping=[0],
                           noise_covar=np.array([[0.04]])),
            GaussianStatePrediction(np.array([[-6.45], [0.7]]),
                                    np.array([[4.1123, 0.0013],
                                              [0.0013, 0.0365]])),
            Detection(np.array([[-6.23]]))
        ),
    ],
    ids=["standard"]
)
def test_information(UpdaterClass, measurement_model, prediction, measurement):
    """Tests the information form of the Kalman filter update step."""

    # This is how the Kalman filter does it
    kupdater = KalmanUpdater(measurement_model)
    kposterior = kupdater.update(SingleHypothesis(prediction, measurement))

    # Create the information state representation
    prediction_precision = np.linalg.inv(prediction.covar)
    info_prediction_mean = prediction_precision @ prediction.state_vector

    info_prediction = InformationStatePrediction(info_prediction_mean, prediction_precision)

    # Initialise a information form of the Kalman updater
    updater = UpdaterClass(measurement_model=measurement_model)

    # Perform and assert state update (without measurement prediction)
    posterior = updater.update(SingleHypothesis(
        prediction=info_prediction,
        measurement=measurement))

    # Check to see if the information matrix is positive definite (i.e. are all the eigenvalues
    # positive?)
    assert(np.all(np.linalg.eigvals(posterior.precision) >= 0))

    # Does the measurement prediction work?
    assert(np.allclose(kupdater.predict_measurement(prediction).state_vector,
                       updater.predict_measurement(info_prediction).state_vector, 0, atol=1.e-14))

    # Do the
    assert(np.allclose(kposterior.state_vector,
                       np.linalg.inv(posterior.precision) @ posterior.state_vector, 0,
                       atol=1.e-14))
    assert(np.allclose(kposterior.covar, np.linalg.inv(posterior.precision), 0, atol=1.e-14))
    assert(np.array_equal(posterior.hypothesis.prediction, info_prediction))

    assert(np.array_equal(posterior.hypothesis.measurement, measurement))
    assert(posterior.timestamp == prediction.timestamp)

    # test that we can get to the inverse matrix
    class LinearGaussianwithInverse(LinearGaussian):

        def inverse_covar(self, **kwargs):
            return np.linalg.inv(self.covar(**kwargs))

    meas_model_winv = LinearGaussianwithInverse(ndim_state=2, mapping=[0],
                                                noise_covar=np.array([[0.04]]))
    updater_winv = UpdaterClass(meas_model_winv)

    # Test this still works
    post_from_inv = updater_winv.update(SingleHypothesis(prediction=info_prediction,
                                                         measurement=measurement))
    # and check
    assert(np.allclose(posterior.state_vector, post_from_inv.state_vector, 0, atol=1.e-14))

    # Can one force symmetric covariance?
    updater.force_symmetric_covariance = True
    posterior = updater.update(SingleHypothesis(
        prediction=info_prediction,
        measurement=measurement))

    assert(np.allclose(posterior.precision - posterior.precision.T,
                       np.zeros(np.shape(posterior.precision)), 0, atol=1.e-14))
