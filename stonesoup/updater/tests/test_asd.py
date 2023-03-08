import datetime

import numpy as np

from ..asd import ASDKalmanUpdater
from ...models.measurement.linear import LinearGaussian
from ...types.detection import Detection
from ...types.hypothesis import SingleHypothesis
from ...types.prediction import ASDGaussianStatePrediction, ASDGaussianMeasurementPrediction
from ...types.state import GaussianState


def test_asdkalman():
    timestamp = datetime.datetime.now()
    measurement_model = LinearGaussian(
        ndim_state=2, mapping=[0], noise_covar=np.array([[0.04]]))
    prediction = ASDGaussianStatePrediction(
        np.array([[-6.45], [0.7]]),
        multi_covar=np.array([[4.1123, 0.0013], [0.0013, 0.0365]]),
        timestamps=[timestamp],
        correlation_matrices=[{'P': np.eye(2)}],
        act_timestamp=timestamp)
    measurement = Detection(np.array([[-6.23]]), timestamp=timestamp)

    # Calculate evaluation variables
    Pi = np.block([[np.eye(prediction.ndim)],
                   [np.zeros((prediction.multi_covar.shape[0] -
                              prediction.ndim, prediction.ndim))]])
    cross_cov = prediction.multi_covar @ Pi @ measurement_model.matrix().T
    eval_measurement_prediction = ASDGaussianMeasurementPrediction(
        measurement_model.matrix() @ prediction.mean,
        multi_covar=measurement_model.matrix()@prediction.covar
        @ measurement_model.matrix().T
        + measurement_model.covar(),
        cross_covar=cross_cov, timestamps=prediction.timestamps)

    kalman_gain = eval_measurement_prediction.cross_covar @ np.linalg.inv(
        eval_measurement_prediction.covar)
    eval_posterior = GaussianState(
        prediction.mean
        + kalman_gain@(measurement.state_vector
                       - eval_measurement_prediction.mean),
        prediction.covar
        - kalman_gain@eval_measurement_prediction.covar@kalman_gain.T)

    # Initialise a kalman updater
    updater = ASDKalmanUpdater(measurement_model=measurement_model)

    # Get and assert measurement prediction
    measurement_prediction = updater.predict_measurement(prediction)
    assert(np.allclose(measurement_prediction.mean,
                       eval_measurement_prediction.mean,
                       0, atol=1.e-14))
    assert(np.allclose(measurement_prediction.covar,
                       eval_measurement_prediction.covar,
                       0, atol=1.e-14))
    assert(np.allclose(measurement_prediction.cross_covar,
                       eval_measurement_prediction.cross_covar,
                       0, atol=1.e-14))

    # Perform and assert state update

    posterior = updater.update(SingleHypothesis(
        prediction=prediction,
        measurement=measurement,
        measurement_prediction=measurement_prediction))
    assert(np.allclose(posterior.mean, eval_posterior.mean, 0, atol=1.e-14))
    assert(np.allclose(posterior.covar, eval_posterior.covar, 0, atol=1.e-14))
    assert(np.array_equal(posterior.hypothesis.prediction, prediction))
    assert (np.allclose(
        posterior.hypothesis.measurement_prediction.state_vector,
        measurement_prediction.state_vector, 0, atol=1.e-14))
    assert (np.allclose(posterior.hypothesis.measurement_prediction.covar,
                        measurement_prediction.covar, 0, atol=1.e-14))
    assert(np.array_equal(posterior.hypothesis.measurement, measurement))
    assert(posterior.timestamp == prediction.timestamp)
