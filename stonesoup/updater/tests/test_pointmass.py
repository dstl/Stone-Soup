"""Test for updater.kalman module"""

import pytest
import numpy as np

from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.types.detection import Detection
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.prediction import (
    GaussianStatePrediction, GaussianMeasurementPrediction, PointMassStatePrediction, PointMassMeasurementPrediction)
from stonesoup.types.state import GaussianState
from stonesoup.updater.pointMass import PointMassUpdater

from stonesoup.models.transition.linear import KnownTurnRate
from datetime import datetime
from datetime import timedelta
from functions import gridCreation
from stonesoup.types.array import StateVectors
import time


@pytest.fixture(params=[PointMassUpdater])


def updater(request):
    updater_class = request.param
    measurement_model = LinearGaussian(
       ndim_state=2, mapping=[0], noise_covar=np.array([[0.04]]))
    return updater_class(measurement_model)

def test_pointmass(updater):
    
    time_difference = timedelta(days=0, hours=0, minutes=0, seconds=1)
    
    
    # Initial condition - Gaussian
    nx = 2
    meanX0 = np.array([36569, 55581]) # mean value
    varX0 = np.diag([90, 160]) # variance
    Npa = np.array([31, 31]) # 33 number of points per axis, for FFT must be ODD!!!!
    N = np.prod(Npa) # number of points - total
    sFactor = 4 # scaling factor (number of sigmas covered by the grid)
    
    
    [predGrid, predGridDelta, gridDimOld, xOld, Ppold] = gridCreation(np.vstack(meanX0),varX0,sFactor,nx,Npa)
    
    meanX0 = np.vstack(meanX0)
    pom = predGrid-np.matlib.repmat(meanX0,1,N)
    denominator = np.sqrt((2*np.pi)**nx)*np.linalg.det(varX0)
    pompom = np.sum(-0.5*np.multiply(pom.T@np.inv(varX0),pom.T),1) #elementwise multiplication
    pomexp = np.exp(pompom)
    predDensityProb = pomexp/denominator # Adding probabilities to points
    predDensityProb = predDensityProb/(sum(predDensityProb)*np.prod(predGridDelta))
    

    start_time = time.time()
    
    prediction = PointMassStatePrediction(state_vector=StateVectors(predGrid),
                          weight=predDensityProb,
                          grid_delta = predGridDelta,
                          grid_dim = gridDimOld,
                          center = xOld,
                          eigVec = Ppold,
                          Npa = Npa,
                          timestamp=start_time),
    prediction = prediction[0]
    measurement = Detection(np.array([[-6.23]]))
    
    measurement_model=updater.measurement_model

    # Calculate evaluation variables
    eval_measurement_prediction = GaussianMeasurementPrediction(
        measurement_model.matrix(time_difference) @ prediction.mean,
        measurement_model.matrix(time_difference) @ prediction.covar
        @ measurement_model.matrix(time_difference).T
        + measurement_model.covar(time_difference),
        cross_covar=prediction.covar @ measurement_model.matrix(time_difference).T)
    kalman_gain = eval_measurement_prediction.cross_covar @ np.linalg.inv(
        eval_measurement_prediction.covar)
    eval_posterior = GaussianState(
        prediction.mean
        + kalman_gain @ (measurement.state_vector
                         - eval_measurement_prediction.mean),
        prediction.covar
        - kalman_gain@eval_measurement_prediction.covar @ kalman_gain.T)


    # Get and assert measurement prediction
    measurement_prediction = updater.predict_measurement(prediction)
    assert np.allclose(measurement_prediction.mean,
                       eval_measurement_prediction.mean,
                       0, atol=1.e-14)
    assert np.allclose(measurement_prediction.covar,
                       eval_measurement_prediction.covar,
                       0, atol=1.e-14)
    assert np.allclose(measurement_prediction.cross_covar,
                       eval_measurement_prediction.cross_covar,
                       0, atol=1.e-13)

    # Perform and assert state update (without measurement prediction)
    posterior = updater.update(SingleHypothesis(
        prediction=prediction,
        measurement=measurement))
    assert np.allclose(posterior.mean, eval_posterior.mean, 0, atol=1.e-14)
    assert np.allclose(posterior.covar, eval_posterior.covar, 0, atol=1.e-13)
    assert np.array_equal(posterior.hypothesis.prediction, prediction)
    assert np.allclose(
        posterior.hypothesis.measurement_prediction.state_vector,
        measurement_prediction.state_vector, 0, atol=1.e-14)
    assert np.allclose(posterior.hypothesis.measurement_prediction.covar,
                       measurement_prediction.covar, 0, atol=1.e-14)
    assert np.array_equal(posterior.hypothesis.measurement, measurement)
    assert posterior.timestamp == prediction.timestamp

    # Perform and assert state update
    posterior = updater.update(SingleHypothesis(
        prediction=prediction,
        measurement=measurement,
        measurement_prediction=measurement_prediction))
    assert np.allclose(posterior.mean, eval_posterior.mean, 0, atol=1.e-14)
    assert np.allclose(posterior.covar, eval_posterior.covar, 0, atol=1.e-13)
    assert np.array_equal(posterior.hypothesis.prediction, prediction)
    assert np.allclose(
        posterior.hypothesis.measurement_prediction.state_vector,
        measurement_prediction.state_vector, 0, atol=1.e-14)
    assert np.allclose(posterior.hypothesis.measurement_prediction.covar,
                       measurement_prediction.covar, 0, atol=1.e-14)
    assert np.array_equal(posterior.hypothesis.measurement, measurement)
    assert posterior.timestamp == prediction.timestamp



if __name__ == "__main__":
    import pytest
    pytest.main(['-v', __file__])