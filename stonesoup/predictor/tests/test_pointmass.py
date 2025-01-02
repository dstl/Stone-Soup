"""Test for updater.particle module"""

import datetime
from datetime import timedelta

import numpy as np
from numpy.linalg import inv

from stonesoup.functions import grid_creation
from stonesoup.models.transition.linear import KnownTurnRate
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.predictor.pointmass import PointMassPredictor
from stonesoup.types.array import StateVectors
from stonesoup.types.state import GaussianState, PointMassState


def test_pointmass():
    start_time = datetime.datetime.now().replace(microsecond=0)
    transition_model = KnownTurnRate(
        turn_noise_diff_coeffs=[2, 2], turn_rate=np.deg2rad(30)
    )
    time_difference = timedelta(days=0, hours=0, minutes=0, seconds=1)

    # Initial condition - Gaussian
    nx = 4
    meanX0 = np.array([36569, 50, 55581, 50])  # mean value
    varX0 = np.diag([90, 5, 160, 5.1])  # variance
    Npa = np.array(
        [33, 33, 33, 33]
    )  # 33 number of points per axis, for FFT must be ODD!!!!
    sFactor = 4  # scaling factor (number of sigmas covered by the grid)

    [predGrid, predGridDelta, gridDimOld, xOld, Ppold] = grid_creation(
        np.vstack(meanX0), varX0, sFactor, nx, Npa
    )

    predictorKF = KalmanPredictor(transition_model)
    priorKF = GaussianState(meanX0, varX0, timestamp=start_time)
    prediction = predictorKF.predict(priorKF, timestamp=start_time + time_difference)

    meanX0 = np.vstack(meanX0)
    pom = predGrid - meanX0
    denominator = np.sqrt((2 * np.pi) ** nx) * np.linalg.det(varX0)
    pompom = np.sum(
        -0.5 * np.multiply(pom.T @ inv(varX0), pom.T), 1
    )  # elementwise multiplication
    pomexp = np.exp(pompom)
    predDensityProb = pomexp / denominator  # Adding probabilities to points
    predDensityProb = predDensityProb / (sum(predDensityProb) * np.prod(predGridDelta))

    priorPMF = PointMassState(
        state_vector=StateVectors(predGrid),
        weight=predDensityProb,
        grid_delta=predGridDelta,
        grid_dim=gridDimOld,
        center=xOld,
        eigVec=Ppold,
        Npa=Npa,
        timestamp=start_time,
    )
    pmfPredictor = PointMassPredictor(transition_model)
    predictionPMF = pmfPredictor.predict(
        priorPMF, timestamp=start_time + time_difference
    )
    predictionPMFnoTime = pmfPredictor.predict(
        priorPMF, timestamp=start_time
    )
    assert np.allclose(predictionPMF.mean, np.ravel(prediction.mean), atol=1)
    assert np.allclose(predictionPMF.covar(), prediction.covar, atol=2)
    assert np.all(predictionPMF.Npa == Npa)
    assert np.all(np.argsort(predictionPMF.grid_delta) == np.argsort(np.diag(varX0)))
    assert np.allclose(
        predictionPMF.center, transition_model.matrix(time_difference) @ xOld, atol=1e-1
    )
    assert np.isclose(
        np.sum(predictionPMF.weight * np.prod(predictionPMF.grid_delta)), 1, atol=1e-1
    )
    assert np.all(priorPMF.state_vector == predictionPMFnoTime.state_vector)
    assert np.all(priorPMF.weight == predictionPMFnoTime.weight)
    assert np.all(priorPMF.grid_delta == predictionPMFnoTime.grid_delta)
    assert np.all(priorPMF.grid_dim == predictionPMFnoTime.grid_dim)
    assert np.all(priorPMF.center == predictionPMFnoTime.center)
    assert np.all(priorPMF.eigVec == predictionPMFnoTime.eigVec)
    assert np.all(priorPMF.Npa == predictionPMFnoTime.Npa)
    assert np.all(priorPMF.timestamp == predictionPMFnoTime.timestamp)
