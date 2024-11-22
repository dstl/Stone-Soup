"""Test for updater.particle module"""

import datetime

import numpy as np
from numpy.linalg import inv

from stonesoup.functions import grid_creation
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.types.array import StateVectors
from stonesoup.types.detection import Detection
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.state import PointMassState
from stonesoup.updater.pointmass import PointMassUpdater


def test_pointmass():
    start_time = datetime.datetime.now().replace(microsecond=0)
    truth = GroundTruthPath(
        [GroundTruthState([36569, 50, 55581, 50], timestamp=start_time)]
    )
    matrix = np.array(
        [
            [1, 0],
            [0, 1],
        ]
    )
    measurement_model = LinearGaussian(ndim_state=4, mapping=(0, 2), noise_covar=matrix)
    measurements = []
    measurement = measurement_model.function(truth, noise=True)
    measurements.append(
        Detection(
            measurement, timestamp=truth.timestamp, measurement_model=measurement_model
        )
    )

    # Initial condition - Gaussian
    nx = 4
    meanX0 = np.array([36569, 50, 55581, 50])  # mean value
    varX0 = np.diag([90, 5, 160, 5])  # variance
    Npa = np.array(
        [31, 31, 27, 27]
    )  # 33 number of points per axis, for FFT must be ODD!!!!
    sFactor = 4  # scaling factor (number of sigmas covered by the grid)

    [predGrid, predGridDelta, gridDimOld, xOld, Ppold] = grid_creation(
        np.vstack(meanX0), varX0, sFactor, nx, Npa
    )

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
    pmfUpdater = PointMassUpdater(measurement_model)
    for measurement in measurements:
        hypothesis = SingleHypothesis(priorPMF, measurement)
        post = pmfUpdater.update(hypothesis)
    assert np.all(post.state_vector == StateVectors(predGrid))
    assert np.all(post.grid_delta == predGridDelta)
    assert np.all(post.grid_dim == gridDimOld)
    assert np.all(post.center == xOld)
    assert np.all(post.eigVec == Ppold)
    assert np.all(post.Npa == Npa)
    assert post.timestamp == start_time
    assert np.isclose(np.sum(post.weight * np.prod(post.grid_delta)), 1, 1e-2)
