import pytest
import numpy as np

from ...hypothesiser.probability import PDAHypothesiser
from ...hypothesiser.distance import DistanceHypothesiser
from ...measures import Mahalanobis
from ...models.measurement.linear import LinearGaussian
from ...types.array import CovarianceMatrix
from ...models.transition.linear import ConstantVelocity, CombinedLinearGaussianTransitionModel
from ...predictor.kalman import KalmanPredictor
from ...updater.kalman import ExtendedKalmanUpdater


@pytest.fixture()
def measurement_model():
    return LinearGaussian(ndim_state=4, mapping=[0, 2],
                          noise_covar=CovarianceMatrix(np.diag([1, 1])))


@pytest.fixture()
def predictor():
    transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(1),
                                                              ConstantVelocity(1)])
    return KalmanPredictor(transition_model)


@pytest.fixture()
def updater(measurement_model):
    return ExtendedKalmanUpdater(measurement_model)


@pytest.fixture()
def probability_hypothesiser(predictor, updater):
    return PDAHypothesiser(predictor, updater,
                           clutter_spatial_density=1.2e-2,
                           prob_detect=0.9, prob_gate=0.99,
                           include_all=True)


@pytest.fixture()
def distance_hypothesiser(predictor, updater):
    return DistanceHypothesiser(predictor, updater, Mahalanobis(), 10)
