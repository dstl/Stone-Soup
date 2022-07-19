# -*- coding: utf-8 -*-
import numpy as np
from stonesoup.plotter import Plotter, Dimension
import pytest
import matplotlib.pyplot as plt

# Setup simulation to test the plotter functionality
from datetime import datetime
from datetime import timedelta

from stonesoup.types.detection import TrueDetection
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.sensor.radar.radar import RadarElevationBearingRange

from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
                                               ConstantVelocity
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState

from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.updater.kalman import KalmanUpdater

from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.measures import Mahalanobis

from stonesoup.dataassociator.neighbour import NearestNeighbour
from stonesoup.types.state import GaussianState

from stonesoup.types.track import Track

start_time = datetime.now()
transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.005),
                                                          ConstantVelocity(0.005)])
truth = GroundTruthPath([GroundTruthState([0, 1, 0, 1], timestamp=start_time)])
for k in range(1, 21):
    truth.append(GroundTruthState(
        transition_model.function(truth[k-1], noise=True, time_interval=timedelta(seconds=1)),
        timestamp=start_time+timedelta(seconds=k)))

prob_det = 0.5

measurement_model = LinearGaussian(
    ndim_state=4,
    mapping=(0, 2),
    noise_covar=np.array([[0.75, 0],
                          [0, 0.75]]))
all_measurements = []
for state in truth:
    measurement_set = set()
    # Generate actual detection from the state with a 1-p_d chance that no detection is received.
    if np.random.rand() <= prob_det:
        measurement = measurement_model.function(state, noise=True)
        measurement_set.add(TrueDetection(state_vector=measurement,
                                          groundtruth_path=truth,
                                          timestamp=state.timestamp,
                                          measurement_model=measurement_model))

    all_measurements.append(measurement_set)

predictor = KalmanPredictor(transition_model)
updater = KalmanUpdater(measurement_model)
hypothesiser = DistanceHypothesiser(predictor, updater, measure=Mahalanobis(), missed_distance=3)
data_associator = NearestNeighbour(hypothesiser)

# Run Kalman filter with data association
# Create prior
prior = GaussianState([[0], [1], [0], [1]], np.diag([1.5, 0.5, 1.5, 0.5]), timestamp=start_time)
track = Track([prior])
for n, measurements in enumerate(all_measurements):
    hypotheses = data_associator.associate([track],
                                           measurements,
                                           start_time + timedelta(seconds=n))
    hypothesis = hypotheses[track]  # get the hypothesis for the specified track

    if hypothesis.measurement:
        post = updater.update(hypothesis)
        track.append(post)
    else:  # When data associator says no detections are good enough, we'll keep the prediction
        track.append(hypothesis.prediction)

plotter = Plotter()
# Test functions


def test_dimension_raise():
    with pytest.raises(TypeError):
        Plotter(dimension=1)  # expected to raise TypeError


def test_dimension_inlist():  # ensure dimension type is in predefined enum list
    with pytest.raises(AttributeError):
        Plotter(dimension=Dimension.TESTERROR)


def test_measurements_legend():
    plotter.plot_measurements(all_measurements, [0, 2])  # Measurements entry in legend dict
    plt.close()
    assert 'Measurements' in plotter.legend_dict


def test_measurement_clutter():  # no clutter should be plotted
    plotter.plot_measurements(all_measurements, [0, 2])
    plt.close()
    assert 'Clutter' not in plotter.legend_dict


def test_particle_3d():  # warning should arise if particle is attempted in 3d mode
    plotter3 = Plotter(dimension=Dimension.THREE)

    with pytest.raises(NotImplementedError):
        plotter3.plot_tracks(track, [0, 1, 2], particle=True, uncertainty=False)


def test_plot_sensors():
    plotter3d = Plotter(Dimension.THREE)
    sensor = RadarElevationBearingRange(
        position_mapping=(0, 2, 4),
        noise_covar=np.array([[0, 0, 0],
                              [0, 0, 0]]),
        ndim_state=6,
        position=np.array([[10], [50], [0]])
    )
    plotter3d.plot_sensors(sensor, marker='o', color='red')
    plt.close()
    assert 'Sensor' in plotter3d.legend_dict


def test_empty_tracks():
    plotter.plot_tracks(set(), [0, 2])
    plt.close()
