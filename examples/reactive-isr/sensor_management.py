#!/usr/bin/env python
# coding: utf-8


import numpy as np
import random
from datetime import datetime, timedelta
import time

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from ordered_set import OrderedSet

from stonesoup.custom.functions import get_camera_footprint
from stonesoup.plotter import Plotter
from stonesoup.types.state import StateVector
from stonesoup.custom.sensor.pan_tilt import PanTiltUAVCamera
from stonesoup.types.angle import Angle
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
    ConstantVelocity
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.updater.kalman import ExtendedKalmanUpdater
from stonesoup.types.state import GaussianState
from stonesoup.types.track import Track
from stonesoup.sensormanager.reward import UncertaintyRewardFunction
from stonesoup.sensormanager import BruteForceSensorManager, OptimizeBruteSensorManager
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.measures import Mahalanobis
from stonesoup.dataassociator.neighbour import GNNWith2DAssignment
from stonesoup.metricgenerator.tracktotruthmetrics import SIAPMetrics
from stonesoup.measures import Euclidean
from stonesoup.dataassociator.tracktotrack import TrackToTruth
from stonesoup.metricgenerator.uncertaintymetric import SumofCovarianceNormsMetric
from stonesoup.metricgenerator.manager import SimpleManager

np.random.seed(1990)
random.seed(1990)

# Parameters
# ==========        # Simulation start time
num_iter = 100  # Number of simulation steps
ntruths = 2  # Number of ground truths
total_no_sensors = 1
start_time = datetime.now()

# Models
# ======l
transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.005),
                                                          ConstantVelocity(0.005),
                                                          ConstantVelocity(0)])

# Simulate Groundtruth
# ====================
gnd_transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.),
                                                              ConstantVelocity(0.),
                                                              ConstantVelocity(0.)])
truths = set()
truth = GroundTruthPath([GroundTruthState([0, 0.2, 0, 0.2, 0, 0], timestamp=start_time)])
for k in range(1, num_iter + 1):
    truth.append(GroundTruthState(
        gnd_transition_model.function(truth[k - 1], noise=True,
                                      time_interval=timedelta(seconds=1)),
        timestamp=start_time + timedelta(seconds=k)))
truths.add(truth)

truth = GroundTruthPath([GroundTruthState([0, 0.2, 20, -0.2, 0, 0], timestamp=start_time)])
for k in range(1, num_iter + 1):
    truth.append(GroundTruthState(
        gnd_transition_model.function(truth[k - 1], noise=True,
                                      time_interval=timedelta(seconds=1)),
        timestamp=start_time + timedelta(seconds=k)))
truths.add(truth)

# Create sensors
# ==============
sensors = set()
for n in range(0, total_no_sensors):
    rotation_offset = StateVector(
        [Angle(0), Angle(-np.pi / 2), Angle(0)])  # Camera rotation offset
    pan_tilt = StateVector([Angle(0), Angle(-np.pi / 32)])  # Camera pan and tilt

    sensor = PanTiltUAVCamera(ndim_state=6, mapping=[0, 2, 4],
                              noise_covar=np.diag([0.001, 0.001, 0.001]),
                              fov_angle=[np.radians(15), np.radians(10)],
                              rotation_offset=rotation_offset,
                              pan=pan_tilt[0], tilt=pan_tilt[1],
                              position=StateVector([10., 10., 100.]))
    sensors.add(sensor)
for sensor in sensors:
    sensor.timestamp = start_time

# Predctor and Updater
# ====================
predictor = KalmanPredictor(transition_model)
updater = ExtendedKalmanUpdater(measurement_model=None)
# measurement model is added to detections by the sensor

# Initialise tracks
# =================
tracks = []
for truth in truths:
    sv = truth[0].state_vector
    prior = GaussianState(sv, np.diag([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]), timestamp=start_time)
    tracks.append(Track([prior]))

# Initialise sensor manager
# =========================
reward_function = UncertaintyRewardFunction(predictor, updater)
sensor_manager = BruteForceSensorManager(sensors, reward_function)
# sensor_manager = OptimizeBruteSensorManager(sensors,
#                                             reward_function=reward_function,
#                                             n_grid_points=15,
#                                             finish=True)

# Hypothesiser and Data Associator
# ================================
hypothesiser = DistanceHypothesiser(predictor, updater, measure=Mahalanobis(), missed_distance=5)
data_associator = GNNWith2DAssignment(hypothesiser)

# Run the sensor manager
# ======================

# Start timer for cell execution time
cell_start_time2 = time.time()
timestamps = []
for k in range(1, num_iter + 1):
    timestamps.append(start_time + timedelta(seconds=k))

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(1, 1, 1)
plt.ion()
for k, timestep in enumerate(timestamps):

    print(timestep)
    # Generate chosen configuration
    chosen_actions = sensor_manager.choose_actions(tracks, timestep)

    # Create empty dictionary for measurements
    measurements = []

    for chosen_action in chosen_actions:
        for sensor, actions in chosen_action.items():
            sensor.add_actions(actions)

    ax.cla()
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_xlim(-10, 30)
    ax.set_ylim(-10, 30)
    ax.set_aspect('equal')

    # Fov ranges (min, center, max)
    xmin, xmax, ymin, ymax = get_camera_footprint(sensor)

    ax.add_patch(
        Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, facecolor='none', edgecolor='b'))

    truth_states = OrderedSet(truth[timestep] for truth in truths)
    for sensor in sensors:
        sensor.act(timestep)

        # Observe this ground truth
        sensor_measurements = sensor.measure(truth_states, noise=True)
        measurements.extend(sensor_measurements)

    # Fov ranges (min, center, max)
    xmin, xmax, ymin, ymax = get_camera_footprint(sensor)

    ax.add_patch(
        Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, facecolor='none', edgecolor='r'))
    for truth in truths:
        data = np.array([state.state_vector for state in truth[:k + 2]])
        ax.plot(data[:, 0], data[:, 2], '--', label="Ground truth")

    hypotheses = data_associator.associate(tracks,
                                           measurements,
                                           timestep)
    for track in tracks:
        hypothesis = hypotheses[track]
        if hypothesis.measurement:
            post = updater.update(hypothesis)
            track.append(post)
        else:  # When data associator says no detections are good enough, we'll keep the prediction
            track.append(hypothesis.prediction)

    for track in tracks:
        data = np.array([state.state_vector for state in track])
        ax.plot(data[:, 0], data[:, 2], '-', label="Ground truth")

    plt.pause(0.1)
cell_run_time2 = round(time.time() - cell_start_time2, 2)

# Plot the results
# ================

# Plot ground truths, tracks and uncertainty ellipses for each target.
plotter = Plotter()
plotter.plot_sensors(sensors)
plotter.plot_ground_truths(truths, [0, 2])
plotter.plot_tracks(set(tracks), [0, 2], uncertainty=True)

# Metrics
siap_generator = SIAPMetrics(position_measure=Euclidean((0, 2)),
                             velocity_measure=Euclidean((1, 3)))
associator = TrackToTruth(association_threshold=30)
uncertainty_generator = SumofCovarianceNormsMetric()

metric_manager = SimpleManager([siap_generator, uncertainty_generator],
                               associator=associator)
metric_manager.add_data(truths, tracks)
metricsA = metric_manager.generate_metrics()

# SIAP metrics
fig, axes = plt.subplots(2)
times = metric_manager.list_timestamps()
pa_metricA = metricsA['SIAP Position Accuracy at times']
va_metricA = metricsA['SIAP Velocity Accuracy at times']
axes[0].set(title='Positional Accuracy', xlabel='Time', ylabel='PA')
axes[0].plot(times, [metric.value for metric in pa_metricA.value],
             label='BruteForceSensorManager')
axes[0].legend()
axes[1].set(title='Velocity Accuracy', xlabel='Time', ylabel='VA')
axes[1].plot(times, [metric.value for metric in va_metricA.value],
             label='BruteForceSensorManager')
axes[1].legend()

# Uncertainty metrics
uncertainty_metricA = metricsA['Sum of Covariance Norms Metric']
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot([i.timestamp for i in uncertainty_metricA.value],
        [i.value for i in uncertainty_metricA.value],
        label='BruteForceSensorManager')
ax.set_ylabel("Sum of covariance matrix norms")
ax.set_xlabel("Time")
ax.legend()

# Print run time
print(f'Optimised Brute Force: {cell_run_time2} s')
