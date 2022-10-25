#!/usr/bin/env python
# coding: utf-8

"""
TimeBasedPlotter Example
========================
This example shows how to animate several state sequences to be plotted in time order.
"""

# %%
# Building a Simple Simulation and Tracker
# ----------------------------------------
# For simplicity, we are going to quickly build a basic Kalman Tracker, with simple Stone Soup
# simulators, including clutter. In this case a 2D constant velocity target, with 2D linear
# measurements of position.

# %%
# All non-generic imports will be given in order of usage.
import datetime

import numpy as np

from stonesoup.dataassociator.neighbour import GNNWith2DAssignment
from stonesoup.deleter.error import CovarianceBasedDeleter
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.initiator.simple import MultiMeasurementInitiator
from stonesoup.measures import Mahalanobis
from stonesoup.models.transition.linear import (
    CombinedLinearGaussianTransitionModel, ConstantVelocity)
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.simulator.simple import MultiTargetGroundTruthSimulator, SimpleDetectionSimulator
from stonesoup.tracker.simple import MultiTargetTracker
from stonesoup.types.array import StateVector, CovarianceMatrix
from stonesoup.types.state import GaussianState
from stonesoup.updater.kalman import KalmanUpdater

# %%
# Set up the platform and detection simulators

# Models
transition_model = CombinedLinearGaussianTransitionModel(
    [ConstantVelocity(1), ConstantVelocity(1)])
measurement_model = LinearGaussian(4, [0, 2], np.diag([0.5, 0.5]))

start_time = datetime.datetime.now()


# Simulators
groundtruth_sim = MultiTargetGroundTruthSimulator(
    transition_model=transition_model,
    initial_state=GaussianState(
        StateVector([[0], [0], [0], [0]]),
        CovarianceMatrix(np.diag([1000, 10, 1000, 10])),
        timestamp=start_time),
    timestep=datetime.timedelta(seconds=5),
    number_steps=60,
    birth_rate=0.2,
    death_probability=0.05
)
detection_sim = SimpleDetectionSimulator(
    groundtruth=groundtruth_sim,
    measurement_model=measurement_model,
    meas_range=np.array([[-1, 1], [-1, 1]]) * 5000,  # Area to generate clutter
    detection_probability=0.9,
    clutter_rate=1,
)

# %%
# Set up the tracker

# Filter
predictor = KalmanPredictor(transition_model)
updater = KalmanUpdater(measurement_model)

# Data Associator
hypothesiser = DistanceHypothesiser(predictor, updater, Mahalanobis(), missed_distance=3)
data_associator = GNNWith2DAssignment(hypothesiser)

# Initiator & Deleter
deleter = CovarianceBasedDeleter(covar_trace_thresh=1E2)
initiator = MultiMeasurementInitiator(
    GaussianState(np.array([[0], [0], [0], [0]]),
                  np.diag([0, 100, 0, 1000]),
                  timestamp=start_time),
    measurement_model=measurement_model,
    deleter=deleter,
    data_associator=data_associator,
    updater=updater,
    min_points=2,
)

# Tracker
tracker = MultiTargetTracker(
    initiator=initiator,
    deleter=deleter,
    detector=detection_sim,
    data_associator=data_associator,
    updater=updater,
)

# %%
# Run the simulation

groundtruth = set()
detections = set()
all_tracks = set()

for time, tracks in tracker:
    groundtruth.update(groundtruth_sim.groundtruth_paths)
    detections.update(detection_sim.detections)
    all_tracks.update(tracks)


# %%
# Create the Animation
# --------------------
from stonesoup.plotter import AnimationPlotter
from matplotlib import pyplot as plt


# %%
# Create the plotter object and use the plot_# functions to assign the data to plot
plotter = AnimationPlotter()
plotter.plot_ground_truths(groundtruth, mapping=[0, 2])
plotter.plot_measurements(detections, mapping=[0, 2])
plotter.plot_tracks(all_tracks, mapping=[0, 2])

# %%
# Run the Animation
# -----------------
# The times to refresh the animation are chosen to co-inside with the simulator times
#
# To avoid the figure becoming too cluttered with old information, states older than 60 seconds will
# not be shown.
times_to_plot = [start_time + x * groundtruth_sim.timestep
                 for x in range(groundtruth_sim.number_steps)]

ani = plotter.run(times_to_plot, plot_item_expiry=datetime.timedelta(seconds=60))
plt.show()

# %%
# Save the Animation
# ------------------
# Save the animation to a gif format. Other formats are available
plotter.save('example_animation.gif')
