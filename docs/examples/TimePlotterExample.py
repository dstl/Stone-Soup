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
timestep = datetime.timedelta(seconds=5)

# Simulators
groundtruth_sim = MultiTargetGroundTruthSimulator(
    transition_model=transition_model,
    initial_state=GaussianState(
        StateVector([[0], [0], [0], [0]]),
        CovarianceMatrix(np.diag([1000, 10, 1000, 10])),
        timestamp=start_time),
    timestep=timestep,
    number_steps=60,
    birth_rate=0.15,
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
deleter = CovarianceBasedDeleter(covar_trace_thresh=1E3)
initiator = MultiMeasurementInitiator(
    prior_state=GaussianState(np.array([[0], [0], [0], [0]]),
                              np.diag([0, 100, 0, 1000]),
                              timestamp=start_time),
    measurement_model=measurement_model,
    deleter=deleter,
    data_associator=data_associator,
    updater=updater,
    min_points=3,
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
# Simulation Results
# --------------------
from stonesoup.types.detection import Clutter, TrueDetection

average_life_of_gt = timestep * sum(len(gt) for gt in groundtruth)/len(groundtruth)
n_clutter = sum(isinstance(det, Clutter) for det in detections)
n_true_detections = sum(isinstance(det, TrueDetection) for det in detections)
average_life_of_track = timestep * sum(len(track) for track in all_tracks)/len(all_tracks)

print("The simulation produced:\n",
      len(groundtruth), "Ground truth paths with an average life of", average_life_of_gt, "\n",
      n_clutter, "Clutter Detections\n",
      n_true_detections, "True Detections\n",
      len(all_tracks), "Tracks with an average life of", average_life_of_track, "\n", )


# %%
# Create the Animation
# --------------------
from stonesoup.plotter import AnimationPlotter

# %%
# Create the plotter object and use the plot_# functions to assign the data to plot
plotter = AnimationPlotter(legend_kwargs=dict(loc='upper left'))
plotter.plot_ground_truths(groundtruth, mapping=[0, 2])
plotter.plot_measurements(detections, mapping=[0, 2])
plotter.plot_tracks(all_tracks, mapping=[0, 2])

# %%
# Run the Animation
# -----------------
# The following is needed to ensure that the animation is made playable via interactive
# player in Jupyter Notebooks.
import matplotlib
matplotlib.rcParams['animation.html'] = 'jshtml'

# %%
# To avoid the figure becoming too cluttered with old information, states older than 60 seconds will
# not be shown.
plotter.run(plot_item_expiry=datetime.timedelta(seconds=60))

# %%
# Save the Animation
# ------------------
# Save the animation to a gif format. Other formats are available.
plotter.save('example_animation.gif')

