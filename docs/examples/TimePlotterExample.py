#!/usr/bin/env python
# coding: utf-8

"""
TimeBasedPlotter Example
===============
This example shows how to animate several state sequences to be plotted in time order
"""

# %%
# Building a Simple Simulation and Tracker
# ----------------------------------------
# For simplicity, we are going to quickly build a basic Kalman Tracker, with simple Stone Soup
# simulators, including clutter. In this case a 2D constant velocity target, with 2D linear
# measurements of position.
import datetime
import random

import numpy as np
import matplotlib.colors as colors

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
from stonesoup.plotter import TimeBasedPlotter
from stonesoup.types.detection import Clutter, TrueDetection

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
    number_steps=50,
    birth_rate=0.05,
    death_probability=0.05
)
detection_sim = SimpleDetectionSimulator(
    groundtruth=groundtruth_sim,
    measurement_model=measurement_model,
    meas_range=np.array([[-1, 1], [-1, 1]]) * 5000,  # Area to generate clutter
    detection_probability=0.9,
    clutter_rate=1,
)

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
    min_points=5,
)

# Tracker
tracker = MultiTargetTracker(
    initiator=initiator,
    deleter=deleter,
    detector=detection_sim,
    data_associator=data_associator,
    updater=updater,
)

groundtruth = set()
detections = set()
tracks = set()

for time, ctracks in tracker:
    groundtruth.update(groundtruth_sim.groundtruth_paths)
    detections.update(detection_sim.detections)
    tracks.update(ctracks)


all_plotting = []
colours = colors.cnames

times_to_plot = [start_time + x * groundtruth_sim.timestep
                 for x in range(groundtruth_sim.number_steps)]

from stonesoup.plotter import TBPlotter

plotter = TBPlotter()
plotter.plot_ground_truths(groundtruth, mapping=[0, 2])
plotter.plot_measurements(detections, mapping=[0, 2])
plotter.plot_tracks(tracks, mapping=[0, 2])

plotter.run(times_to_plot)
plotter.save('example2.mp4')

from matplotlib import pyplot as plt

plt.show()

"""
for idx, ground_truth in enumerate(groundtruth):
    all_plotting.append(TimeBasedPlotter(plotting_data=ground_truth.states,
                                         legend_key='Ground Truth'+str(idx),
                                         linestyle='--', marker='o', alpha=0.5,
                                         color=random.choice(list(colours.values()))
                                         ))

for idx, track in enumerate(tracks):
    all_plotting.append(TimeBasedPlotter(plotting_data=track.states, legend_key='Track'+str(idx),
                                         linestyle='-', marker='x', alpha=0.5,
                                         color=random.choice(list(colours.values()))
                                         ))

all_plotting.append(TimeBasedPlotter(
    plotting_data=[detection for detection in detections if isinstance(detection, Clutter)],
    legend_key='Clutter', linestyle='', marker='.', alpha=0.5, color='k'))

all_plotting.append(TimeBasedPlotter(
    plotting_data=[detection for detection in detections if isinstance(detection, TrueDetection)],
    legend_key='True Detections', linestyle='', marker='.', alpha=0.5, color='r'))



line_ani = TimeBasedPlotter.run_animation(times_to_plot, all_plotting,
                                          plot_item_expiry=datetime.timedelta(seconds=60))

line_ani.save('example.mp4')
"""
