#!/usr/bin/env python
# coding: utf-8

"""
10 - Simulation & Tracking Components
=====================================
Running through the tutorials, the simulation code and tracking code has developed as new elements
have been added. In the last tutorial, this ended with a simulation of varying number of targets,
and a multi-target tracking handling initialisation and deletion of tracks. This tutorial just
wraps up these elements by showing the Stone Soup components that implement the similar simulation
and tracking features in existing components.
"""

# %%
# Creating Simulation
# -------------------
# First create models as seen in previous tutorials
import numpy as np

from stonesoup.models.transition.linear import (
    CombinedLinearGaussianTransitionModel, ConstantVelocity)
from stonesoup.models.measurement.linear import LinearGaussian

transition_model = CombinedLinearGaussianTransitionModel(
    [ConstantVelocity(0.05), ConstantVelocity(0.05)])
measurement_model = LinearGaussian(4, [0, 2], np.diag([0.25, 0.25]))

# %%
# And with those models, create a multi target ground truth simulator. This has a number of
# configurable parameters, e.g. defining where tracks are born and at what rate, death probability.
# This implements similar logic to the code in previous tutorial section
# :ref:`auto_tutorials/09_Initiators_&_Deleters:Simulating Multiple Targets`
import datetime

from stonesoup.simulator.simple import MultiTargetGroundTruthSimulator, SimpleDetectionSimulator
from stonesoup.types.array import StateVector, CovarianceMatrix
from stonesoup.types.state import GaussianState

groundtruth_sim = MultiTargetGroundTruthSimulator(
    transition_model=transition_model,
    initial_state=GaussianState(
        StateVector([[0], [0], [0], [0]]),
        CovarianceMatrix(np.diag([4, 0.5, 4, 0.5]))),
    timestep=datetime.timedelta(seconds=5),
    number_steps=20,
    birth_rate=0.3,
    death_probability=0.05
)

# %%
# This simulated ground truth will then be passed to a simple detection simulator. This again has a
# number of configurable parameters, e.g. where clutter is generated and at what rate, and
# detection probability. This implements similar logic to the code in the previous tutorial section
# :ref:`auto_tutorials/09_Initiators_&_Deleters:Generate Detections and Clutter`
detection_sim = SimpleDetectionSimulator(
    groundtruth=groundtruth_sim,
    measurement_model=measurement_model,
    detection_probability=0.9,
    meas_range=np.array([[-1, 1], [-1, 1]])*30,  # Area to generate clutter
    clutter_rate=1,
)

# %%
# Creating the Tracker Components
# -------------------------------
# Similar tracking components will be created as in the previous tutorials, using same models used
# in simulation above. In this example, Kalman filter is used with Global Nearest Neighbour.
from stonesoup.predictor.kalman import KalmanPredictor
predictor = KalmanPredictor(transition_model)

from stonesoup.updater.kalman import KalmanUpdater
updater = KalmanUpdater(measurement_model)

from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.measures import Mahalanobis
hypothesiser = DistanceHypothesiser(predictor, updater, measure=Mahalanobis(), missed_distance=3)

from stonesoup.dataassociator.neighbour import GNNWith2DAssignment
data_associator = GNNWith2DAssignment(hypothesiser)

from stonesoup.deleter.error import CovarianceBasedDeleter
deleter = CovarianceBasedDeleter(2)

from stonesoup.initiator.simple import MultiMeasurementInitiator
initiator = MultiMeasurementInitiator(
    prior_state=GaussianState([[0], [0], [0], [0]], np.diag([0, 0.5, 0, 0.5])),
    measurement_model=measurement_model,
    deleter=deleter,
    data_associator=data_associator,
    updater=updater,
    min_points=3,
)

# %%
# Creating and Running the Tracker
# --------------------------------
# With the components created, the multi-target tracker component will be created, constructed from
# the components created above. This is logically the same as tracking code in the previous
# tutorial section :ref:`auto_tutorials/09_Initiators_&_Deleters:Running the Tracker`
from stonesoup.tracker.simple import MultiTargetTracker

tracker = MultiTargetTracker(
    initiator=initiator,
    deleter=deleter,
    detector=detection_sim,
    data_associator=data_associator,
    updater=updater,
)

# %%
# And finally plotting the output using a Stone Soup plotting metric generator. This will produce a
# plot very similar to that seen across all these tutorials.
groundtruth = set()
detections = set()
tracks = set()
for time, ctracks in tracker:
    groundtruth.update(groundtruth_sim.groundtruth_paths)
    detections.update(detection_sim.detections)
    tracks.update(ctracks)

from stonesoup.metricgenerator.plotter import TwoDPlotter
plotter = TwoDPlotter(track_indices=[0, 2], gtruth_indices=[0, 2], detection_indices=[0, 1])
fig = plotter.plot_tracks_truth_detections(tracks, groundtruth, detections).value

ax = fig.axes[0]
ax.set_xlim([-30, 30])
_ = ax.set_ylim([-30, 30])
