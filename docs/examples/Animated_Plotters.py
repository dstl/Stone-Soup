#!/usr/bin/env python
# coding: utf-8

"""
Animated Plotters
=================
This example shows how to animate several 2D state sequences to be plotted in time order. First,
truths, detections, and tracks are created. These are then plotted as animations using two
options that are provided by Stone Soup: Matplotlib-based :class:`~.AnimationPlotter`, and
Plotly-based :class:`~.AnimatedPlotterly`. The two options are then compared with pros and cons
given for both.
"""

# %%
# Creating Ground Truths, Detections, and Tracks
# ----------------------------------------------
# For simplicity, we are going to quickly make a simulation with a basic Kalman Tracker using
# Stone Soup simulators. To see the animations in action, scroll down to "Creating animations".

# %%
# All non-generic imports will be given in order of usage.
from datetime import datetime, timedelta

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
# Set up the platform and detection simulators:

# Models
transition_model = CombinedLinearGaussianTransitionModel(
    [ConstantVelocity(1), ConstantVelocity(1)])
measurement_model = LinearGaussian(4, [0, 2], np.diag([20, 20]))

start_time = datetime.now().replace(microsecond=0)
timestep = timedelta(seconds=5)

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
    meas_range=np.array([[-1, 1], [-1, 1]]) * 2500,  # Area to generate clutter
    detection_probability=0.9,
    clutter_rate=1,
)

# %%
# Set up the tracker:

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
# Generate the Truths, Detections, and Tracks:

groundtruth = set()
detections = set()
all_tracks = set()

for time, tracks in tracker:
    groundtruth.update(groundtruth_sim.groundtruth_paths)
    detections.update(detection_sim.detections)
    all_tracks.update(tracks)


# %%
# Simulation overview:
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
# Creating Animations
# -------------------
# Now we have our data, we can show off the animation functionality in Stone Soup and compare them.

# %%
# AnimationPlotter
# ^^^^^^^^^^^^^^^^
# :class:`~.AnimationPlotter` is built on Matplotlib. Here we show off some of its functionality,
# and save the output. First, we create the plotter object and add an argument for the legend. One
# drawback with this plotter is that the user cannot currently set a custom title.
from stonesoup.plotter import AnimationPlotter
plotter = AnimationPlotter(legend_kwargs=dict(loc='upper left'))

# %%
# Plot the truths, detections, and tracks, and provide the mapping from state space to Cartesian:
plotter.plot_ground_truths(groundtruth, mapping=[0, 2])
plotter.plot_measurements(detections, mapping=[0, 2])
plotter.plot_tracks(all_tracks, mapping=[0, 2])

# %%
# To run the animation, the following is needed to ensure that it is playable via interactive
# player in Jupyter Notebooks:
import matplotlib
matplotlib.rcParams['animation.html'] = 'jshtml'

# %%
# Run the animation. To prevent a cluttered plot, include an argument that deletes
# information older than 60 seconds:
plotter.run(plot_item_expiry=timedelta(seconds=60))

# %%
# Finally, save the animation:
plotter.save('example_animation.gif')

# %%
# AnimatedPlotterly
# ^^^^^^^^^^^^^^^^^
# We now create the Plotly-based :class:`~.AnimatedPlotterly`. For the plotter to initialise, we
# must provide it with a list of equally-spaced simulation timesteps. There are also optional
# arguments that are explained in the docs.

from stonesoup.plotter import AnimatedPlotterly

timesteps = [start_time + timedelta(seconds=5*k) for k in range(60)]
fig = AnimatedPlotterly(timesteps, tail_length=0.2, title="Plotterly Animation")

# %%
# Plot the data and show the animation:
fig.plot_ground_truths(groundtruth, mapping=[0, 2])
fig.plot_measurements(detections, mapping=[0, 2])
fig.plot_tracks(all_tracks, mapping=[0, 2])
fig.show()

# %%
# Comparing Plotters
# ------------------
#
# Looking at the two plotters, it's fairly apparent that :class:`~.AnimatedPlotterly` offers
# more functionality and interaction than :class:`~.AnimationPlotter`. The user can view
# information on each data point by hovering over it, zoom into a specific area, and turn on and
# off specific traces. This is especially useful for this example because the ground truth is
# impossible to view if the tracks are plotted over it. I.e. in :class:`~.AnimationPlotter`,
# to view ground truth, you would need to make a new plotter that doesn't plot ground truth.
#
# However, there are a couple of drawbacks to :class:`~.AnimatedPlotterly`. If you can't extract
# a list of equally-spaced timesteps from your simulation, :class:`~.AnimatedPlotterly` won't work.
# It also more computationally expensive than :class:`~.AnimationPlotter`, so struggles to load
# and render large volumes of data. This can be seen by forcing the animation in Tutorial 4 to
# display every particle - set the plotter's `tail_length` to 1. Then, when plotting tracks, set
# `particle` to True, and `plot_history` to True. In addition, there is no current functionality
# to save the animation without using a screen-capturing tool.
#
# A final comment when comparing these animations is to note that :class:`~.AnimatedPlotterly` has
# a more complex framework than :class:`~.AnimationPlotter`, meaning that adding custom data is
# harder. An example is somewhat shown in the sensor management tutorials, where a sensor's field
# of view is manually added, but a dedicated example may be written in the future.


# %%
# Conclusion
# ^^^^^^^^^^
# In conclusion, AnimatedPlotterly provides a more detailed and interactive user experience than
# AnimationPlotter, so is recommended for most use cases. However, AnimationPlotter may be the
# better choice if:
#
# 1. the simulation timesteps are non-linear, or cannot be easily extracted
# 2. many data points are being displayed at once
# 3. the user desires to save the animation without using screen-capturing tools
# 4. the user desires to add custom data quickly.
