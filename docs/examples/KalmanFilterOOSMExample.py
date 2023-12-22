#!/usr/bin/env python
# coding: utf-8

"""
===============================================
Kalman filter with Out-of-Sequence measurements
===============================================
"""

# %
# In this example we present how to deal with
# out of sequence measurements (OOSM) using
# Kalman filters. This problem is significant
# in real-world applications where data
# from different sources can have some delays
# and different timesteps (e.g., two sensors
# observing a target). In literature there are
# examples (e.g., [#]_) on how to deal with
# such time-delays and uncertain timesteps.
# In this example we present a simpler application
# using the Ensemble Kalman Filter algorithm
# available in Stone Soup to tackle this problem.
#
# This example follows the following structure:
# 1) prepare the ground truth;
# 2) set up the sensors and generate the measaurements;
# 3) instantiate the tracking components;
# 4) run the tracker and visualise the results.
#

# %%
# General imports
# ^^^^^^^^^^^^^^^
import numpy as np
from datetime import datetime, timedelta

# Simulations parameters
start_time = datetime.now().replace(microsecond=0)
np.random.seed(2000)
num_steps = 50  # number of timesteps of the simulation

# %%
# Stone soup imports
# ^^^^^^^^^^^^^^^^^^
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
    ConstantVelocity
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.types.state import GaussianState

# instantiate the transition model
transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.05),
                                                          ConstantVelocity(0.05)])

# initiate the groundtruth
truth = GroundTruthPath([GroundTruthState([0, 1, 0, 1], timestamp=start_time)])

# iterate over the various timesteps
for k in range(1, num_steps):
    truth.append(GroundTruthState(
        transition_model.function(truth[k - 1], noise=True,
                                  time_interval=timedelta(seconds=1)),
        timestamp=start_time + timedelta(seconds=k)))

# %%
# 2) Set up the sensors and generate the measaurements;
# -----------------------------------------------------
# In this example we consider two ideal radars using
# :class:`~.CartesianToBearingRange` measurement model.
# The second sensor sends the detections with a
# fixed delay of 5 seconds. So the two sets of
# detections have a fixed, constant, delay.
# Now, we can collect the measurements from the
# two sensors. The two measurement model have
# differnet translation offset from the location of
# the sensors.
#

# Load the measurement model
from stonesoup.models.measurement.nonlinear import CartesianToBearingRange

measurement_model_1 = CartesianToBearingRange(  # relative to the first sensor
    ndim_state=4,
    mapping=(0, 2),
    noise_covar=np.diag([np.radians(0.1), 3]),
    translation_offset=np.array([[-60], [0]]))

measurement_model_2 = CartesianToBearingRange(  # relative to the first sensor
    ndim_state=4,
    mapping=(0, 2),
    noise_covar=np.diag([np.radians(0.1), 3]),
    translation_offset=np.array([[20], [60]]))

# Generate the detections
from stonesoup.types.detection import Detection

# Instantiate two list for the detections
measurements1 = []
measurements2 = []
for state in truth:  # loop over the groud truth detections
    measurement = measurement_model_1.function(state, noise=True)
    measurements1.append(Detection(measurement, timestamp=state.timestamp,
                                  measurement_model=measurement_model_1))
    # collect the measurements for the delayed radar
    measurement = measurement_model_2.function(state, noise=True)
    measurements2.append(Detection(measurement, timestamp=state.timestamp + timedelta(seconds=5),
                                  measurement_model=measurement_model_2))

# %%
# We have generated two sets of
# detections of the same target, one for each radar,
# with the latter where the detection timestamp
# has a fixed delay of 5 seconds. Let's visualise the track and the
# set of detections. To plot the sensors we use a
# :class:`~.FixedPlatform` object.
#
from stonesoup.platform.base import FixedPlatform

# Only for plotting purposes
sensor1_platform = FixedPlatform(
    states=GaussianState([-60, 0, 0, 0],
                         np.diag([1, 0, 1, 0])),
    position_mapping=(0, 2),
    sensors=None)

sensor2_platform = FixedPlatform(
    states=GaussianState([20, 0, 60, 0],
                         np.diag([1, 0, 1, 0])),
    position_mapping=(0, 2),
    sensors=None)

from stonesoup.plotter import Plotterly

plotter = Plotterly()
plotter.plot_ground_truths(truth, [0, 2])
plotter.plot_measurements(measurements1, [0, 2], marker=dict(color='blue', symbol='0'),
                          measurements_label='Detections with no lag')
plotter.plot_measurements(measurements2, [0, 2], marker=dict(color='orange', symbol='0'),
                          measurements_label='Detections with lag')
plotter.plot_sensors([sensor1_platform, sensor2_platform],
                     [0, 1], marker=dict(color='black', symbol='129', size=15),
                     sensor_label='Fixed Platforms')
plotter.fig

# %%
# 3) Instantiate the tracking components;
# ---------------------------------------
# In this example we employ an Ensemble Kalman filter
# by loading the predictor and updater using :class:`~.EnsemblePredictor`
# and :class:`~.EnsembleUpdater`. This filter combines
# the functionality of the Kalman filter and the particle
# filter. To access the two sets of detections we
# initialise two different updaters, with the correct
# measurement models, to take into account the
# detection translation offset.
# For simplicity, we assume that the detections
# obtained by the second sensor can be corrected by the delay
# and correspond to the first set of measurements.
#

from stonesoup.predictor.ensemble import EnsemblePredictor
from stonesoup.updater.ensemble import EnsembleUpdater

predictor = EnsemblePredictor(transition_model)
# We employ two updaters for accounting the translation offset
updater1 = EnsembleUpdater(measurement_model_1)
updater2 = EnsembleUpdater(measurement_model_2)

# Load the Ensemble for the prior
from stonesoup.types.state import EnsembleState

ensemble = EnsembleState.generate_ensemble(
    mean=np.array([0, 1, 0, 1]),
    covar=np.diag([5, 2, 5, 2]),
    num_vectors=100)
# This approach is similar to setting up the
# priors for the Particle filter tracking
prior = EnsembleState(state_vector=ensemble, timestamp=start_time)

from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.track import Track

# %%
# 4) Run the tracker and visualise the results.
# ---------------------------------------------
# We have prepared the tracker components and we
# are ready to generate the final track.
# Before passing the measurements to the
# tracking algorithm we need to correct the
# detections by the delay so we can process them
# correctly. Finally we can plot the resulting track.
#

# Evaluate the delay between the measurements
delay = measurements2[0].timestamp - measurements1[0].timestamp

# Initiate the empty track
track = Track()
for k in range(num_steps):  # loop over the timestep
    # We consider the same prior
    prediction = predictor.predict(prior, timestamp=measurements1[k].timestamp)
    hypothesis = SingleHypothesis(prediction, measurements1[k])
    post = updater1.update(hypothesis)
    track.append(post)

    # correct the measurements timestamp with the delay
    prediction = predictor.predict(prior, timestamp=measurements2[k].timestamp - delay)
    hypothesis = SingleHypothesis(prediction, measurements2[k])
    post = updater2.update(hypothesis)
    track.append(post)
    prior = track[-1]

plotter.plot_tracks(track, [0, 2], uncertainty=True, particle=True)
plotter.fig

# %%
# Conclusions
# -----------
# In this simple example we have presented
# how it is possible to perform the tracking
# with the presence of out of sequence or delayed
# measurements from a sensor. We have employed
# an Ensemble Kalman filter to ease the tracking.
#

# References
# ----------
# .. [#] S. Pornsarayouth and M. Yamakita, "Ensemble Kalman
#        filtering of out-of-sequence measurements for
#        continuous-time model," 2012 American Control
#        Conference (ACC), Montreal, QC, Canada, 2012,
#        pp. 4801-4806, doi: 10.1109/ACC.2012.6315469.
