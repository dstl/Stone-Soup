#!/usr/bin/env python
# coding: utf-8

"""
===============================================
Kalman filter with Out-of-Sequence measurements
===============================================
"""

# %%
# In other examples we have shown how to deal with out-of-sequence measurements (OOSM)
# with methods using inverse-time dynamics or creating a buffer where store and re-order
# the measurements.
# In this example we present how to deal with OOS measurements using Kalman
# filters. The problem of OOS measurements is significant in real-world applications
# where data from different sources can have some delays and different timesteps
# (e.g., two sensors observing a target).
# In the literature, there are examples (e.g., [#]_) on how to deal with such time-delays and
# uncertain timesteps.
# In this example, we present a simpler application using the Ensemble Kalman Filter algorithm
# available in Stone Soup to tackle this problem and we compare against an Extended Kalman
# filter to measure the differences.
#
# This example follows the following structure:
#
# 1. prepare the ground truth;
# 2. set up the sensors and generate the measurements;
# 3. instantiate the tracking components;
# 4. run the tracker and visualise the results.
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
transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.5),
                                                          ConstantVelocity(0.5)])

# %%
# 1) Prepare the ground truth;
# ----------------------------

# initiate the groundtruth
truth = GroundTruthPath([GroundTruthState([0, 1, 0, 1], timestamp=start_time)])

# iterate over the various timesteps
for k in range(1, num_steps):
    truth.append(GroundTruthState(
        transition_model.function(truth[k - 1], noise=True,
                                  time_interval=timedelta(seconds=1)),
        timestamp=start_time + timedelta(seconds=k)))

# %%
# 2) Set up the sensors and generate the measurements;
# -----------------------------------------------------
# In this example we consider two ideal radars using :class:`~.CartesianToBearingRange` measurement
# model.
# The second sensor sends the detections with a fixed delay of 5 seconds. So the two sets of
# detections have a fixed, constant, delay.
# Now, we can collect the measurements from the two sensors.
# The two measurement model have different translation offset from the location of the sensors.
#

# Load the measurement model
from stonesoup.models.measurement.nonlinear import CartesianToBearingRange

measurement_model_1 = CartesianToBearingRange(  # relative to the first sensor
    ndim_state=4,
    mapping=(0, 2),
    noise_covar=np.diag([np.radians(1), 20]),
    translation_offset=np.array([[-60], [0]]))

measurement_model_2 = CartesianToBearingRange(  # relative to the second sensor
    ndim_state=4,
    mapping=(0, 2),
    noise_covar=np.diag([np.radians(1), 20]),
    translation_offset=np.array([[20], [60]]))

# Generate the detections
from stonesoup.types.detection import Detection

# Instantiate two list for the detections
measurements1 = []
measurements2 = []
for state in truth:  # loop over the ground truth detections
    measurement = measurement_model_1.function(state, noise=True)
    measurements1.append(Detection(measurement, timestamp=state.timestamp,
                                   measurement_model=measurement_model_1))
    # collect the measurements for the delayed radar
    measurement = measurement_model_2.function(state, noise=True)
    measurements2.append(Detection(measurement, timestamp=state.timestamp + timedelta(seconds=5),
                                   measurement_model=measurement_model_2))

# %%
# We have generated two sets of detections of the same target, one for each radar, with the latter
# where the detection timestamp has a fixed delay of 5 seconds. Let's visualise the track and the
# set of detections. To plot the sensors we use a :class:`~.FixedPlatform`.
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

from stonesoup.plotter import AnimatedPlotterly

time_steps = [start_time + timedelta(seconds=i) for i in range(num_steps + 10)]

plotter = AnimatedPlotterly(timesteps=time_steps)
plotter.plot_ground_truths(truth, [0, 2])
plotter.plot_measurements(measurements1, [0, 2], marker=dict(color='blue', symbol='0'),
                          measurements_label='Detections with no lag')
plotter.plot_measurements(measurements2, [0, 2], marker=dict(color='orange', symbol='0'),
                          measurements_label='Detections with lag')
plotter.plot_sensors([sensor1_platform, sensor2_platform],
                     marker=dict(color='black', symbol='129', size=15),
                     sensor_label='Fixed Platforms')
plotter.show()

# %%
# 3) Instantiate the tracking components;
# ---------------------------------------
# In this example we employ an Ensemble Kalman filter by loading the predictor and updater using
# :class:`~.EnsemblePredictor` and :class:`~.EnsembleUpdater` and an Extended Kalman filter using
# :class:`~.ExtendedKalmanPredictor` and :class:`~.ExtendedKalmanUpdater`.
# The Ensemble Kalman filter combines the functionality of the Kalman filter and the
# particle filter. To access the two sets of detections we initialise two different updaters,
# with the correct measurement models, to take into account the detection translation offset.
# For simplicity, we assume that the detections obtained by the second sensor can be corrected by
# the delay and correspond to the first set of measurements.
#

# Load the ensemble components
from stonesoup.predictor.ensemble import EnsemblePredictor
from stonesoup.updater.ensemble import EnsembleUpdater

# load the extended kalman filter components
from stonesoup.updater.kalman import ExtendedKalmanUpdater
from stonesoup.predictor.kalman import ExtendedKalmanPredictor

# EKF components
predictor = ExtendedKalmanPredictor(transition_model)

# We employ two updaters to account for the different sensor translation offsets
updater1 = ExtendedKalmanUpdater(measurement_model_1)
updater2 = ExtendedKalmanUpdater(measurement_model_2)

# EnKF componens
EnKFpredictor = EnsemblePredictor(transition_model)

EnKFupdater1 = EnsembleUpdater(measurement_model_1)
EnKFupdater2 = EnsembleUpdater(measurement_model_2)

# Load the Ensemble for the prior
from stonesoup.types.state import EnsembleState

ensemble = EnsembleState.generate_ensemble(
    mean=np.array([0, 1, 0, 1]),
    covar=np.diag([5, 2, 5, 2]),
    num_vectors=100)
# This approach is similar to setting up the priors for Particle filter tracking
EnKFprior = EnsembleState(state_vector=ensemble, timestamp=start_time)

# EKF prior
prior = GaussianState(state_vector=np.array([0, 1, 0, 1]),
                        covar=np.diag([1, 1, 1, 1]),
                        timestamp=start_time)

# %%
# 4) Run the tracker and visualise the results.
# ---------------------------------------------
# We have prepared the tracker components and we are ready to generate the final tracks.
# Before passing the measurements to the tracking algorithm we need to correct the detections by
# the delay so we can process them correctly. Finally we can plot the resulting track.
#

from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.track import Track

# Evaluate the delay between the measurements
delay = measurements2[0].timestamp - measurements1[0].timestamp

# Initiate the empty track
track = Track(prior)
EnKFtrack = Track(EnKFprior)

for k in range(num_steps):  # loop over the timestep

    # We consider the same prior - EKF
    prediction = predictor.predict(prior, timestamp=measurements1[k].timestamp)
    hypothesis = SingleHypothesis(prediction, measurements1[k])
    post = updater1.update(hypothesis)
    track.append(post)

    # correct the measurements timestamp with the delay
    prediction = predictor.predict(prior, timestamp=measurements2[k].timestamp - delay)
    hypothesis = SingleHypothesis(prediction, measurements2[k])
    post = updater2.update(hypothesis)
    EnKFtrack.append(post)
    prior = track[-2]

    # EnKF
    prediction = EnKFpredictor.predict(EnKFprior, timestamp=measurements1[k].timestamp)
    hypothesis = SingleHypothesis(prediction, measurements1[k])
    post = EnKFupdater1.update(hypothesis)
    EnKFtrack.append(post)

    # correct the measurements timestamp with the delay
    prediction = EnKFpredictor.predict(EnKFprior, timestamp=measurements2[k].timestamp - delay)
    hypothesis = SingleHypothesis(prediction, measurements2[k])
    post = EnKFupdater2.update(hypothesis)
    EnKFtrack.append(post)
    EnKFprior = EnKFtrack[-2]

plotter.plot_tracks(track, [0, 2], track_label='EKF')
plotter.plot_tracks(EnKFtrack, [0, 2], uncertainty=False, particle=False,
                    track_label='EnKF', line=dict(color='red'))
plotter.show()

# %%
# Conclusions
# -----------
# In this simple example we have presented how it is possible to perform the tracking with the
# presence of out of sequence or delayed measurements from a sensor.
# We have shown two cases one using an Extended Kalman filter and the Ensemble Kalman filter.
# In both cases we have corrected the measurements with delay and we have performed the tracking,
# in this simpler example we don't appreciate significant differences between the two algorithms.
# In more complex scenario (multi-target and/or cluttered scans) the particle-filter like approach of the
# Ensemble Kalman Filter can provide better tracking performances.

# %%
# References
# ----------
# .. [#] S. Pornsarayouth and M. Yamakita, "Ensemble Kalman filtering of out-of-sequence
#        measurements for continuous-time model," 2012 American Control Conference (ACC),
#        Montreal, QC, Canada, 2012, pp. 4801-4806, doi: 10.1109/ACC.2012.6315469.
