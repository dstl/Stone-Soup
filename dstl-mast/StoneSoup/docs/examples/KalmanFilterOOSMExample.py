#!/usr/bin/env python
# coding: utf-8

"""
===============================================
Kalman filter with Out-of-Sequence measurements
===============================================
"""

# %%
# In other examples we have shown how to deal with out-of-sequence measurements (OOSM)
# with methods like using inverse-time dynamics or creating a buffer where store and re-order
# the measurements according to their delay.
# In this example we present a method on how to deal with OOS measurements using Kalman
# filters, by making the assumption that the delay of the delayed measurements is known, and constant.
# The problem of OOS measurements is significant in real-world applications
# where data from different sources can have some delays and different timesteps
# (e.g., two sensors observing a target) due to systems configuration and different processing
# chain length.
#
# In the literature, there are examples on how to deal with such time-delays and
# uncertain timesteps [#]_.
# In this example we consider different approaches on how to deal with the OOSM,
# we have a tracker (called Tracker 1) which will run as the measurements arrive without any
# treatment of delay, in this tracker we are looping every timestep as :math:`t_{\text{now}}` and
# process every new detection without any adjustments on their time arrival.
#
# A second tracker (Tracker 2), built with the same components of Tracker 1, iterates
# at :math:`t_{\text{now}}`-:math:`t_{\text{delay}}` basically waiting for the delayed detections to arrive
# and adjusting for the delay. This tracker will lag behind the ground-truth detections which
# arrive at :math:`t_{\text{now}}`.
#
# As a control, we consider a third tracker (Tracker 3) which will work by excluding all the
# delayed detections (in this case considering only the detections from one sensor).
# The third tracker is a valid application and it is often suggested as viable option when dealing
# with OOSM.
#
# In this example, we consider Extended Kalman Filter algorithm components for each tracker.
#
# This example follows this structure:
#
# 1. prepare the ground truth;
# 2. set up the sensors and generate the measurements;
# 3. instantiate the tracking components;
# 4. run the trackers and visualise the results.
#

# %%
# General imports
# ^^^^^^^^^^^^^^^
import numpy as np
from datetime import datetime, timedelta
from copy import deepcopy

# Simulations parameters
start_time = datetime.now().replace(microsecond=0)
np.random.seed(2000)
num_steps = 50  # number of timesteps of the simulation

# %%
# Stone Soup imports
# ^^^^^^^^^^^^^^^^^^
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
    ConstantVelocity
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.types.state import GaussianState


# instantiate the transition model
transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.5),
                                                          ConstantVelocity(0.5)])

# %%
# 1. Prepare the ground truth;
# ----------------------------
# In this example, we consider a single target moving with near constant velocity.

# initiate the groundtruth
truth = GroundTruthPath([GroundTruthState([0, 1, 0, 1], timestamp=start_time)])

# iterate over the various timesteps
for k in range(1, num_steps):
    truth.append(GroundTruthState(
        transition_model.function(truth[k - 1], noise=True,
                                  time_interval=timedelta(seconds=2)),
        timestamp=start_time + timedelta(seconds=2*k)))

# %%
# 2. Set up the sensors and generate the measurements;
# ----------------------------------------------------
# We consider two ideal sensors using :class:`~.CartesianToBearingRange` measurement model.
# The second sensor sends the detections with a fixed delay of 5 seconds.
# In this scenario, there is a fixed, constant, delay between the two sets of detections.
# The two measurement models have different translation offsets due to the location of the sensors.
# In this scenario we consider a negligible clutter noise.

# Load the measurement model
from stonesoup.models.measurement.nonlinear import CartesianToBearingRange

measurement_model_1 = CartesianToBearingRange(  # relative to the first sensor
    ndim_state=4,
    mapping=(0, 2),
    noise_covar=np.diag([np.radians(3), 20]),
    translation_offset=np.array([[-60], [0]]))

measurement_model_2 = CartesianToBearingRange(  # relative to the second sensor
    ndim_state=4,
    mapping=(0, 2),
    noise_covar=np.diag([np.radians(3), 20]),
    translation_offset=np.array([[-150], [60]]))

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
# We have generated two sets of detections of the same target, one for each sensor, with the latter
# where the detection timestamp has a fixed delay of 5 seconds.
#
# Let's visualise the track and the set of detections. We use :class:`~.FixedPlatform` to show the
# sensors locations.

from stonesoup.platform.base import FixedPlatform

# Only for plotting purposes
sensor1_platform = FixedPlatform(
    states=GaussianState([-60, 0, 0, 0],
                         np.diag([1, 0, 1, 0])),
    position_mapping=(0, 2),
    sensors=None)

sensor2_platform = FixedPlatform(
    states=GaussianState([-200, 0, 60, 0],
                         np.diag([1, 0, 1, 0])),
    position_mapping=(0, 2),
    sensors=None)

from stonesoup.plotter import AnimatedPlotterly

time_steps = [start_time + timedelta(seconds=2*i) for i in range(num_steps + 5)]

plotter = AnimatedPlotterly(timesteps=time_steps)
plotter.plot_ground_truths(truth, [0, 2])
plotter.plot_measurements(measurements1, [0, 2], marker=dict(color='blue', symbol='0'),
                          measurements_label='Detections with no lag')
plotter.plot_measurements(measurements2, [0, 2], marker=dict(color='orange', symbol='0'),
                          measurements_label='Detections with lag')
plotter.plot_sensors([sensor1_platform, sensor2_platform],
                     marker=dict(color='black', symbol='129', size=15),
                     sensor_label='Fixed Platforms')
plotter.fig

# %%
# 3) Instantiate the tracking components;
# ---------------------------------------
# In this example we employ an Extended Kalman Filter (EKF) components by loading the predictor and updater using
# :class:`~.ExtendedKalmanPredictor` and :class:`~.ExtendedKalmanUpdater`.
# The choice of these components comes from the non-linear nature of the measurement
# model chosen.

# load the extended kalman filter components
from stonesoup.updater.kalman import ExtendedKalmanUpdater
from stonesoup.predictor.kalman import ExtendedKalmanPredictor

# EKF predictor
predictor = ExtendedKalmanPredictor(transition_model)

# We employ two updaters to account for the different sensor translation offsets
updater1 = ExtendedKalmanUpdater(measurement_model_1)
updater2 = ExtendedKalmanUpdater(measurement_model_2)

# Track priors
prior1 = GaussianState(state_vector=np.array([0, 1, 0, 1]),
                        covar=np.diag([1, 1, 1, 1]),
                        timestamp=start_time)

prior2 = GaussianState(state_vector=np.array([0, 1, 0, 1]),
                        covar=np.diag([1, 1, 1, 1]),
                        timestamp=start_time+timedelta(seconds=5))
prior3 = deepcopy(prior1)

# %%
# 4) Run the trackers and visualise the results.
# ----------------------------------------------
# We have prepared the tracker components and we are ready to generate the final tracks.
#
# Tracker 1 will consider all the detections as they are arriving from the sensors considering
# each timestep as :math:`t_{\text{now}}`. We should expect that as the delayed detections start to
# arrive the tracking quality will significantly drop.
#
# Tracker 2 will be lagging behind the timesteps, at :math:`t_{\text{now}}+t_{\text{delay}}`, in this way
# the tracker will wait for the delayed detections to arrive and will consider them in the
# correct order and correct timestep. However the tracks will be behind the ground-truth track.
#
# The final tracker (3) will ignore all detections from delayed sensor.
#
# As we have obtained all the tracks for each tracker we will visualise them.

from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.track import Track

# Evaluate the delay between the measurements
delay = measurements2[0].timestamp - measurements1[0].timestamp

# Initiate the empty track for each tracker
track1 = Track(prior1)  # Tracker 1 prior
track2 = Track(prior2)  # Tracker 2 prior
track3 = Track(prior3)  # Tracker 3 prior


for k in range(num_steps+5):  # loop over the timestep

    # Check if we have already get the delay
    check_delay = timedelta(seconds=k)

    # When we reach the delay fix the number of timestep
    if check_delay == delay:
        timestep_delay = k

    if check_delay < delay:  # if we are below the delay, use only the first detections
        prediction = predictor.predict(prior1, timestamp=measurements1[k].timestamp)
        hypothesis = SingleHypothesis(prediction, measurements1[k])
        post = updater1.update(hypothesis)
        track1.append(post)
        prior1 = track1[-1]

    else:  # got the delay
        if k < num_steps:  # if we are not at the end of first scans
            prediction = predictor.predict(prior1, timestamp=measurements1[k].timestamp)
            hypothesis = SingleHypothesis(prediction, measurements1[k])
            post = updater1.update(hypothesis)
            track1.append(post)

            prediction = predictor.predict(prior1,
                                           timestamp=measurements2[k - timestep_delay].timestamp)
            hypothesis = SingleHypothesis(prediction,
                                          measurements2[k - timestep_delay])
            post = updater2.update(hypothesis)
            track1.append(post)
            prior1 = track1[-1]

        else:  # consider only the second sensors detections
            prediction = predictor.predict(prior1,
                                           timestamp=measurements2[k - timestep_delay].timestamp)
            hypothesis = SingleHypothesis(prediction,
                                          measurements2[k - timestep_delay])
            post = updater2.update(hypothesis)
            track1.append(post)
            prior1 = track1[-1]

    # Tracker 2
    if check_delay >= delay:
        prediction = predictor.predict(prior2,
                                       timestamp=measurements1[k - timestep_delay].timestamp + delay)
        hypothesis = SingleHypothesis(prediction, measurements1[k - timestep_delay])
        post = updater1.update(hypothesis)
        track2.append(post)
        prediction = predictor.predict(prior2,
                                       timestamp=measurements2[k - timestep_delay].timestamp)
        hypothesis = SingleHypothesis(prediction,
                                      measurements2[k - timestep_delay])
        post = updater2.update(hypothesis)
        track2.append(post)
        prior2 = track2[-1]

    # Tracker 3, the "control tracker"
    if k < num_steps:
        prediction = predictor.predict(prior3,
                                       timestamp=measurements1[k].timestamp)
        hypothesis = SingleHypothesis(prediction,
                                      measurements1[k])
        post = updater1.update(hypothesis)
        track3.append(post)
        prior3 = track3[-1]


# %%
# Visualise the tracks
# ^^^^^^^^^^^^^^^^^^^^
# The current implementation of :class:`~.AnimatedPlotterly` does not allow for a
# "live" representation of the tracks with out of sequence measurements and tracks lagging
# behind the :math:`t_{\text{now}}`. As well, the tracks are produced after the end of the simulation
# with the latest :math:`t_{\text{delay}}` timesteps of the track are not yet available.
# However, it is interesting to see a 1-to-1 comparison between the three trackers, even if the
# Tracker 2 track is not, visually, lagging behind.

plotter.plot_tracks(track1, [0, 2], track_label='Tracker 1')
plotter.plot_tracks(track2, [0, 2], track_label='Tracker 2',
                    line=dict(color='red'))
plotter.plot_tracks(track3, [0, 2], track_label='Tracker 3',
                    line=dict(color='green'))
plotter.fig

# %%
# Conclusions
# -----------
# In this simple example we have presented how it is possible to perform the tracking with the
# presence of out of sequence or delayed measurements from a sensor.
# We have shown a comparison between three different approaches using the same algorithm.
# Tracker 1, which ignores the time of arrival, has a more uncertain track when the delayed
# detections are arriving. Tracker 2, fixes the time arrival of the detections and runs behind live
# time (which can be considered Tracker 1). The final one, which ignores the delayed detections
# is used as control, and in this simplistic case with no clutter and a simple trajectory performs
# quite as well as the second tracker.

# %%
# References
# ----------
# .. [#] S. R. Maskell, R. G. Everitt, R. Wright, M. Briers, 2005,
#        Multi-target out-of-sequence data association: Tracking using
#        graphical models, Information Fusion.

