#!/usr/bin/env python
# coding: utf-8

"""
=========================================
Dealing with Out-Of-Sequence Measurements
=========================================
"""

# %%
# In real world tracking situations, out of sequence measurements (OOSM) are a frequent
# issue and it is important to have tools to minimise their impact in our tracking capabilities.
#
# In literature there are sophisticated solutions to adress this challenge and in a series of examples and
# we aim to provide a toolkit of approaches, better than just chosing to ignore such measurements.
#
# In this example we focus on the simpler approach, also known as algorithm A [#]_ (also in [#]_ and [#]_),
# where we create a ``fixed lag'' storage of measurements and we go over the detections
# and place the OOSM in the correct order chain of measurements.
# The issue with this approach is that the :math:`\ell`-storage of measurements can grow
# quickly if we are dealing with large number of sensors or targets, therefore computationally expensive.
# As comparison we add a track where the detections are processed as they arrive at the tracker (no
# reordering based on the detection timestamp) to prove that the tracking performance is worse in that scenario.
#
# In other examples we present other algorithms and approaches to deal with OOSM in different manners (e.g. time
# inverse dynamics).
#
# This example follows this structure:
#   1. Create ground truth and detections;
#   2. Instantiate the tracking components;
#   3. Run the tracker and visualise the results.
#

# %%
# General imports
# ^^^^^^^^^^^^^^^
import numpy as np
from datetime import datetime, timedelta
from copy import deepcopy

# Simulation parameters
start_time = datetime.now().replace(microsecond=0)
np.random.seed(1908)
num_steps = 65  # simulation steps

# %%
# Stone Soup Imports
# ^^^^^^^^^^^^^^^^^^

from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
    ConstantVelocity
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.types.state import GaussianState, State, StateVector

# %%
# 1) Create ground truth and detections;
# --------------------------------------
# In this example we consider a single target moving on a nearly constant velocity trajectory
# and an sensor obtains the detections.
#
# For simplicity we assume clutter to be negligible in this example. The OOS measurements are assumed to
# have a, known, fixed lag, the scans are happening every 5 seconds and these measurements
# are coming with a delay of 25 seconds.
#
# To model the delayed arrival of detections we record the arrival time of the scans and
# we add the delay on the arrival time only, while keeping the detection timestamp correct.
# Then,  we re-order the scans by their arrival time. In this way the tracker will receive the
# detection as their arrival time, creating the presence of delayed measurements and allowing to apply the
# algorithm.

# instantiate the transition model
transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.05),
                                                          ConstantVelocity(0.05)])

# Create a list of timesteps
timestamps = [start_time]

# create a set of truths
truths = set()

# Instantiate the groundtruth for the first target
truth = GroundTruthPath([GroundTruthState([0, 1, -100, 0.3], timestamp=start_time)])

for k in range(1, num_steps):
    truth.append(GroundTruthState(
        transition_model.function(truth[k-1], noise=True,
                                  time_interval=timedelta(seconds=2)),
        timestamp=start_time + timedelta(seconds=5*k)))
    timestamps.append(truth.timestamp)

truths.add(truth)

# Create the measurements models
from stonesoup.models.measurement.linear import LinearGaussian

measurement_model = LinearGaussian(
    ndim_state=4,
    mapping=(0, 2),
    noise_covar=np.diag([50, 50]))

# Collect the measurements using scans
scans_detections = []
scans = []

# Create the detections
from stonesoup.types.detection import TrueDetection

# Loop over the timesteps and collect the detections
for k in range(num_steps):
    detections = set()

    # Introduce the delay
    if k%5==0:
        delay = 25
    else:
        delay = 0

    measurement = measurement_model.function(truth[k], noise=True)
    detections.add(TrueDetection(state_vector=measurement,
                                  groundtruth_path=truth,
                                  timestamp=truth[k].timestamp))
    # store the scans into a list for plotting
    scans_detections.append(detections)

    # Scans for tracking and reordering
    scans.append((truth[k].timestamp + timedelta(seconds=delay), detections))

# Reorder the scans by their arrival time
arrival_time_ordered = sorted(scans, key=lambda dscan: dscan[0])

# %%
# 2. Instantiate the tracking components;
# ---------------------------------------
# We have the scans containing the detections ordered by their arrival time.
# It is time to prepare the tracking components, in this simple example we employ a
# :class:`~.KalmanPredictor` and :class:`~.KalmanUpdater` to perform the tracking.
#

from stonesoup.updater.kalman import KalmanUpdater
from stonesoup.predictor.kalman import KalmanPredictor
predictor = KalmanPredictor(transition_model)
updater = KalmanUpdater(measurement_model)

# create the prior
from stonesoup.types.state import GaussianState

prior = GaussianState(state_vector=[0, 1, -100, 0.3],
                      covar=np.diag([1, 1, 1, 1]),
                      timestamp=start_time)

# duplicate the prior
prior_lag = deepcopy(prior)

# %%
# 3. Run the tracker and visualise the results.
# ---------------------------------------------
# We have the detections and the tracking components ready to be used.
#
# It is known that some detections arrived with a fixed delay, therefore we create a
# buffer storage of fixed dimension (:math:`\ell`) where we store the detections and where we check the
# timestamps and adjust them to their correct order. When one detection comes with an
# earlier timestamp (:math:`\tau < t_{k}`), we shuffle the detections to keep the chain of detections.
#
# To show how the delayed detections impact the tracking we run a tracker with the detections
# as they arrive to the tracker without any modification.

# Load tracking components
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.track import Track

# Instantiate the empty tracks
track = Track(prior)
track_lag = deepcopy(track)

# Load the plotter
from stonesoup.plotter import AnimatedPlotterly

# Consider the case without the algorithm
for k in range(len(arrival_time_ordered)):  # loop over the scans

    scan = arrival_time_ordered[k][1]

    for detection in scan:
        prediction = predictor.predict(prior_lag, timestamp=detection.timestamp)
        hypothesis = SingleHypothesis(prediction, detection)
        post = updater.update(hypothesis)
        track_lag.append(post)
        prior_lag = track_lag[-1]

# %%
# Visualise the tracking
# ^^^^^^^^^^^^^^^^^^^^^^
# Now visualise the detections and the track without the algorithm applied.

plotter = AnimatedPlotterly(timesteps=timestamps)
plotter.plot_ground_truths(truths, [0, 2])
plotter.plot_measurements(scans_detections, [0, 2], measurements_label='Detections',
                          measurement_model=measurement_model)
plotter.plot_tracks(track_lag, [0, 2], line= dict(color='grey'), track_label='Track with lag')
plotter.fig

# create the lag-storage to process the detections
lag_storage = []

# Run the algorithm to deal with the
for k in range(len(arrival_time_ordered)):
    # create a list of times for the timestamps
    original_timestamps = []

    # fill the buffer
    if k <= 4:
        lag_storage.append(arrival_time_ordered[k][1])  # feed the in the arrival order
    else:
        # now we have the buffer filled, check the detection timestamps order and release the earliest
        for scan in lag_storage:
            for detection in scan:
                original_timestamps.append(detection.timestamp)

        # Re-order the timestamps detections
        index = np.argsort(original_timestamps)

        # Re-order the scans by their detections
        lag_storage = [lag_storage[i] for i in index]

        # feed the detections to the tracker
        for detection in lag_storage[0]:
            prediction = predictor.predict(prior, timestamp=detection.timestamp)
            hypothesis = SingleHypothesis(prediction, detection)
            post = updater.update(hypothesis)
            track.append(post)
            prior = track[-1]

        # clean the storage removing the first entry already used and add the last one considered
        lag_storage.pop(0)
        lag_storage.append(arrival_time_ordered[k][1])

        # if we are at the last iteration, loop over the stored detections
        if k>=len(arrival_time_ordered)-1:
            for kk in range(0, 4):
                for detection in lag_storage[kk]:
                    prediction = predictor.predict(prior, timestamp=detection.timestamp)
                    hypothesis = SingleHypothesis(prediction, detection)
                    post = updater.update(hypothesis)
                    track.append(post)
                    prior = track[-1]
# %%
# Adding the track with algorithm
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

plotter.plot_tracks(track, [0, 2], line= dict(color='blue'), track_label='Track with algorithm')
plotter.fig

# %%
# Conclusion
# ----------
# In this simple example we have shown an algorithm to deal with out of sequence measurements,
# with the use a fixed lag buffer where we store the detections and we re-order their arrival time to
# adjust for any delay. As well, we have shown how poorly is the tracking if we don't
# consider any changes in the detection order.
# In other examples we present more complex algorithms to handle the OOSM and perform accurate tracking.
#

# %%
# References
# ----------
# .. [#] Y. Bar-Shalom, M. Mallick, H. Chen, R. Washburn, 2002,
#        One-step solution for the general out-of-sequence measurement
#        problem in tracking, Proceedings of the 2002 IEEE Aerospace
#        Conference.
# .. [#] Y. Bar-Shalom, 2002, Update with out-of-sequence measurements in tracking:
#        exact solution, IEEE Transactons on Aerospace and Electronic Systems 38.
# .. [#] S. R. Maskell, R. G. Everitt, R. Wright, M. Briers, 2005,
#        Multi-target out-of-sequence data association: Tracking using
#        graphical models, Information Fusion.
