#!/usr/bin/env python
# coding: utf-8

"""
=========================================
Dealing with Out-Of-Sequence Measurements
=========================================
"""

# %%
# In real world tracking situations out of sequence
# measurements (OOSM) are a frequent issue and it is important to
# have tools to minimise their impact in our tracking
# capabilities.
# In literature there are sophisticated solutions
# to adress this challenge and in a series of examples we aim to
# provide a toolkit of approaches, better than just ignoring a set
# of measurements. In this example we focus on the simpler approach,
# also known as algorithm A [#]_ (also in [#]_ and [#]_), where we
# create a fixed lag storage of measurements and we go over the detections
# and place the OOSM in the correct order chain of measurements.
# The issue with this approach is that the $\math{\ell}$-storage
# of measurements can grow quickly if we are dealing with large number
# of sensors or targets, therefore computationally expensive.
# In other examples we present other algorithms and approaches to deal
# with OOSM.
#
# This example follows this structure:
# 1) Create ground truth and detections;
# 2) Instantiate the tracking components;
# 3) Run the tracker and visualise the results.
#

# %%
# General imports
# ^^^^^^^^^^^^^^^
import numpy as np
from datetime import datetime, timedelta

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
# In this example we consider a single target moving on a
# nearly constant velocity and an sensor obtains the detections.
# For simplicity we assume clutter to be negligible in this example.
# The OOS measurements are assumed to have a, known, fixed lag,
# the scans are happening every 5 seconds and these measurements
# are coming with a delay of 25 seconds.
#

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
    noise_covar=np.diag([15, 15]))

# Collect the measurements using scans
scan = []

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
                                  timestamp=truth[k].timestamp - timedelta(seconds=delay)))

    scan.append(detections)

# %%
# 2) Instantiate the tracking components;
# ---------------------------------------
# We have the scans containing the detections. It is time
# to prepare the tracking components, in this simple example we
# employ a :class:`~.KalmanPredictor` and :class:`~.KalmanUpdater`
# to perform the tracking.
#

from stonesoup.predictor.kalman import KalmanPredictor
predictor = KalmanPredictor(transition_model)

from stonesoup.updater.kalman import KalmanUpdater
updater = KalmanUpdater(measurement_model)

# create the prior
from stonesoup.types.state import GaussianState

prior = GaussianState(state_vector=[0, 1, -100, 0.3],
                      covar=np.diag([1, 1, 1, 1]),
                      timestamp=start_time)

# %%
# 3) Run the tracker and visualise the results.
# ---------------------------------------------
# We have the detections and the tracking components ready
# to be used. It is known that some detections arrived with a
# fixed delay, therefore we create a buffer storage of fixed dimension
# ($\math{\ell}$) where we store the detections and where we check the
# timestamps and their correct order. When one detection comes with an
# earlier timestamp ($\math{\tau}$<$t_{k}$), we shuffle the detections
# to keep the chain of detections.
#

# Load tracking components
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.track import Track

# Instantiate the empty track
track = Track()

# create the lag-storage to process the detections
lag_storage = []

for k in range(1, len(scan)):  # loop over the scans

    detections = scan[k]

    for detection in detections:
        # fill the buffer
        if k <= 5:
            lag_storage.append(detection)
        else:
            # now we have the buffer filled, check the detection timestamps order and release the earliest
            original_timestamps = [meas.timestamp for meas in lag_storage]

            # Obtain the ordered indeces
            index = np.argsort(original_timestamps)

            # re-order the arrival measurements
            lag_storage = [lag_storage[i] for i in index]

            # perform the tracking with the earliest arrival
            prediction = predictor.predict(prior, timestamp=lag_storage[0].timestamp)
            hypothesis = SingleHypothesis(prediction, lag_storage[0])
            post = updater.update(hypothesis)
            track.append(post)
            prior = track[-1]

            # clean the storage removing the first entry already used and add the last one considered
            lag_storage.pop(0)
            lag_storage.append(detection)

            # if we are at the last iteration, loop over the stored detections
            if k>=len(scan)-1:
                for kk in range(0, 4):
                    prediction = predictor.predict(prior, timestamp=lag_storage[kk].timestamp)
                    hypothesis = SingleHypothesis(prediction, lag_storage[kk])
                    post = updater.update(hypothesis)
                    track.append(post)
                    prior = track[-1]

# Now visualise the track
from stonesoup.plotter import AnimatedPlotterly
plotter = AnimatedPlotterly(timesteps=timestamps)
plotter.plot_ground_truths(truths, [0, 2])
plotter.plot_measurements(scan, [0, 2], measurements_label='Detections',
                          measurement_model=measurement_model)
plotter.plot_tracks(track, [0, 2])
plotter.fig

# %%
# Conclusion
# ----------
# In this simple example we have shown an algorithm to deal with
# out of sequence measurements, with the use a fixed lag buffer where
# we store the detections and we re-order their arrival time to
# adjust for any delay. In other examples we present more complex
# algorithms to handle the OOSM and perform accurate tracking.
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
#
