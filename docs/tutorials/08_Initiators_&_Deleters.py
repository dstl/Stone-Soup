#!/usr/bin/env python
# coding: utf-8

"""
8 - Initiators & Deleters
=========================
So far we have provided a prior in all our examples, defining were we think our tracks will start.
This also has been for a fixed number of tracks. In practice, targets may be "born" and "die"
all the time. This could be because they are going in/out of range of the sensor's field of view.
The location/state of the targets' "birth" may also not be known and varying.
"""

# %%
# Simulating Multiple Targets
# ---------------------------
# Here we'll simulate multiple targets moving at a constant velocity. A poisson distribution will
# be used to decide how many new targets are born at a particular timestamp, and simple random draw
# will be used to decide if the targets will be removed. Each target will have an random position
# and velocity on birth.
from datetime import datetime
from datetime import timedelta

import numpy as np
from matplotlib import pyplot as plt

from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
                                               ConstantVelocity
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState

start_time = datetime.now()
truths = set()  # Truths across all time
current_truths = set()  # Truths alive at current time

transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.005),
                                                          ConstantVelocity(0.005)])

for k in range(20):
    # Death
    for truth in current_truths.copy():
        if np.random.rand() <= 0.05:  # Death probability
            current_truths.remove(truth)
    # Update truths
    for truth in current_truths:
        truth.append(GroundTruthState(
            transition_model.function(truth[-1], noise=True, time_interval=timedelta(seconds=1)),
            timestamp=start_time + timedelta(seconds=k)))
    # Birth
    for _ in range(np.random.poisson(0.6)):  # Birth probability
        x, y = initial_position = np.random.rand(2) * [20, 20]  # Range [0, 20] for x and y
        x_vel, y_vel = (np.random.rand(2))*2 - 1  # Range [-1, 1] for x and y velocity
        state = GroundTruthState([x, x_vel, y, y_vel], timestamp=start_time + timedelta(seconds=k))

        # Add to truth set for current and for all timestamps
        truth = GroundTruthPath([state])
        current_truths.add(truth)
        truths.add(truth)


fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_ylim(-5, 25)
ax.set_xlim(-5, 25)

for truth in truths:
    ax.plot([state.state_vector[0] for state in truth],
            [state.state_vector[2] for state in truth],
            linestyle="--",)

# %%
# Generate Detections and Clutter
# -------------------------------
# Next, generate detections with clutter just as in the previous tutorials, skipping over the truth
# paths that weren't alive at the current time step.

from scipy.stats import uniform
from stonesoup.types.detection import TrueDetection
from stonesoup.types.detection import Clutter
from stonesoup.models.measurement.linear import LinearGaussian
measurement_model = LinearGaussian(
    ndim_state=4,
    mapping=(0, 2),
    noise_covar=np.array([[0.25, 0],
                          [0, 0.25]])
    )
all_measurements = []

for k in range(20):
    measurement_set = set()
    timestamp = start_time + timedelta(seconds=k)

    for truth in truths:
        try:
            truth_state = truth[timestamp]
        except IndexError:
            # This truth not alive at this time.
            continue
        # Generate actual detection from the state with a 10% chance that no detection is received.
        if np.random.rand() <= 0.9:
            # Generate actual detection from the state
            measurement = measurement_model.function(truth_state, noise=True)
            measurement_set.add(TrueDetection(state_vector=measurement,
                                              groundtruth_path=truth,
                                              timestamp=truth_state.timestamp))

            # Generate clutter at this time-step
            truth_x = truth_state.state_vector[0]
            truth_y = truth_state.state_vector[2]
            for _ in range(np.random.randint(2)):
                x = uniform.rvs(truth_x - 10, 20)
                y = uniform.rvs(truth_y - 10, 20)
                measurement_set.add(Clutter(np.array([[x], [y]]), timestamp=timestamp))
    all_measurements.append(measurement_set)

# Plot detections and clutter
for set_ in all_measurements:
    # Plot actual detections.
    ax.scatter([state.state_vector[0] for state in set_ if isinstance(state, TrueDetection)],
               [state.state_vector[1] for state in set_ if isinstance(state, TrueDetection)],
               color='g')
    # Plot clutter.
    ax.scatter([state.state_vector[0] for state in set_ if isinstance(state, Clutter)],
               [state.state_vector[1] for state in set_ if isinstance(state, Clutter)],
               color='y',
               marker='2')
fig

# %%
# Creating a Tracker
# ------------------
# We'll now create the tracker components as we did with the multi-target examples previously.

from stonesoup.predictor.kalman import KalmanPredictor
predictor = KalmanPredictor(transition_model)

from stonesoup.updater.kalman import KalmanUpdater
updater = KalmanUpdater(measurement_model)

from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.measures import Mahalanobis
hypothesiser = DistanceHypothesiser(predictor, updater, measure=Mahalanobis(), missed_distance=3)

from stonesoup.dataassociator.neighbour import GNNWith2DAssignment
data_associator = GNNWith2DAssignment(hypothesiser)

# %%
# Creating a Deleter
# ------------------
# Here we are going to create an error based deleter, which will delete any :class:`~.Track` where
# trace of the covariance is over a certain threshold i.e. when we have a high uncertainty. This
# simply requires a threshold to be defined, that will depend on units and number of dimensions of
# your state vector.
from stonesoup.deleter.error import CovarianceBasedDeleter
deleter = CovarianceBasedDeleter(4)

# %%
# Creating an Initiator
# ---------------------
# Here we are going to use a measurement based initiator, which will create a track from the
# unassociated :class:`~.Detection` objects. This still requires a prior to be defined for the
# :class:`~.Track`, but elements of the state vector that are measured are replaced by that of the
# measurement, and the measurement's uncertainty (these defined by the
# :class:`~.MeasurementModel`). In this example, as our sensor measures position, we only need to
# be concerned about the values provided for the velocity and it's variance.
#
# As we are dealing with clutter, here we are going to be using a multi-measurement initiator. This
# requires that multiple measurements are added to a track before being initiated. In this example,
# this initiator effectively runs a mini version of the same tracker, but you could use different
# components.
from stonesoup.types.state import GaussianState
from stonesoup.initiator.simple import MultiMeasurementInitiator
initiator = MultiMeasurementInitiator(
    prior_state=GaussianState([[0], [0], [0], [0]], np.diag([0, 1, 0, 1])),
    measurement_model=measurement_model,
    deleter=deleter,
    data_associator=data_associator,
    updater=updater,
    min_points=2,
    )

# %%
# Running the Tracker
# -------------------
# Loop through the predict, hypothesise, associate and update steps like before, but note on update
# which detections we've used at each time step. In each loop the deleter is called, returning
# tracks that are to be removed. Then the initiator is called with the unassociated detections, by
# taking the associated detections from the full set. The order of the deletion and initiation is
# important, so tracks that have just been created, aren't deleted straight away. (The
# implementation below is the same as :class:`~.MultiTargetTracker`)

tracks = set()

for n, measurements in enumerate(all_measurements):
    # Calculate all hypothesis pairs and associate the elements in the best subset to the tracks.
    hypotheses = data_associator.associate(tracks,
                                           measurements,
                                           start_time + timedelta(seconds=n))
    associated_measurements = set()
    for track in tracks:
        hypothesis = hypotheses[track]
        if hypothesis.measurement:
            post = updater.update(hypothesis)
            track.append(post)
            associated_measurements.add(hypothesis.measurement)
        else:  # When data associator says no detections are good enough, we'll keep the prediction
            track.append(hypothesis.prediction)

    # Carry out deletion and initiation
    tracks -= deleter.delete_tracks(tracks)
    tracks |= initiator.initiate(measurements - associated_measurements)

# %%
# Plot the resulting tracks.

tracks_list = list(tracks)
for track in tracks:
    # Plot track.
    ax.plot([state.state_vector[0, 0] for state in track],
            [state.state_vector[2, 0] for state in track],
            marker=".")
fig

# %%
# Plotting ellipses representing the gaussian estimate state at each update.

from matplotlib.patches import Ellipse
for track in tracks:
    for state in track[1:]:  # Skip the prior
        w, v = np.linalg.eig(measurement_model.matrix()@state.covar@measurement_model.matrix().T)
        max_ind = np.argmax(v[0, :])
        orient = np.arctan2(v[max_ind, 1], v[max_ind, 0])
        ellipse = Ellipse(xy=state.state_vector[(0, 2), 0],
                          width=np.sqrt(w[0])*2,
                          height=np.sqrt(w[1])*2,
                          angle=np.rad2deg(orient),
                          alpha=0.2)
        ax.add_artist(ellipse)
fig

# sphinx_gallery_thumbnail_number = 4
