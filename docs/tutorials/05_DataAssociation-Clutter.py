#!/usr/bin/env python
# coding: utf-8

"""
5 - Data Association - Clutter tutorial
=======================================
"""

# %%
# **Tracking a single target in a sea of clutter and missed detections:**
#
# In the process of tracking a target, we might receive a collection of incoming measurements from
# our sensor at any single point in time, some of which not originating from the target we are
# tracking. In this tutorial we explore a method to determine which measurement in this *clutter*
# we should take to update our prediction with. In particular, we use the **nearest neighbour**
# data association method, which greedily picks a detection that falls 'closest' to where we
# predict an incoming measurement should be, according to a particular metric.

# %%
# As in previous tutorials, we start with a target moving linearly in the 2D cartesian plane.

from datetime import datetime
from datetime import timedelta

from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
                                               ConstantVelocity
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState

start_time = datetime.now()
transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.005),
                                                          ConstantVelocity(0.005)])
truth = GroundTruthPath([GroundTruthState([0, 1, 0, 1], timestamp=start_time)])
for k in range(1, 21):
    truth.append(GroundTruthState(
        transition_model.function(truth[k-1], noise=True, time_interval=timedelta(seconds=1)),
        timestamp=start_time+timedelta(seconds=k)))

# %%
# And create the Kalman predictor.

from stonesoup.predictor.kalman import KalmanPredictor
predictor = KalmanPredictor(transition_model)

# %%
# Simulating clutter
# ------------------
# Next, generate some measurements with uniform probability of detection, and add in some clutter
# at each time-step. We use the :class:`~.TrueDetection` and :class:`~.Clutter` subclasses of
# :class:`~.Detection` to help with discerning data types later in plotting. We'll introduce the
# possibility that, at any time-step, our sensor receives no true detection from the target.

import numpy as np

from scipy.stats import uniform

from stonesoup.types.detection import TrueDetection
from stonesoup.types.detection import Clutter
from stonesoup.models.measurement.linear import LinearGaussian
measurement_model = LinearGaussian(
    ndim_state=4,
    mapping=(0, 2),
    noise_covar=np.array([[0.75, 0],
                          [0, 0.75]])
    )
all_measurements = []
for state in truth:
    measurement_set = set()

    # Generate actual detection from the state with a 10% chance that no detection is received.
    if np.random.rand() <= 0.9:
        measurement = measurement_model.function(state, noise=True)
        measurement_set.add(TrueDetection(state_vector=measurement,
                                          groundtruth_path=truth,
                                          timestamp=state.timestamp))

    # Generate clutter at this time-step
    truth_x = state.state_vector[0]
    truth_y = state.state_vector[2]
    for _ in range(np.random.randint(10)):
        x = uniform.rvs(truth_x - 10, 20)
        y = uniform.rvs(truth_y - 10, 20)
        measurement_set.add(Clutter(np.array([[x], [y]]), timestamp=state.timestamp))

    all_measurements.append(measurement_set)

# %%
# And create the Kalman updater.

from stonesoup.updater.kalman import KalmanUpdater
updater = KalmanUpdater(measurement_model)

# %%
# Plot the ground truth and measurements with clutter.

from matplotlib import pyplot as plt
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_ylim(0, 25)
ax.set_xlim(0, 25)

ax.plot([state.state_vector[0] for state in truth],
        [state.state_vector[2] for state in truth],
        linestyle="--",)

for set_ in all_measurements:
    # Plot actual detections.
    ax.scatter([state.state_vector[0] for state in set_ if isinstance(state, TrueDetection)],
               [state.state_vector[1] for state in set_ if isinstance(state, TrueDetection)],
               color='b')
    # Plot clutter.
    ax.scatter([state.state_vector[0] for state in set_ if isinstance(state, Clutter)],
               [state.state_vector[1] for state in set_ if isinstance(state, Clutter)],
               color='y',
               marker='2')
fig

# %%
# Distance Hypothesiser and Nearest Neighbour
# -------------------------------------------
# At each time-step, which measurement should our tracker pick-up and update with?
# The simplest method would be to rank each detection at a time-step, and pick the highest ranking
# to contribute to the track.
#
# An appropriate metric to rank points around a state described by Gaussian distributions would be
# the *Mahalanobis distance*. This metric measures the distance of a point relative to a given
# distribution.
# In the case of a point :math:`\vec{x} = (x_{1}, ..., x_{N})^T`, and distribution with mean
# :math:`\vec{\mu} = (\mu_{1}, ..., \mu_{N})^T` and covariance matrix :math:`P`, the Mahalanobis
# distance of :math:`\vec{x}` from the distribution is given by:
#
# .. math::
#       \sqrt{(\vec{x} - \vec{\mu})^T P^{-1} (\vec{x} - \vec{\mu})}
#
# It is the multi-dimensional measure of how many standard deviations a point is away from a
# distribution's mean.
#
# The :class:`~.DistanceHypothesiser` pairs incoming detections with track predictions, and scores
# each according to the detection's 'distance' (according to a given :class:`Measure` class) from
# the prediction.

# %%
# Create a hypothesiser that ranks detections against predicted measurement according to the
# Mahalanobis distance, where those that fall outside of :math:`3` standard deviations of the
# predicted state's mean are ignored.
# The hypothesiser must use a predicted state given by the predictor, create a measurement
# prediction using the updater, and compare this to a detection given a specific metric. Hence, it
# takes the predictor, updater, measure (metric) and missed distance as its arguments.

from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.measures import Mahalanobis
hypothesiser = DistanceHypothesiser(predictor, updater, measure=Mahalanobis(), missed_distance=3)

# %%
# Now we use the :class:`~.NearestNeighbour` data associator, which picks the hypothesis pair
# (prediction and detection) with the highest 'score' (in this instance, those that are 'closest'
# to each other).
#
# .. image:: ../_static/NN_Association_Diagram.png
#   :width: 500
#   :alt: Image showing NN association for one track
#
# In the diagram above, there are three possible detections to be considered for association (some
# of which may be clutter). The detection with a score of :math:`0.4` is selected by the nearest
# neighbour algorithm.

from stonesoup.dataassociator.neighbour import NearestNeighbour
data_associator = NearestNeighbour(hypothesiser)

# %%
# Running the Kalman Filter
# -------------------------
# With these components, we can run the simulated data and clutter through the Kalman filter.

# Create prior
from stonesoup.types.state import GaussianState
prior = GaussianState([[0], [1], [0], [1]], np.diag([1.5, 0.5, 1.5, 0.5]), timestamp=start_time)

# %%
# Loop through the predict, hypothesise, associate and update steps.

from stonesoup.types.track import Track

track = Track([prior])
for n, measurements in enumerate(all_measurements):
    hypotheses = data_associator.associate([track],
                                           measurements,
                                           start_time + timedelta(seconds=n))
    hypothesis = hypotheses[track]

    if hypothesis.measurement:
        post = updater.update(hypothesis)
        track.append(post)
    else:  # When data associator says no detections are good enough, we'll keep the prediction
        track.append(hypothesis.prediction)

# %%
# Plot the resulting track

ax.plot([state.state_vector[0, 0] for state in track[1:]],  # Skip plotting the prior
        [state.state_vector[2, 0] for state in track[1:]],
        marker=".")
fig

# %%
# Plotting ellipses representing the gaussian estimate state at each update.

from matplotlib.patches import Ellipse
for state in track[1:]:  # Skip the prior
    w, v = np.linalg.eig(measurement_model.matrix()@state.covar@measurement_model.matrix().T)
    max_ind = np.argmax(v[0, :])
    orient = np.arctan2(v[max_ind, 1], v[max_ind, 0])
    ellipse = Ellipse(xy=state.state_vector[(0, 2), 0],
                      width=np.sqrt(w[0])*2, height=np.sqrt(w[1])*2,
                      angle=np.rad2deg(orient),
                      alpha=0.2)
    ax.add_artist(ellipse)
fig

# sphinx_gallery_thumbnail_number = 2

# %%
# It may be the case that the track drifts off (sometimes quite far) from the ground truth path.
# There are issues with using a 'greedy' method of association such as the nearest neighbour
# algorithm.
