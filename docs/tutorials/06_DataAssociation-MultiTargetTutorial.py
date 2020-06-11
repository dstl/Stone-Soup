#!/usr/bin/env python
# coding: utf-8

"""
6 - Data Association - Multi-Target Tracking Tutorial
=====================================================
"""

# %%
# **Tracking multiple targets through clutter:**
#
# When tracking multiple targets it may be difficult to ascertain the source of an incoming
# measurement, especially with surrounding clutter.
# In this tutorial we introduce the **global nearest neighbour** data association technique, which
# attempts to find a collection of hypotheses such that their overall probabilities are maximised.

# %%
# Simulating 2 targets
# --------------------
# We start by simulating 2 targets transiting across the 2D cartesian plane.
# They start at (0, 0) and (0, 20), and will intersect roughly half-way through their transit.

from datetime import datetime
from datetime import timedelta

from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
                                               ConstantVelocity
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState

start_time = datetime.now()
truths = set()

transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.005),
                                                          ConstantVelocity(0.005)])

truth = GroundTruthPath([GroundTruthState([0, 1, 0, 1], timestamp=start_time)])
for k in range(1, 21):
    truth.append(GroundTruthState(
        transition_model.function(truth[k-1], noise=True, time_interval=timedelta(seconds=1)),
        timestamp=start_time+timedelta(seconds=k)))
truths.add(truth)

truth = GroundTruthPath([GroundTruthState([0, 1, 20, -1], timestamp=start_time)])
for k in range(1, 21):
    truth.append(GroundTruthState(
        transition_model.function(truth[k-1], noise=True, time_interval=timedelta(seconds=1)),
        timestamp=start_time+timedelta(seconds=k)))
truths.add(truth)

from matplotlib import pyplot as plt

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_ylim(0, 25)
ax.set_xlim(0, 25)

for truth in truths:
    ax.plot([state.state_vector[0] for state in truth],
            [state.state_vector[2] for state in truth],
            linestyle="--",)
fig

# %%
# And create the Kalman predictor.

from stonesoup.predictor.kalman import KalmanPredictor
predictor = KalmanPredictor(transition_model)

# %%
# Generate Clutter
# ----------------
# Next, generate detections with clutter just as in the previous tutorial. This time, we generate
# clutter about each state at each time-step.

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

for k in range(20):
    measurement_set = set()

    for truth in truths:
        # Generate actual detection from the state with a 10% chance that no detection is received.
        if np.random.rand() <= 0.9:
            measurement = measurement_model.function(truth[k], noise=True)
            measurement_set.add(TrueDetection(state_vector=measurement,
                                              groundtruth_path=truth,
                                              timestamp=truth[k].timestamp))

        # Generate clutter at this time-step
        truth_x = truth[k].state_vector[0]
        truth_y = truth[k].state_vector[2]
        for _ in range(np.random.randint(10)):
            x = uniform.rvs(truth_x - 10, 20)
            y = uniform.rvs(truth_y - 10, 20)
            measurement_set.add(Clutter(np.array([[x], [y]]), timestamp=truth[k].timestamp))
    all_measurements.append(measurement_set)

# Plot clutter.
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
# And create the Kalman updater.

from stonesoup.updater.kalman import KalmanUpdater
updater = KalmanUpdater(measurement_model)

# %%
# As in the 'clutter' tutorial, we will rank measurements using the Mahalanobis distance.

from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.measures import Mahalanobis
hypothesiser = DistanceHypothesiser(predictor, updater, measure=Mahalanobis(), missed_distance=3)

# %%
# Global Nearest Neighbour
# ------------------------
# With multiple targets to track, the nearest neighbour algorithm compiles a list of all hypotheses
# and selects pairings from highest 'score' first.
#
# .. image:: ../_static/NN_Association_MultiTarget_Diagram.png
#   :width: 500
#   :alt: Image showing NN association of two tracks
#
# As shown in the diagram above, the higher marker is selected for association with the blue track,
# as this has the highest 'score' of :math:`0.5`, and (as each measurement is associated at most
# once) the remaining marker must then be associated with the orange track, giving a net global
# score of :math:`0.51`.
#
# The :class:`~.GlobalNearestNeighbour` evaluates all possible (distance-based) hypotheses
# (measurement-prediction pairs), removes those that are invalid, and selects the subset with the
# greatest net 'score' (the collection of hypotheses pairs which have a minimum sum of distances
# overall).
#
# .. image:: ../_static/GNN_Association_Diagram.png
#   :width: 500
#   :alt: Image showing GNN association of two tracks
#
# In the diagram above, the blue track is associated to the lower marker even though the higher
# one is ranked better relative to it. This association would lead to a global score of
# :math:`0.51`. However, the alternate associations would lead to a global score of :math:`0.6` - a
# better net score of hypotheses.

from stonesoup.dataassociator.neighbour import GlobalNearestNeighbour
data_associator = GlobalNearestNeighbour(hypothesiser)

# %%
# Running the Kalman filter
# -------------------------
# We create 2 priors as the targets have different initial y-velocities.

from stonesoup.types.state import GaussianState
prior1 = GaussianState([[0], [1], [0], [1]], np.diag([1.5, 0.5, 1.5, 0.5]), timestamp=start_time)
prior2 = GaussianState([[0], [1], [20], [-1]], np.diag([1.5, 0.5, 1.5, 0.5]), timestamp=start_time)

# %%
# Loop through the predict, hypothesise, associate and update steps.

from stonesoup.types.track import Track
tracks = {Track([prior1]), Track([prior2])}

for n, measurements in enumerate(all_measurements):
    # Calculate all hypothesis pairs and associate the elements in the best subset to the tracks.
    hypotheses = data_associator.associate(tracks,
                                           measurements,
                                           start_time + timedelta(seconds=n))
    for track in tracks:
        hypothesis = hypotheses[track]
        if hypothesis.measurement:
            post = updater.update(hypothesis)
            track.append(post)
        else:  # When data associator says no detections are good enough, we'll keep the prediction
            track.append(hypothesis.prediction)

# %%
# Plot the resulting tracks.

tracks_list = list(tracks)
for track in tracks:
    # Plot track.
    ax.plot([state.state_vector[0, 0] for state in track[1:]],  # Skip plotting the prior
            [state.state_vector[2, 0] for state in track[1:]],
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
