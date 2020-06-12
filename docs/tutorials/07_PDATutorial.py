#!/usr/bin/env python
# coding: utf-8

"""
7 - Probabilistic Data Association Tutorial
===========================================
"""

# %%
# Associating a single hypothesis to a track has the disadvantage that posterior uncertainties
# can be underestimated.
# The probabilistic data associator overcomes this issue by:
#
# - Calculating a posterior for each hypothesis;
#
# .. image:: ../_static/PDA_Hypothesis_Diagram.png
#   :width: 500
#   :alt: Image showing NN association for one track
#
# - Weighting each posterior state according to the probability that its corresponding hypothesis
#   was true (and the prediction (null hypothesis) according to the missed-detection probability),
#   normalising such that the sum of the weights is 1;
#
# .. image:: ../_static/PDA_Weighting_Diagram.png
#   :width: 500
#   :alt: Image showing NN association for one track
#
# - Merging the resulting estimate states in to a single posterior Gaussian approximation(taking
#   the minimum mean squared error estimate).
#
# .. image:: ../_static/PDA_Merge_Diagram.png
#   :width: 500
#   :alt: Image showing NN association for one track
#
# This results in a better approximation to the posterior state covariances.

# %%
# Simulate ground truth.
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
# Simulate clutter.
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

prob_detect = 0.9  # 90% chance of detection.

all_measurements = []
for state in truth:
    measurement_set = set()

    # Generate detection.
    if np.random.rand() <= prob_detect:
        measurement = measurement_model.function(state, noise=True)
        measurement_set.add(TrueDetection(state_vector=measurement,
                                          groundtruth_path=truth,
                                          timestamp=state.timestamp))

    # Generate clutter.
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

# %%
# Create Probabilistic Data Associator
# ------------------------------------
# The :class:`~.PDAHypothesiser` and :class:`~.PDA` generate track predictions and calculate
# probabilities for all prediction-detection pairs for a single prediction and multiple detections.
from stonesoup.hypothesiser.probability import PDAHypothesiser
hypothesiser = PDAHypothesiser(predictor=predictor,
                               updater=updater,
                               clutter_spatial_density=0.125,
                               prob_detect=prob_detect)

from stonesoup.dataassociator.probability import PDA
data_associator = PDA(hypothesiser=hypothesiser)

# %%
# Running the PDA Filter
# ----------------------
# With these components, we can run the simulated data and clutter through the Kalman filter.

# Create prior
from stonesoup.types.state import GaussianState
prior = GaussianState([[0], [1], [0], [1]], np.diag([1.5, 0.5, 1.5, 0.5]), timestamp=start_time)

# Loop through the predict, hypothesise, associate and update steps.
from stonesoup.types.track import Track
from stonesoup.types.array import StateVectors  # For storing state vectors during association
from stonesoup.functions import gm_reduce_single  # For merging states to get posterior estimate
from stonesoup.types.update import GaussianStateUpdate  # To store posterior estimate

track = Track([prior])
for n, measurements in enumerate(all_measurements):
    hypotheses = data_associator.associate([track],
                                           measurements,
                                           start_time + timedelta(seconds=n))

    hypotheses = hypotheses[track]

    # Loop through each hypothesis, creating posterior states for each, and merge to calculate
    # approximation to actual posterior state mean and covariance.
    posterior_states = []
    posterior_state_weights = []
    for hypothesis in hypotheses:
        if not hypothesis:
            posterior_states.append(hypothesis.prediction)
        else:
            posterior_state = updater.update(hypothesis)
            posterior_states.append(posterior_state)
        posterior_state_weights.append(
            hypothesis.probability)

    means = StateVectors([state.state_vector for state in posterior_states])
    covars = np.stack([state.covar for state in posterior_states], axis=2)
    weights = np.asarray(posterior_state_weights)

    # Reduce mixture of states to one posterior estimate Gaussian.
    post_mean, post_covar = gm_reduce_single(means, covars, weights)

    # Add a Gaussian state approximation to the track.
    track.append(GaussianStateUpdate(
        post_mean, post_covar,
        hypotheses,
        hypotheses[0].measurement.timestamp))


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

# %%
# Joint Probabilistic Data Association
# ------------------------------------
# A joint probabilistic data association (JPDA) filter handles multi-target tracking through
# clutter. Similar to the PDA, the JPDA algorithm calculates hypothesis pairs for every measurement
# for every track. The weight of a track-measurement hypothesis is calculated by the normalised sum
# of conditional probabilities that every other track is associated to every other measurement
# (including missed detection). For example, with 3 tracks :math:`(A, B, C)` and 3 measurements
# :math:`(x, y, z)` (including missed detection :math:`None`), the probability of track :math:`A`
# being associated with measurement :math:`x` (:math:`A \to x`) is given by:
#
# .. math::
#       p(A \to x) &= \bar{p}(A \to x \cap B \to y \cap C \to z)\\
#                  &+ \bar{p}(A \to x \cap B \to z \cap C \to y) +\\
#                  &+ \bar{p}(A \to x \cap B \to None \cap C \to z) +\\
#                  &+ \bar{p}(A \to x \cap B \to None \cap C \to y) + ...
#
# where :math:`\bar{p}(multi-hypothesis)` is the normalised probability of the multi-hypothesis.

# %%

# Simulate ground truth.
truths = set()
truths.add(truth)

truth = GroundTruthPath([GroundTruthState([0, 1, 20, -1], timestamp=start_time)])
for k in range(1, 21):
    truth.append(GroundTruthState(
        transition_model.function(truth[k-1], noise=True, time_interval=timedelta(seconds=1)),
        timestamp=start_time+timedelta(seconds=k)))
truths.add(truth)

multi_fig = plt.figure(figsize=(10, 6))
axm = multi_fig.add_subplot(1, 1, 1)
axm.set_xlabel("$x$")
axm.set_ylabel("$y$")
axm.set_ylim(0, 25)
axm.set_xlim(0, 25)

# Plot ground truth.
for truth in truths:
    axm.plot([state.state_vector[0] for state in truth],
             [state.state_vector[2] for state in truth],
             linestyle="--",)

# Generate measurements.
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

# Plot measurements.
for set_ in all_measurements:
    # Plot actual detections.
    axm.scatter([state.state_vector[0] for state in set_ if isinstance(state, TrueDetection)],
                [state.state_vector[1] for state in set_ if isinstance(state, TrueDetection)],
                color='g')
    # Plot clutter.
    axm.scatter([state.state_vector[0] for state in set_ if isinstance(state, Clutter)],
                [state.state_vector[1] for state in set_ if isinstance(state, Clutter)],
                color='y',
                marker='2')

# %%
from stonesoup.predictor.kalman import KalmanPredictor
predictor = KalmanPredictor(transition_model)

# %%
from stonesoup.updater.kalman import KalmanUpdater
updater = KalmanUpdater(measurement_model)

# %%
# Initial hypotheses are calculated (per track) in the same manner as the PDA.
# Therefore, in Stone Soup, the JPDA filter uses the :class:`~.PDAHypothesiser` to create these
# hypotheses.
# The :class:`~.JPDA` assocaitor then adjusts hypothesis weights according to the method described
# above.
from stonesoup.hypothesiser.probability import PDAHypothesiser
# This doesn't need to be created again, but for the sake of visualising the process, it has been
# added.
hypothesiser = PDAHypothesiser(predictor=predictor,
                               updater=updater,
                               clutter_spatial_density=0.125,
                               prob_detect=prob_detect)

from stonesoup.dataassociator.probability import JPDA
data_associator = JPDA(hypothesiser=hypothesiser)

# %%
# Running the JPDA filter
# -----------------------

prior1 = GaussianState([[0], [1], [0], [1]], np.diag([1.5, 0.5, 1.5, 0.5]), timestamp=start_time)
prior2 = GaussianState([[0], [1], [20], [-1]], np.diag([1.5, 0.5, 1.5, 0.5]), timestamp=start_time)

tracks = {Track([prior1]), Track([prior2])}

for n, measurements in enumerate(all_measurements):
    hypotheses = data_associator.associate(tracks,
                                           measurements,
                                           start_time + timedelta(seconds=n))

    # Loop through each track, performing the association step with weights adjusted according to
    # JPDA.
    for track in tracks:
        track_hypotheses = hypotheses[track]

        posterior_states = []
        posterior_state_weights = []
        for hypothesis in track_hypotheses:
            if not hypothesis:
                posterior_states.append(hypothesis.prediction)
            else:
                posterior_state = updater.update(hypothesis)
                posterior_states.append(posterior_state)
            posterior_state_weights.append(hypothesis.probability)

        means = StateVectors([state.state_vector for state in posterior_states])
        covars = np.stack([state.covar for state in posterior_states], axis=2)
        weights = np.asarray(posterior_state_weights)

        # Reduce mixture of states to one posterior estimate Gaussian.
        post_mean, post_covar = gm_reduce_single(means, covars, weights)

        # Add a Gaussian state approximation to the track.
        track.append(GaussianStateUpdate(
            post_mean, post_covar,
            track_hypotheses,
            track_hypotheses[0].measurement.timestamp))

# %%
# Plot the resulting tracks.
tracks_list = list(tracks)
for track in tracks:
    # Plot track.
    axm.plot([state.state_vector[0, 0] for state in track[1:]],  # Skip plotting the prior
             [state.state_vector[2, 0] for state in track[1:]],
             marker=".")

# Plot ellipses representing the gaussian estimate state at each update.
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
        axm.add_artist(ellipse)
multi_fig

# sphinx_gallery_thumbnail_number = 5
