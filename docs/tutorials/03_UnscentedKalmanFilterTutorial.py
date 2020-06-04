#!/usr/bin/env python
# coding: utf-8

"""
3 - Unscented Kalman filter tutorial
======================
"""

# %%
# **Handling divergence and sub-optimal performance with non-linear system tracking:**
#
# The Extended Kalman filter propogates covariance matrices through linearisations of non-linear
# models. In highly non-linear systems, this can lead to over-estimated covariances or divergence
# of tracks, to the point where uncertainties make our estimates useless.
#
# We take a deterministic sampling approach to the problem.
# Using the *unscented transformation* technique, we characterize a relevant probability
# distributions with a finite set of statistics. In the case of Kalman filtering that would concern
# the mean and covariance of Gaussian states.  To attain this characterization, we create a minimal
# selection of sample (sigma) points, distributed about the mean with corresponding weights, which
# determine their relevance/contribution to the state mean and covariance.
# As before, the state distribution is approximated as Gaussian. But now we create a minimal
# selection of sample (sigma) points that can fully describe the distribution.
# By distributing in the state space with appropriate weights, we attain a collection of discrete
# point, mean weight and covariance weight triplets that capture the mean and covariance of the
# underlying distribution.
#
# We can then propogate these points through the non-linear system, and define a new distribution
# determined by their new layout.
# Without proof, the posterior mean and covariance will be accurate to the 3rd order Taylor
# expansion for all non-linearities that need approximation.

# %%
# For a simple gaussian state near the origin we can create a set of sigma points around its centre
# that will fully describe its mean and variance:

# Some general imports.
from datetime import datetime

import numpy as np

from stonesoup.functions import gauss2sigma
from stonesoup.types.state import GaussianState

# Make example gaussian state (high uncertainty in x, low uncertainty in y).
some_gaussian_state = GaussianState([[2.5], [1], [1.5], [0]], np.diag([5, 1, 0.4, 1]),
                                    timestamp=datetime.now())

# %%
# Create sigma points. 'alpha' defines the point spread.
sigma_points, sigma_weights, sigma_covars = gauss2sigma(some_gaussian_state, alpha=0.3)

# %%
# Set-up for plotting.
from matplotlib import pyplot as plt
fig1 = plt.figure(figsize=(10, 6))
ax = fig1.add_subplot(1, 1, 1)
ax.set_ylim(0, 3)
ax.set_xlim(0, 5)

# Plot gaussian distribution (for positional coordinates of the state (remember that state space
# also has velocity coordinates) to one standard deviation).
from matplotlib.patches import Ellipse
w, v = np.linalg.eig(some_gaussian_state.covar)
max_ind = np.argmax(v[0, :])
orient = np.arctan2(v[max_ind, 1], v[max_ind, 0])
ellipse = Ellipse(xy=(some_gaussian_state.state_vector[0],  # x-coord
                      some_gaussian_state.state_vector[2]),  # y-coord
                  width=np.sqrt(w[0])*2, height=np.sqrt(w[2])*2,
                  angle=np.rad2deg(orient),
                  alpha=0.2,
                  color='b')
ax.add_artist(ellipse)

# Plot sigma points that describe this distribution (ellipse).
x = [sigma.state_vector[0, 0] for sigma in sigma_points]
y = [sigma.state_vector[2, 0] for sigma in sigma_points]
ax.scatter(x, y, color='r', s=3)

# %%
# Where the individual sigma point weights are given by:
sigma_weights

# %%
# We might be given data from a sensor that provides accurate range, but incredibly bad bearing
# measurements of targets.

# %%
# Set up a prediction state for the sensor to create a predicted measurement from.

from stonesoup.types.prediction import GaussianStatePrediction
# Make prediction state that we will use to make our measurement predictions from.
prediction = GaussianStatePrediction(state_vector=[[20], [1], [20], [1]],
                                     covar=np.diag([5, 1, 5, 1]),
                                     timestamp=datetime.now())

from stonesoup.functions import cart2pol
psigmas, pweights, pcovars = gauss2sigma(prediction, alpha=0.3)
rthetas = [cart2pol(sigma.state_vector[0, 0], sigma.state_vector[2, 0]) for sigma in psigmas]
rs = [coord[0] for coord in rthetas]
thetas = [coord[1] for coord in rthetas]

# %%
# Create terrible bearing, range sensor.

from stonesoup.models.measurement.nonlinear import CartesianToBearingRange

sensor_x = 0
sensor_y = 0

measurement_model = CartesianToBearingRange(
    ndim_state=4,
    mapping=(0, 2),
    noise_covar=np.diag([np.radians(10), 0.001]),  # bad bearing, good range uncertainties
    translation_offset=np.array([[sensor_x], [sensor_y]])
)

# %%
# The Extended Kalman Filter would certainly handle the non-linearity of the mapping
# :math:`cartesian\mapsto \polar`, but would fall short (or in fact too far) in its prediction.
# Our sensor has a large bearing uncertainty. Consider the shape of this. We essentially have a
# banana-shaped region of where we might find our measurement: A curve of possible bearings, and
# a width of ranges.
# The Unscented Kalman Filter will have sampled (made some sigma points) in the state space
# (cartesian), and converted them to measurement space (polar) to form a new distribution.
# However, the EKF will have simply converted the state distribution's mean to measurement space,
# and approximated a covariance by taking the 1st order linearisation of the mapping from cartesian
# to polar.
# This results in the UKF giving a distribution that is 'shifted' back slightly from the EKF's,
# better describing the banana distribution we would expect, and highlighting its advantage over
# the EKF.

# Create updaters
from stonesoup.updater.kalman import UnscentedKalmanUpdater, ExtendedKalmanUpdater
unscented_updater = UnscentedKalmanUpdater(measurement_model, alpha=0.3)
extended_updater = ExtendedKalmanUpdater(measurement_model)

# Get predicted measurements from the state prediction.
ukf_pred_meas = unscented_updater.predict_measurement(prediction)
ekf_pred_meas = extended_updater.predict_measurement(prediction)

from matplotlib import pyplot as plt

fig2 = plt.figure(figsize=(10, 6))
ax = fig2.add_subplot(1, 1, 1, polar=True)
ax.set_ylim(0, 40)
ax.set_xlim(0, np.radians(90))

# Plot the sigma points of the state after being converted to measurement space (granted, we are
# actually still plotting in cartesian space here, just with range/bearing lines superimposed) that
# are used by the UKF.
ax.scatter([thetas], [rs], color='b', s=3)

# Plot UKF's predicted measurement distribution (red)
w, v = np.linalg.eig(ukf_pred_meas.covar)
max_ind = np.argmax(v[0, :])
orient = np.arctan2(v[max_ind, 1], v[max_ind, 0])
ukf_ellipse = Ellipse(xy=(ukf_pred_meas.state_vector[0], ukf_pred_meas.state_vector[1]),
                      width=np.sqrt(w[0])*2, height=np.sqrt(w[1])*2,
                      angle=np.rad2deg(orient),
                      alpha=0.2,
                      color='r')
ax.add_artist(ukf_ellipse)

# Plot EKF's predicted measurement distribution (green)
w, v = np.linalg.eig(ekf_pred_meas.covar)
max_ind = np.argmax(v[0, :])
orient = np.arctan2(v[max_ind, 1], v[max_ind, 0])
ekf_ellipse = Ellipse(xy=(ekf_pred_meas.state_vector[0], ekf_pred_meas.state_vector[1]),
                      width=np.sqrt(w[0])*2, height=np.sqrt(w[1])*2,
                      angle=np.rad2deg(orient),
                      alpha=0.2,
                      color='g')
ax.add_artist(ekf_ellipse)

# %%
# If we were to continue with applying the EKF, it is clear that deviation would be inevitable.
# The clear shift of the UKF's distribution (red) closer to the sensor shows its better
# approximation of the banana-shaped distribution that we would expect.

# %%
# Running the Unscented Kalman Filter
# -----------------------------------
# We will do a quick run of the UKF on a very noisy (uncertain) sensor that gives great range, but
# awful bearing readings.

# %%
# Set-up plot and ground truth as before.
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot()
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.axis('equal')

from datetime import timedelta
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
                                               ConstantVelocity

start_time = datetime.now()
transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.05),
                                                          ConstantVelocity(0.05)])

truth = GroundTruthPath([GroundTruthState([0, 1, 0, 1], timestamp=start_time)])

for k in range(1, 21):
    truth.append(GroundTruthState(
        transition_model.function(truth[k-1], noise=True, time_interval=timedelta(seconds=1)),
        timestamp=start_time+timedelta(seconds=k)))

# Plot the result
ax.plot([state.state_vector[0] for state in truth],
        [state.state_vector[2] for state in truth],
        linestyle="--")

# %%
# Create Kalman predictor
# -----------------------
# The transition of the target state is linear, so we have no need for a
# :class:`~.ExtendedKalmanPredictor` or :class:`~.UnscentedKalmanPredictor`.

from stonesoup.predictor.kalman import KalmanPredictor
predictor = KalmanPredictor(transition_model)

# %%
# Simulate noisy measuring
# ------------------------

from stonesoup.models.measurement.nonlinear import CartesianToBearingRange

sensor_x = 0
sensor_y = 0

# Make noisy measurement.
measurement_model = CartesianToBearingRange(
    ndim_state=4,
    mapping=(0, 2),
    noise_covar=np.diag([np.radians(3), 0.001]),  # bearing variance = 3 degrees
    translation_offset=np.array([[sensor_x], [sensor_y]])
)

# %%
from stonesoup.types.detection import Detection
from stonesoup.functions import pol2cart

# Make sensor that produces the noisy measurements.
measurements = []
for state in truth:
    measurement = measurement_model.function(state, noise=True)
    measurements.append(Detection(measurement, timestamp=state.timestamp))

# Plot the measurements (turning them back in to cartesian coordinates (for the sake of a nice
# plot)).
x, y = pol2cart(
    np.hstack([state.state_vector[1, 0] for state in measurements]),
    np.hstack([state.state_vector[0, 0] for state in measurements]))
ax.scatter(x + sensor_x, y + sensor_y, color='b')
fig

# %%
# Running the Unscented Kalman Filter
# -----------------------------------

# Create :class:`~.UnscentedKalmanUpdater`.

from stonesoup.updater.kalman import UnscentedKalmanUpdater
unscented_updater = UnscentedKalmanUpdater(measurement_model)  # Keep alpha as default = 0.5

prior = GaussianState([[0], [1], [0], [1]], np.diag([1, 1, 1, 1]), timestamp=start_time)

# %%

# Plot UKF track (red)
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.track import Track

track = Track()
for measurement in measurements:
    prediction = predictor.predict(prior, timestamp=measurement.timestamp)
    hypothesis = SingleHypothesis(prediction, measurement)
    post = unscented_updater.update(hypothesis)
    track.append(post)
    prior = track[-1]

ax.plot([state.state_vector[0, 0] for state in track],
        [state.state_vector[2, 0] for state in track],
        marker=".", color='r')

# Plot UKF errors (red)
HH = np.array([[1.,  0.,  0.,  0.],
               [0.,  0.,  1.,  0.]])
for state in track:
    w, v = np.linalg.eig(HH@state.covar@HH.T)
    max_ind = np.argmax(v[0, :])
    orient = np.arctan2(v[max_ind, 1], v[max_ind, 0])
    ellipse = Ellipse(xy=(state.state_vector[0], state.state_vector[2]),
                      width=np.sqrt(w[0])*2, height=np.sqrt(w[1])*2,
                      angle=np.rad2deg(orient),
                      alpha=0.2,
                      color='r')
    ax.add_artist(ellipse)
fig

# sphinx_gallery_thumbnail_number = 2

# %%
# Although a bit manic, it is clear that the UKF better represents well the uncertainty of the
# sensor throughout the filtering process.
