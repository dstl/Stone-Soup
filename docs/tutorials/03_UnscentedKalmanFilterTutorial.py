#!/usr/bin/env python
# coding: utf-8

"""
3 - Non-linear models: unscented Kalman filter tutorial
=======================================================
"""

# %%
# The previous tutorial showed how the extended Kalman filter propagates estimates using a
# first-order linearisation of the transition and/or sensor models. Clearly there are limits to
# such an approximation, and in situtations where models deviate significantly from linearity,
# performance can suffer.
#
# In such situations it can be beneficial to seek alternative approximations. One such comes via
# the so-called *unscented transform* (UT). In this we charaterise a Gaussian distribution using a series of
# weighted samples, *sigma points*, and propagate these through the non-linear function. A transformed
# Gaussian is then reconstructed from the new sigma points. This forms the basis for the unscented
# Kalman filter (UKF).
#
# <Without proof, the posterior mean and covariance will be accurate to the 3rd order Taylor
# expansion for all non-linearities that need approximation.> [Reference?]
#
# This tutorial will first run a simulation in an entirely equivalent fashion to the previous
# (EKF) tutorial. We'll then look into more precise details concerning the UT and try and develop
# some intuition into the reasons for its effectiveness.

# %%
# Background
# ----------
# There's a fair amount of theory on how the UT and the UKF work; see for example [1,2,3]. We
# restrict ourselves here to how Stone Soup does things.
#
# For dimension :math:`N_d`, a set of :math:`2 N_d + 1` sigma points are calculated at:
#
# .. math::
#           \mathbf{s}_j = \mathbf{x}, \ \ j = 0
#
#           \mathbf{s}_j = \mathbf{x} + \alpha \sqrt{\kappa} A_j, \ \ j = 1, ..., N_d
#
#           \mathbf{s}_j = \mathbf{x} - \alpha \sqrt{\kappa} A_j, \ \ j = N_d + 1, ..., 2 N_d
#
# where :math:`A_j` is the :math:`j` th column of :math:`A`, a *square root matrix* of the covariance,
# :math:`P = AA^T`, of the state to be approximated, and :math:`\mathbf{x}` is its mean.
#
# Two sets of weights, mean and covariance, are calculated:
#
# .. math::
#           W^m_0 = \frac{\lambda}{c}
#
#           W^c_0 = \frac{\lambda}{c} + (1 - \alpha^2 + \beta)
#
#           W^m_j = W^c_j = \frac{1}{2 c}
#
# where :math:`\lambda = \alpha^2 (N_d + \kappa) - N_d`, :math:`c = N_d + \lambda`. The parameters
# :math:`\alpha, \ \beta, \ \kappa` are user-selectable parameters with default values of
# :math:`1, \ 2, \ 3 - N_d`.
#
# After the sigma points are transformed :math:`\mathbf{s^{\prime}} = f( \mathbf{s} )`, the
# distribution is reconstructed as:
#
# .. math::
#           \mathbf{x}^\prime = \sum\limits^{2 N_d}_{0} W^{m}_j \mathbf{s}^{\prime}_j
#
#           P = (\mathbf{s}^{\prime} - \mathbf{x}^\prime) \, diag(W^c) \, (\mathbf{s}^{\prime} -
#           \mathbf{x}^\prime)^T + Q
#

# %%
# A simple example
# ----------------
# This example is equivalent to that in the previous (EKF) tutorial. As with that one, you are invited
# to play with the parameters and watch what happens.
#
# Some general imports and initialise time
import numpy as np
from matplotlib import pyplot as plt  # Set-up for plotting

from datetime import datetime, timedelta
start_time = datetime.now()

# %%
# Create ground truth
# ^^^^^^^^^^^^^^^^^^^
#
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
                                               ConstantVelocity

transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.05),
                                                          ConstantVelocity(0.05)])
truth = GroundTruthPath([GroundTruthState([0, 1, 0, 1], timestamp=start_time)])

for k in range(1, 21):
    truth.append(GroundTruthState(
        transition_model.function(truth[k-1], noise=True, time_interval=timedelta(seconds=1)),
        timestamp=start_time+timedelta(seconds=k)))

# %%
# Set-up plot to render ground truth, as before.
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot()
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.axis('equal')

# Plot the result
ax.plot([state.state_vector[0] for state in truth],
        [state.state_vector[2] for state in truth],
        linestyle="--")

# %%
# Simulate the measurement
# ^^^^^^^^^^^^^^^^^^^^^^^^
#
from stonesoup.models.measurement.nonlinear import CartesianToBearingRange
# Sensor position
sensor_x = 50
sensor_y = 0

# Make noisy measurement.
measurement_model = CartesianToBearingRange(
    ndim_state=4,
    mapping=(0, 2),
    noise_covar=np.diag([np.radians(0.2), 1]),  # bearing variance = 3 degrees
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
# Create unscented Kalman filter components
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Note that the transition of the target state is linear, so we have no real need for a
# :class:`~.UnscentedKalmanPredictor`. But we'll use one anyway, if nothing else to demonstrate
# that a linear model won't break anything.
from stonesoup.predictor.kalman import UnscentedKalmanPredictor
predictor = UnscentedKalmanPredictor(transition_model)
# Create :class:`~.UnscentedKalmanUpdater`
from stonesoup.updater.kalman import UnscentedKalmanUpdater
unscented_updater = UnscentedKalmanUpdater(measurement_model)  # Keep alpha as default = 0.5

# %%
# Run the Unscented Kalman Filter
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Create a prior
from stonesoup.types.state import GaussianState
prior = GaussianState([[0], [1], [0], [1]], np.diag([1.5, 0.5, 1.5, 0.5]), timestamp=start_time)

# %%
# Populate the track
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.track import Track

track = Track()
for measurement in measurements:
    prediction = predictor.predict(prior, timestamp=measurement.timestamp)
    hypothesis = SingleHypothesis(prediction, measurement)
    post = unscented_updater.update(hypothesis)
    track.append(post)
    prior = track[-1]

# %%
# And do the plot, in red
ax.plot([state.state_vector[0, 0] for state in track],
        [state.state_vector[2, 0] for state in track],
        marker=".", color='r')

# Plot UKF errors (red)
from matplotlib.patches import Ellipse
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

# %%
# An look in more depth
# ------------------------
# In an attempt to build some intuition we can look a little deeper at the UT. Let's do this by considering
# the bearing-range sensor used above. We'll create a new 'prior'.
#
new_prior_state = GaussianState([[np.radians(90)], [10]], np.diag([0.1, 1]),
                                    timestamp=datetime.now())
# %%
# Create sigma points. 'alpha' (defines the point spread) is somewhat non-standard
from stonesoup.functions import gauss2sigma
sigma_points, sigma_weights, sigma_covars = gauss2sigma(new_prior_state, alpha=0.3)


# %%
# Plot this...
fig2 = plt.figure(figsize=(10, 6), tight_layout=True)
ax = fig2.add_subplot(1, 1, 1, polar=True)
ax.set_ylim(0, 25)
ax.set_xlim(0, np.radians(180))

# Plot gaussian distribution (for positional coordinates of the state (remember that state space
# also has velocity coordinates) to one standard deviation).
w, v = np.linalg.eig(new_prior_state.covar)
max_ind = np.argmax(v[0, :])
orient = np.arctan2(v[max_ind, 1], v[max_ind, 0])
ellipse = Ellipse(xy=(new_prior_state.state_vector[0],  # x-coord
                      new_prior_state.state_vector[1]),  # y-coord
                  width=np.sqrt(w[0])*2, height=np.sqrt(w[1])*2,
                  angle=np.rad2deg(orient),
                  alpha=0.2,
                  color='b')
ax.add_artist(ellipse)








# %%
# Set-up for plotting.
fig1 = plt.figure(figsize=(10, 6))
ax = fig1.add_subplot(1, 1, 1)
ax.set_ylim(0, 3)
ax.set_xlim(0, 5)

# Plot gaussian distribution (for positional coordinates of the state (remember that state space
# also has velocity coordinates) to one standard deviation).
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
prediction = GaussianStatePrediction(state_vector=[[0], [1], [20], [1]],
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
# :math:`cartesian\mapsto polar`, but would fall short (or in fact too far) in its prediction.
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

fig2 = plt.figure(figsize=(10, 6), tight_layout=True)
ax = fig2.add_subplot(1, 1, 1, polar=True)
ax.set_ylim(0, 25)
ax.set_xlim(0, np.radians(180))

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
# References
# ----------
#
# 1. Julier & Uhlman
#

# sphinx_gallery_thumbnail_number = 2

