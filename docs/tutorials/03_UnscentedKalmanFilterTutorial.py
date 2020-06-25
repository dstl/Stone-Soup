#!/usr/bin/env python
# coding: utf-8

"""
=======================================================
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
# the so-called *unscented transform* (UT). In this we charaterise a Gaussian distribution using a
# series of weighted samples, *sigma points*, and propagate these through the non-linear function.
# A transformed Gaussian is then reconstructed from the new sigma points. This forms the basis for
# the unscented Kalman filter (UKF).
#
# This tutorial will first run a simulation in an entirely equivalent fashion to the previous
# (EKF) tutorial. We'll then look into more precise details concerning the UT and try and develop
# some intuition into the reasons for its effectiveness.

# %%
# Background
# ----------
# Limited detail on how Stone Soup does the UKF is provided below. See Julier et al. (2000) [#]_
# for fuller, better details of the UKF.
#
# For dimension :math:`N_d`, a set of :math:`2 N_d + 1` sigma points are calculated at:
#
# .. math::
#           \mathbf{s}_j &= \mathbf{x}, \ \ j = 0
#
#           \mathbf{s}_j &= \mathbf{x} + \alpha \sqrt{\kappa} A_j, \ \ j = 1, ..., N_d
#
#           \mathbf{s}_j &= \mathbf{x} - \alpha \sqrt{\kappa} A_j, \ \ j = N_d + 1, ..., 2 N_d
#
# where :math:`A_j` is the :math:`j` th column of :math:`A`, a *square root matrix* of the
# covariance, :math:`P = AA^T`, of the state to be approximated, and :math:`\mathbf{x}` is its
# mean.
#
# Two sets of weights, mean and covariance, are calculated:
#
# .. math::
#           W^m_0 &= \frac{\lambda}{c}
#
#           W^c_0 &= \frac{\lambda}{c} + (1 - \alpha^2 + \beta)
#
#           W^m_j &= W^c_j = \frac{1}{2 c}
#
# where :math:`\lambda = \alpha^2 (N_d + \kappa) - N_d`, :math:`c = N_d + \lambda`. The parameters
# :math:`\alpha, \ \beta, \ \kappa` are user-selectable parameters with default values of
# :math:`1, \ 2, \ 3 - N_d`.
#
# After the sigma points are transformed :math:`\mathbf{s^{\prime}} = f( \mathbf{s} )`, the
# distribution is reconstructed as:
#
# .. math::
#           \mathbf{x}^\prime &= \sum\limits^{2 N_d}_{0} W^{m}_j \mathbf{s}^{\prime}_j
#
#           P &= (\mathbf{s}^{\prime} - \mathbf{x}^\prime) \, diag(W^c) \, (\mathbf{s}^{\prime} -
#           \mathbf{x}^\prime)^T + Q
#
# The posterior mean and covariance are accurate to the 3rd order Taylor expansion for all
# non-linear models. [#]_

# %%
# Nearly-constant velocity example
# --------------------------------
# This example is equivalent to that in the previous (EKF) tutorial. As with that one, you are
# invited to play with the parameters and watch what happens.

# Some general imports and initialise time
import numpy as np
from matplotlib import pyplot as plt  # Set-up for plotting

from datetime import datetime, timedelta
start_time = datetime.now()

# %%

# import random
# random.seed(1991)

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

# Make noisy measurement (with bearing variance = 0.2 degrees).
measurement_model = CartesianToBearingRange(ndim_state=4,
                                            mapping=(0, 2),
                                            noise_covar=np.diag([np.radians(0.2), 1]),
                                            translation_offset=np.array([[sensor_x], [sensor_y]]))

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
# And plot
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
    min_ind = np.argmin(v[0, :])
    orient = np.arctan2(v[max_ind, 1], v[max_ind, 0])
    ellipse = Ellipse(xy=(state.state_vector[0], state.state_vector[2]),
                      width=np.sqrt(w[max_ind])*2, height=np.sqrt(w[min_ind])*2,
                      angle=np.rad2deg(orient),
                      alpha=0.2,
                      color='r')
    ax.add_artist(ellipse)
fig

# %%
# The UT in slightly more depth
# -----------------------------
# Now try and get a sense of what actually happens to the uncertainty when a non-linear combination
# of functions happens. Instead of deriving this analytically (and potentially getting bogged-down
# in the maths), let's just use a sampling method.
# We can start with a prediction, which is Gauss-distributed in state space, that we will use to
# make our measurement predictions from.
from stonesoup.types.prediction import GaussianStatePrediction
prediction = GaussianStatePrediction(state_vector=[[50], [0], [20], [0]],
                                     covar=np.diag([1.5, 0.5, 1.5, 0.5]),
                                     timestamp=datetime.now())

# %%
# We'll recapitulate the fact that the sensor position is where it previously was. But this time
# we'll make the measurement much noisier.
sensor_x = 50
sensor_y = 0

measurement_model = CartesianToBearingRange(
    ndim_state=4,
    mapping=(0, 2),
    noise_covar=np.diag([np.radians(8), 0.1]),  # bearing variance = 8 degrees (accurate range)
    translation_offset=np.array([[sensor_x], [sensor_y]])
)

# %%
# The next tutorial will go into much more detail on sampling methods. For the moment we'll just
# assert that we're generating 2000 points from the state prediction above.
#
# We need these imports and parameters:
from scipy.stats import multivariate_normal

from stonesoup.types.particle import Particle
from stonesoup.types.numeric import Probability  # Similar to a float type
from stonesoup.types.state import ParticleState

number_particles = 2000

# Sample from the Gaussian prediction distribution
samples = multivariate_normal.rvs(prediction.state_vector.ravel(),
                                  prediction.covar,
                                  size=number_particles)
particles = [
    Particle(sample.reshape(-1, 1), weight=Probability(1/number_particles)) for sample in samples]
# Create prior particle state.
pred_samples = ParticleState(particles, timestamp=start_time)

from stonesoup.resampler.particle import SystematicResampler
resampler = SystematicResampler()
from stonesoup.updater.particle import ParticleUpdater
pupdater = ParticleUpdater(measurement_model, resampler)

predict_meas_samples = pupdater.predict_measurement(pred_samples)

# %%
# Don't worry what all this means for the moment. It's a convenient way of showing the 'true'
# distribution of the predicted measurement - which is rendered as a blue cloud. Note that
# no noise is added by the :meth:`~.predict_measurement` method so we add some noise below.
# This is additive Gaussian in the sensor coordinates.
fig2 = plt.figure(figsize=(10, 6), tight_layout=True)
ax = fig2.add_subplot(1, 1, 1, polar=True)
ax.set_ylim(0, 30)
ax.set_xlim(0, np.radians(180))

data = np.array([particle.state_vector for particle in predict_meas_samples.particles])
noise = multivariate_normal.rvs(np.array([0, 0]), measurement_model.covar(), size=len(data))

ax.plot(data[:, 0].ravel()+noise[:, 0],
        data[:, 1].ravel()+noise[:, 1],
        linestyle='',
        marker=".",
        markersize=1.5,
        alpha=0.4)

fig2

# %%
# We can now see what happens when we create EKF and UKF updaters and compare their effect.
#
# Create updaters:
from stonesoup.updater.kalman import UnscentedKalmanUpdater, ExtendedKalmanUpdater
unscented_updater = UnscentedKalmanUpdater(measurement_model, alpha=1, beta=2)
extended_updater = ExtendedKalmanUpdater(measurement_model)

# Get predicted measurements from the state prediction.
ukf_pred_meas = unscented_updater.predict_measurement(prediction)
ekf_pred_meas = extended_updater.predict_measurement(prediction)

# %%
# Plot UKF (red) and EKF (green) predicted measurement distributions.

# Plot UKF's predicted measurement distribution
w, v = np.linalg.eig(ukf_pred_meas.covar)
max_ind = np.argmax(v[0, :])
min_ind = np.argmin(v[0, :])
orient = np.arctan2(v[max_ind, 1], v[max_ind, 0])
ukf_ellipse = Ellipse(xy=(ukf_pred_meas.state_vector[0], ukf_pred_meas.state_vector[1]),
                      width=np.sqrt(w[max_ind])*2, height=np.sqrt(w[min_ind])*2,
                      angle=np.rad2deg(orient),
                      alpha=0.4,
                      color='r')
ax.add_artist(ukf_ellipse)


# Plot EKF's predicted measurement distribution
w, v = np.linalg.eig(ekf_pred_meas.covar)
max_ind = np.argmax(v[0, :])
min_ind = np.argmin(v[0, :])
orient = np.arctan2(v[max_ind, 1], v[max_ind, 0])
ekf_ellipse = Ellipse(xy=(ekf_pred_meas.state_vector[0], ekf_pred_meas.state_vector[1]),
                      width=np.sqrt(w[max_ind])*2, height=np.sqrt(w[min_ind])*2,
                      angle=np.rad2deg(orient),
                      alpha=0.3,
                      color='g')
ax.add_artist(ekf_ellipse)

fig2

# sphinx_gallery_thumbnail_number = 5

# %%
# You may have to spend some time fiddling with the parameters to see major differences between the
# EKF and UKF. Indeed the point to make is not that there is any great magic about the UKF. Its
# power is that it harnesses some extra free parameters to give a more flexible description of the
# transformed distribution.

# %%
# Key points
# ----------
# 1. The unscented Kalman filter offers a powerful alternative to the EKF when undertaking tracking
#    in non-linear regimes.

# %%
# References
# ----------
# .. [#] Julier S., Uhlmann J., Durrant-Whyte H.F. 2000, A new method for the nonlinear
#        transformation of means and covariances in filters and estimators," in IEEE Transactions
#        on Automatic Control, vol. 45, no. 3, pp. 477-482, doi: 10.1109/9.847726.
# .. [#] https://www.seas.harvard.edu/courses/cs281/papers/unscented.pdf
