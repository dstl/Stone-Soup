#!/usr/bin/env python
# coding: utf-8

"""
2 - Extended Kalman filter tutorial
======================
"""

# %%
# **Linear approximation to non-linear systems:**
#
# To derive the recursive equations utilised by a Kalman Filter, we made the assumption that the
# transition of state :math:`\vec{x}_{k-1}` at 'time' :math:`k-1` to state :math:`\vec{x}_{k}` at
# time :math:`k` and a measurement :math:`\vec{\mu}_{expected}` at this time could be modelled
# linearly:
#
# .. math::
#                \vec{x}_{k} &= F_{k}\vec{x}_{k-1} + B_{k}\vec{u}_{k}\\
#       \vec{\mu}_{expected} &= H_{k}\hat{x}_{k}
#
# (as well as assuming that the control factor could be modelled linearly by a control matrix and
# vector :math:`B_{k}` and :math:`\vec{u}_{k}` respectively).
#
# For example, the case where a target moves at constant velocity in the :math:`(x,y)` plane:
#
# .. math::
#       \begin{bmatrix}
#       x_{t + \triangle t}\\
#       \dot{x}_{t + \triangle t}\\
#       y_{t + \triangle t}\\
#       \dot{y}_{t + \triangle t}\\
#       \end{bmatrix}
#       =
#       \begin{bmatrix}
#       1 & \triangle t & 0 & 0\\
#       0 & 1 & 0 & 0\\
#       0 & 0 & 1 & \triangle t\\
#       0 & 0 & 0 & 1\\
#       \end{bmatrix}
#       \begin{bmatrix}x_{t}\\
#       \dot{x}_{t}\\
#       y_{t}\\
#       \dot{y}_{t}\\
#       \end{bmatrix}
#       +
#       noise
#
# and the simple instance of measuring position in the same coordinate space as the state:
#
# .. math::
#       \begin{bmatrix}
#       x_{measure}\\
#       y_{measure}\\
#       \end{bmatrix}
#       =
#       \begin{bmatrix}
#       1 & 0 & 0 & 0\\
#       0 & 0 & 1 & 0\\
#       \end{bmatrix}
#       \begin{bmatrix}
#       x_{state}\\
#       \dot{x}_{state}\\
#       y_{state}\\
#       \dot{y}_{state}\\
#       \end{bmatrix}
#       +
#       noise
#
# However, consider a target moving in a circle.
# Perhaps :math:`(x,y) = (cos(t), sin(t))`.
# There is no linear mapping :math:`F:\ state\mapsto\ state` as:
#
# .. math::
#       x_{t + \triangle t} &= \cos(t + \triangle t)\\
#                           &= \cos(\triangle t)\cos(t) - \sin(\triangle t)\sin(t)\\
#                           &= \cos(\triangle t)x_{t} \pm \sin(\triangle t)\sqrt{1-x_{t}^2}
#
# which certainly is not a linear expression of :math:`x_{t}` (there is no matrix :math:`F` such
# that :math:`x_{k} = Fx_{k-1} + noise`).
#
# Similarly, maybe our sensor takes the cartesian state space :math:`(x, y)` and returns the
# bearing and range of the target :math:`(\theta, r)`.
# So at 'time' :math:`k`, given a state :math:`\begin{bmatrix}x_{k}\\y_{k}\\\end{bmatrix}` we
# receive:
#
# .. math::
#       \begin{bmatrix}
#       \theta\\
#       r\\
#       \end{bmatrix}
#       =
#       \begin{bmatrix}
#       \arctan(\frac{y}{x})\\
#       \sqrt{x^2 + y^2}
#       \end{bmatrix}
#       +
#       noise
#
# Clearly this is not expressible linearly in :math:`x` and :math:`y`.
#
# Granted, there is nothing wrong in determining a new state from a prior, as we simply apply the
# function :math:`F` to it. What gets in the way, however, is that our Kalman Filter's recursive
# equations rely on a matrix expression for posterior covariances and to determine the Kalman
# gain:
#
# .. math::
#       \hat{x}_{k}' = \hat{x}_{k} + K'(\vec{z_{k}} - H_{k}\hat{x}_{k})\\
#       P'_{k} = P_{k} - K'H_{k}P_{k}\\
#       K' = P_{k}H_{k}^T (H_{k}P_{k}H_{k}^T + R_{k})^{-1}
#
# Therefore we cannot use the Kalman Filter as before.
# Instead, we approximate the offending non-linear map using a Taylor expansion:
#
# .. math::
#       g(\textbf{x}) \approx g(\textbf{u}) +
#                             (\textbf{x}-\textbf{u})^T Dg(\textbf{u}) +
#                             \frac{1}{2!}(\textbf{x}-\textbf{u})^T
#                             D^2 g(\textbf{u})(\textbf{x}-\textbf{u}) +
#                             ...
#
# Truncating at the first term we have:
#
# .. math::
#       g(\textbf{x}) \approx g(\textbf{u}) + (\textbf{x}-\textbf{u})^T Dg(\textbf{u})\\
#
# for points 'close' to :math:`\textbf{u}`.
#
# This is starting to look like :math:`\vec{x}_{k} = F_{k}\vec{x}_{k-1} + B_{k}\vec{u}_{k}`
# (where :math:`F_{k}` is replaced by :math:`Dg(\textbf{u})`).
# In the case of a vector function, :math:`D` is the Jacobian matrix (denoted :math:`\textbf{J}`).
#
# By expanding near the prior/prediction mean (depending on whether transition or measurement are
# non-linear), the task is simplified to calculating the Jacobian matrix at each iteration, where:
#
# .. math::
#       \textbf{J}(\textbf{g})
#       =
#       \begin{bmatrix}
#       \frac{\partial g_{1}}{\partial x_{1}} & \dots & \frac{\partial g_{1}}{\partial x_{n}}\\
#       \vdots & \ddots & \vdots\\
#       \frac{\partial g_{m}}{\partial x_{1}} & \dots & \frac{\partial g_{m}}{\partial x_{n}}\\
#       \end{bmatrix}
#
# for :math:`\textbf{g}: \textbf{R}^n \mapsto \textbf{R}^m`.
#
# For example, the above measurements were modelled as:
#
# .. math::
#       \textbf{g}(x,y) = \begin{bmatrix}
#                          f(x,y)\\
#                          g(x,y)\\
#                          \end{bmatrix}
#                       = \begin{bmatrix}
#                          \arctan(\frac{y}{x})\\
#                          \sqrt{x^2 + y^2}\\
#                          \end{bmatrix}
#
# So,
#
# .. math::
#       \textbf{J}(\textbf{g}(x,y))
#       =
#       \begin{bmatrix}
#       \frac{\partial f(x,y)}{\partial x} & \frac{\partial f(x,y)}{\partial y}\\
#       \frac{\partial g(x,y)}{\partial x} & \frac{\partial g(x,y)}{\partial y}\\
#       \end{bmatrix}
#       =
#       \begin{bmatrix}
#       \frac{-y}{y^2 + x^2} & \frac{1}{x}\sec^2 (\frac{y}{y})\\
#       \frac{x}{\sqrt{x^2 + y^2}} & \frac{y}{\sqrt{x^2 + y^2}}
#       \end{bmatrix}
#
# The Jacobian is then used as an approximation to the measurement matrix :math:`H`  as in:
#
# .. math::
#       \vec{\mu}_{expected} &= H_{k}\hat{x}_{k}\\
#          \Sigma_{expected} &= H_{k}P_{k}H_{k}^T
#
# This process can be used to handle non-linear transitions, control and measurements.
# Using this linear approximation, the predict and update stages proceed as with the ordinary
# Kalman Filter.

# %%
# Simulate target
# ---------------
# As with the Kalman Filter tutorial, we will model a target moving with constant velocity.

from datetime import datetime
from datetime import timedelta

from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
    ConstantVelocity
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState

start_time = datetime.now()

transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.05),
                                                          ConstantVelocity(0.05)])

truth = GroundTruthPath([GroundTruthState([0, 1, 0, 1], timestamp=start_time)])

for k in range(1, 21):
    truth.append(GroundTruthState(
        transition_model.function(truth[k-1], noise=True, time_interval=timedelta(seconds=1)),
        timestamp=start_time+timedelta(seconds=k)))

# %%
# Plotting the ground truth path.
from matplotlib import pyplot as plt

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot()
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.axis('equal')

ax.plot([state.state_vector[0] for state in truth],
        [state.state_vector[2] for state in truth],
        linestyle="--")

# %%
# Create Extended Kalman predictor
# --------------------------------
# In Stone Soup the :class:`~.ExtendedKalmanPredictor` will check for linearity before predicting
# forward. If the system is linear, it will predict as with the normal Kalman Filter (without
# bothering to taylor expand and truncate (as this would be pointless)).

from stonesoup.predictor.kalman import ExtendedKalmanPredictor
predictor = ExtendedKalmanPredictor(transition_model)

# %%
# Simulate measurement
# --------------------
# We will model a non-linear measurement. Our sensor takes the cartesian state space and
# returns bearing, range pairs. This is modelled by the :class:`~.CartesianToBearingRange`
# measurement model.
#
# i.e. :math:`H: \begin{pmatrix}x\\ \dot{x}\\ y\\ \dot{y}\end{pmatrix} \mapsto
# \begin{pmatrix}\theta\\ r\end{pmatrix}` (the mapping will ignore velocities, only positional
# coordinates are important here).

import numpy as np

from stonesoup.models.measurement.nonlinear import CartesianToBearingRange

sensor_x = 10  # Placing the sensor off-centre
sensor_y = 0

measurement_model = CartesianToBearingRange(
    ndim_state=4,
    mapping=(0, 2),
    noise_covar=np.diag([np.radians(0.2), 1]),  # Covariance matrix. 0.2 degree variance in
    # bearing and 1 metre in range
    translation_offset=np.array([[sensor_x], [sensor_y]])  # Offset measurements to location of
    # sensor in cartesian.
)

# %%

from stonesoup.types.detection import Detection

measurements = []
for state in truth:
    measurement = measurement_model.function(state, noise=True)
    measurements.append(Detection(measurement, timestamp=state.timestamp))

# %%
# Plotting our detections. Bearing and range lines are superimposed on to the cartesian space. The
# actual measurement space would be two perpendicular axes for bearing and range, which is a little
# harder to visualise from.

fig2 = plt.figure(figsize=(10, 6))
fig2.add_subplot(1, 1, 1, polar=True)
plt.polar([state.state_vector[0, 0] for state in measurements],
          [state.state_vector[1, 0] for state in measurements])

# %%
# Mapping these detections back in cartesian coordinates (the state space's positional subspace).
from stonesoup.functions import pol2cart
x, y = pol2cart(
    np.hstack([state.state_vector[1, 0] for state in measurements]),
    np.hstack([state.state_vector[0, 0] for state in measurements]))
ax.scatter(x + sensor_x, y + sensor_y, color='b')
fig

# %%
# Create Extended Kalman updater
# ------------------------------
# As our measurement model is a non-linear mapping from state space to measurement space, the
# ordinary Kalman Filter update step will not work. As described above, an approximation of the
# measurement model will be required. Therefore we use the :class:`~.ExtendedKalmanUpdater` to
# update our prediction with a measurement at each step.

from stonesoup.updater.kalman import ExtendedKalmanUpdater
updater = ExtendedKalmanUpdater(measurement_model)

# %%
# Running the Extended Kalman Filter
# ----------------------------------
# First, we create a prior state as in the Kalman Filter tutorial.

from stonesoup.types.state import GaussianState
prior = GaussianState([[1], [1], [1], [1]], np.diag([1, 1, 1, 1]), timestamp=start_time)

# %%

from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.track import Track

track = Track()
for measurement in measurements:
    prediction = predictor.predict(prior, timestamp=measurement.timestamp)
    hypothesis = SingleHypothesis(prediction, measurement)  # Group a prediction and measurement
    post = updater.update(hypothesis)
    track.append(post)
    prior = track[-1]

# Plot the resulting track.
ax.plot([state.state_vector[0, 0] for state in track],
        [state.state_vector[2, 0] for state in track],
        marker=".")
fig

# %%
# Adding error ellipses at each estimate.

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
                      alpha=0.2)
    ax.add_artist(ellipse)
fig

# sphinx_gallery_thumbnail_number = 2

# %%
# The first order approximations used by the EKF provide a simple way to handle non-linear tracking
# problems. However, in highly non-linear systems these simplifications can lead to large errors in
# both the posterior state mean and covariance. In instances where we have noisy transition, or
# perhaps unreliable measurement, this could lead to a sub-optimal performance or even divergence
# of the filter. The **Unscented Kalman Filter** addresses these issues.
