#!/usr/bin/env python
# coding: utf-8

"""
1 - Kalman filter tutorial
======================
"""

# %%
# **An exercise in deriving the best estimate of an unknown feature using Kalman filtering:**
#
# Let :math:`\textbf{x}_{k}` be a hidden state vector at some (time) :math:`k` and
# :math:`\textbf{z}_{k}` be a measurement.
#
# We want to infer :math:`\textbf{x}_{k}` given a sequence of measurements
# :math:`(\textbf{z}_{1}, ..., \textbf{z}_{k})`.
#
# Obviously, the state of an object at time :math:`k` depends on its state at previous time
# :math:`k-1`.
# We can define :math:`\textbf{f}:\ state \mapsto\ state` such that:
#
# .. math::
#           \textbf{f}(\textbf{x}_{k}\ |\ \textbf{x}_{k-1},\ \textbf{w}_{k})
#
# is a function describing the transition from the state at time :math:`k-1` to the state at time
# :math:`k`, with the consideration that the presence noise :math:`\textbf{w}_{k}` will effect the
# resultant state.
#
# For example, the linear acceleration of a body would provide noise to an object that is modelled
# to be travelling at constant velocity.
# (In this situation (with velocity :math:`\textbf{u}`) one could define this noise as:
#
# .. math::
#           \textbf{x}_{k} &= \textbf{x}_{k-1} + \triangle t \textbf{u}
#                                              + \frac{1}{2} \textbf{a}\triangle t^2\\
#                          &= \textbf{F}(\textbf{x}_{k-1}) + \frac{1}{2} \textbf{a}\triangle t^2
#
# where :math:`\textbf{F}` accounts for the transition due to constant velocity, and we have
# adjusted the end state with an addition that takes acceleration in to account)
#
# Similarly, we define the function :math:`\textbf{h}:\ state \mapsto\ measurement`
#
# .. math::
#           \textbf{h}(\textbf{z}_{k}\ |\ \textbf{x}_{k},\ \textbf{v}_{k})
#
# which describes how the sensor (providing us with measurements) responds to the state
# :math:`\textbf{x}_{k}` of the object (at time :math:`k`), with some noise :math:`\textbf{v}_{k}`
# (since our measurements come with their own uncertainties).
#
#
# We can rewrite:
#
# .. math::
#           \textbf{w}_{k} &\sim \textbf{H}_{k}(\eta_{1}, \eta_{2}, ...)\\
#           \textbf{v}_{k} &\sim \textbf{Z}_{k}(\zeta_{1}, \zeta_{2}, ...)
#
# where :math:`\textbf{H}` and :math:`\textbf{Z}` are stochastic processes (something
# demonstrating randomness in output) with parameters :math:`\eta` and :math:`\zeta`.
# These functions generate instances of the noises represented by :math:`\textbf{w}` and
# :math:`\textbf{v}`.
#
# So,
#
# .. math::
#           \textbf{x}_{k|k-1} &\sim p(\textbf{x}_{k}\ |\ \textbf{x}_{k-1},\ \eta_{k})\\
#               \textbf{z}_{k} &\sim p(\textbf{z}_{k}\ |\ \textbf{x}_{k},\ \zeta_{k})
#
# In other words, we concern ourselves with the probability distribution of the state and
# measurement.
#
# Then, adapting the law of total probability and Baye's rule:
#
# .. math::
#           p(x) &= \int_{X} p(x|x')p(x')dx'\\
#           P(X|Y) &= \frac{P(Y|X)P(X)}{P(Y)}
#
# We get:
#
# The Chapman-Kolmogorov equation, which defines the probability
# :math:`p(\textbf{x}_{k}\ |\ \textbf{z}_{1:k-1})` of the current state given that all observations
# up to the previous step :math:`k-1` have occurred):
#
# .. math::
#           p(\textbf{x}_{k}\ |\ \textbf{z}_{1:k-1}) =
#           \int_{-\infty}^{\infty} p(\textbf{x}_{k}\ |\ \textbf{x}_{k-1})
#           p(\textbf{x}_{k-1}\ |\ \textbf{z}_{1:k-1})\ d\textbf{x}_{k-1}
#
# and the probability :math:`p(\textbf{x}_{k}\ |\ \textbf{z}_{1:k})` of the current state given all
# observations up to (time) :math:`k`:
#
# .. math::
#           p(\textbf{x}_{k}\ |\ \textbf{z}_{1:k}) =
#           \frac{p(\textbf{z}_{k}\ |\ \textbf{x}_{k})p(\textbf{x}_{k}\ |\ \textbf{z}_{1:k-1})}
#                {p(\textbf{z}_{k})}
#
#
# Our state estimation process runs recursively: Given a *prior* estimate of a state vector, we
# *predict* the state at the next time step, and *update* this with a measurement to get a
# *posterior* state estimate.

# %%
# **The following is a derivation of the analytic solution to the recursive equations above,
# alongside a demonstration of the Kalman Filter within Stone Soup:**

# %%
# Simulate target
# -----------------
#
# Consider the case of a target having state:
#
# .. math::
#       \vec{x} = \begin{bmatrix} x\\ \dot{x}\\ y\\ \dot{y} \end{bmatrix}
#
# where :math:`(x, y)` is the target's position, and :math:`(\dot{x}, \dot{y})` its constant
# velocities in the :math:`x` and :math:`y` directions respectively.
#
# Let our best estimate of the target's state at time :math:`k` be denoted by :math:`\hat{x}_{k}`.
#
# Position and velocity are correlated (higher velocity :math:`\Rightarrow` further
# travelled in a time-step).
# To capture this correlation, we define a *covariance matrix* :math:`\Sigma` where
# :math:`\Sigma_{ij}` is the degree of correlation between the :math:`ith` and :math:`jth` state
# variable.
# For example, :math:`\Sigma_{x\dot{x}}` denotes the correlation between the target's x-position
# and its velocity in that direction.
#
# The target's general covariance matrix will be of the form:
#
# .. math::
#       P_{k} = \begin{bmatrix}
#               \Sigma_{xx} & \Sigma_{x\dot{x}} & \Sigma_{xy} & \Sigma_{x\dot{y}}\\
#               \Sigma_{\dot{x}x} & \Sigma_{\dot{x}\dot{x}} & \Sigma_{\dot{x}y} &
#               \Sigma_{\dot{x}\dot{y}}\\
#               \Sigma_{yx} & \Sigma_{y\dot{x}} & \Sigma_{yy} & \Sigma_{y\dot{y}}\\
#               \Sigma_{\dot{y}x} & \Sigma_{\dot{y}\dot{x}} & \Sigma_{\dot{y}y} &
#               \Sigma_{\dot{y}\dot{y}}
#               \end{bmatrix}
#
# This is actually a definition of the variables' join variances.
# For example for the variables :math:`X, Y`: :math:`Covariance(X, Y) = E[(X-E[X])(Y-E[Y])]`.
# So, the elements on the diagonal are the variances for the corresponding variables.
#
#
# Simulate target transition
# ----------------------------
# The target's state at time :math:`k` must depend on its state at previous time :math:`k-1`.
#
# Ie. :math:`\exists` :math:`F_{k}` such that :math:`\vec{x}_{k} = F_{k}\vec{x}_{k-1}`.
#
# In the example, we consider velocity to be constant:
#
# .. math::
#       \textbf{position}_{k} &= \textbf{position}_{k-1} +
#       \textbf{velocity}_{k-1}\ *\ \triangle t\\
#       \textbf{velocity}_{k} &= \textbf{velocity}_{k-1}
#
# Therefore, our prediction at time :math:`k` relates to that at time :math:`k-1` by the following:
#
# .. math::
#           \hat{x}_{k} &= \begin{bmatrix}
#                     1 & \triangle t & 0 & 0\\
#                     0 & 1 & 0 & 0\\
#                     0 & 0 & 1 & \triangle t\\
#                     0 & 0 & 0 & 1\\
#                     \end{bmatrix} \hat{x}_{k-1}\\
#                       &= F_{k}\hat{x}_{k-1}
#
# We must also predict the target's next covariance matrix :math:`P_{k}` (at time *k*), given the
# matrix at time *k-1* :math:`P_{k-1}`.
# Using the identity:
# :math:`Covariance(Ax) = A\Sigma A^T` (for some matrix :math:`A`) :math:`\forall x` where
# :math:`\Sigma` is the corresponding covariance matrix for state :math:`x`,
#
# we can predict that:
#
# .. math::
#           P_{k} &= Covariance(F_{k}x_{k-1})\\
#                 &= F_{k}Covariance(x_{k-1})F_{k}^T\\
#                 &= F_{k}P_{k-1}F_{k}^T
#
# We can account for additive external disturbances that we understand by appending a *control*
# factor composed of a control matrix :math:`B_{k}` and control vector :math:`\vec{u}_{k}` to the
# state estimate:
#
# .. math::
#           \hat{x}_{k} = F_{k}\hat{x}_{k-1} + B_{k}\vec{u}_{k}
#
# For example, we might know that the target has acceleration :math:`a`.
# Then:
#
# .. math::
#       \textbf{position}_{k} &= \textbf{position}_{k-1} + \triangle t \textbf{velocity}_{k-1} +
#       \frac{1}{2} a\triangle t^2\\
#       \textbf{velocity}_{k} &= \textbf{velocity}_{k-1} + a\triangle t
#
# Simulate transition noise
# -------------------------
# We must also account for external influences on the state that we do not know about.
#
# By accounting for additive noise in our prediction, we will define a distribution of possible
# resultant states. Consider the previous state :math:`\hat{x}_{k-1}` to have transitioned to a
# range of possible resultant states, with that distribution's mean being :math:`\hat{x}_{k}`.
#
# Define this as a Gaussian with covariance :math:`Q_{k}`.
# Appending this to the covariance matrix gives us the full expressions for our prediction:
#
# .. math::
#           \hat{x}_{k} &= F_{k}\hat{x}_{k-1} + B_{k}\vec{u}_{k}\\
#                 P_{k} &= F_{k}P_{k-1}F_{k}^T + Q_{k}
#
# Or, put in to words:
#
# predicted state (distribution) = previous state (distribution) after transition + some noise
#
# correlations between state parts = previous correlations after transition + some noise

# %%
# To start we'll create a simple :class:`~.GroundTruthPath`, which moves in 1 second intervals.
# We will represent an object moving with :class:`~.ConstantVelocity` in both :math:`x` and
# :math:`y` by stacking two 1-dimensional transition models.
# The class :class:`~.ConstantVelocity` models transition as :math:`x_k = F_k x_{k-1} + w_{k}`
# where :math:`x = \begin{bmatrix} x_{pos}\\ x_{vel} \end{bmatrix}` and
# :math:`F_{k=t+\triangle t} = \begin{bmatrix} 1 & \triangle t\\ 0 & 1 \end{bmatrix}`.
# At each time step, we will create a new :class:`~.GroundTruthState` and append this to a
# :class:`~.GroundTruthPath`.

# Some general imports and set up
from datetime import datetime
from datetime import timedelta

from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
    ConstantVelocity
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState

start_time = datetime.now()

# Variance of 0.05 in both x and y directions.
transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.05),
                                                          ConstantVelocity(0.05)])

# The target starts at the origin, with an initial velocity of :math:`1` in both the :math:`x` and
# :math:`y` directions and begins at the time stamp specified.
truth = GroundTruthPath([GroundTruthState([0, 1, 0, 1], timestamp=start_time)])
for k in range(1, 21):
    # Append a new ground truth state to the ground truth path in 1 second intervals.
    truth.append(GroundTruthState(
        transition_model.function(truth[k-1], noise=True, time_interval=timedelta(seconds=1)),
        timestamp=start_time+timedelta(seconds=k)))

# %%
# Plotting the ground truth.
from matplotlib import pyplot as plt

# Making a figure for future plotting.
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.axis('equal')

ax.plot([state.state_vector[0] for state in truth],
        [state.state_vector[2] for state in truth],
        linestyle="--")

# %%
# This model gives us our full transition matrix (shown here for a period of 1 second):
transition_model.matrix(time_interval=timedelta(seconds=1))

# %%
# And the *process noise* covariance (shown here for a period of 1 second):
transition_model.covar(time_interval=timedelta(seconds=1))

# %%
# Create Kalman predictor
# ---------------------------------
# Prediction relies on a model for the target's transition, so the :class:`~.KalmanPredictor` takes
# this as its argument.
# In reality, the true transitions of the target would be unknown, and we would use a transition
# model to approximate movement. For the sake of these tutorials, we will use the same model that
# created the ground truth for the prediction step.

from stonesoup.predictor.kalman import KalmanPredictor
predictor = KalmanPredictor(transition_model)

# %%
# Simulate measurement
# --------------------
# Given a prediction of the target's state and covariance at time :math:`k`, we receive a
# measurement from a sensor (by weighting our "trust" in our own prediction, and that of the
# sensor's measurement, we can provide a refined estimate of the target's state at time :math:`k`).
#
# Obviously, a measurement depends on the actual state of the system. We can infer that, given our
# prediction, there is some mapping :math:`H_{k}:\ state\mapsto\ measurement` such that we
# can define a distribution of possible measurements we might get (and corresponding expected
# covariance), given :math:`\hat{x}_{k}`:
#
# .. math::
#       \vec{\mu}_{expected} &= H_{k}\hat{x}_{k}\\
#          \Sigma_{expected} &= H_{k}P_{k}H_{k}^T

# %%
# Simulate measurement errors
# ---------------------------
#
# Adjusting our measurement in the same manner to the predicted state gives:
#
# .. math::
#       \vec{\mu}_{expected} &= H_{k}\hat{x}_{k}\\
#          \Sigma_{expected} &= H_{k}P_{k}H_{k}^T + R_{k}
#
# accounting for some noise :math:`R_{k}`.
# The Gaussian distribution defined by this has a mean equal to the detection/measurement received
# from the sensor.
#
# Next we'll create our model which will describe errors of measuring in the observable state, and
# how the hidden state maps to observable state. In this case we can simply map the :math:`x` and
# :math:`y` directly.

import numpy as np

from stonesoup.models.measurement.linear import LinearGaussian
measurement_model = LinearGaussian(
    ndim_state=4,  # Our state space has 4 dimensions (2 positional and 2 velocity coordinates)
    mapping=(0, 2),  # The 0 and 2 indices of the state vector correspond to position
    noise_covar=np.array([[0.75, 0],
                          [0, 0.75]])  # Covariance matrix. 0.75 metre variance in x and y
    )

# %%
# Pass the ground truth through the measurement model, and make a list of :class:`~.Detection`
# types.

from stonesoup.types.detection import Detection

measurements = []
for state in truth:
    measurement = measurement_model.function(state, noise=True)
    measurements.append(Detection(measurement, timestamp=state.timestamp))

# Plot the measurements on top of ground truth.
ax.scatter([state.state_vector[0] for state in measurements],
           [state.state_vector[1] for state in measurements],
           color='b')
fig

# %%
# This will give us the observation model :math:`H_{k}`:
measurement_model.matrix()

# %%
# And observation model covariance :math:`R_{k}`:
measurement_model.covar()

# %%
# Create Kalman updater
# ---------------------
# As the update step combines prediction with measurement, the :class:`~.KalmanUpdater` requires a
# measurement model as an argument.

from stonesoup.updater.kalman import KalmanUpdater
updater = KalmanUpdater(measurement_model)

# %%
# For the sake of simplicity, consider just 1 dimension of this problem (determining :math:`x` and
# :math:`\dot{x}`).
# We now have two Gaussian distributions: One surrounding our prediction, and the other around the
# detection/measurement.
# Multiplying these distributions together gives another Gaussian, defining a region encompassing
# states that satisfy the uncertainty in our prediction and the uncertainty in the sensor's
# measurement.
#
# Let:
#
# .. math::
#       K = \Sigma_{state} (\Sigma_{state} + \Sigma_{meas})^{-1}
#
# where :math:`\Sigma_{state}` is the covariance matrix of our prediction, and
# :math:`\Sigma_{meas}` that of the measurement.
#
# Then the mean :math:`\mu'` and standard deviation :math:`\sigma'` of the resulting Gaussian
# distribution are given by:
#
# .. math::
#       \vec{\mu}' &= \vec{\mu}_{state} + K(\vec{\mu}_{meas} - \vec{\mu}_{state})\\
#          \Sigma' &= \Sigma_{state} - K\Sigma_{state}
#
# where :math:`\vec{\mu}_{state}` and :math:`\vec{\mu}_{meas}` are the means of our prediction
# uncertainty distribution and that of the measurement respectively.
#
# Without explicitly showing substitution and simplification, using these formulae with the derived
# prediction, measurement and corresponding covariance expressions gives:
#
# .. math::
#       \hat{x}_{k}' = \hat{x}_{k} + K'(\vec{z_{k}} - H_{k}\hat{x}_{k})\\
#       P'_{k} = P_{k} - K'H_{k}P_{k}\\
#       K' = P_{k}H_{k}^T (H_{k}P_{k}H_{k}^T + R_{k})^{-1}
#
# This defines our best estimate :math:`\hat{x}_{k}'` of the state :math:`\vec{x}_{k}` of the
# target at time :math:`k`.
# :math:`K'` is called the 'Kalman gain' and can be considered as a sort weighting factor,
# determining how much you trust your measurement over your prediction.
# This can now be used in another phase of prediction, followed by adjustment from a measurement
# etc.

# %%
# Running the Kalman Filter
# --------------------------

# Now we have the components, we can run our simulated data through the Kalman Filter.
# To start, we'll need to create a prior estimate of where we think our target will be (as the
# Kalman Filter recursive equations require a prior state at every iteration, hence the first step
# will need something to kick things off). For this tutorial, we know exactly how our target starts
# off, so a simple :class:`~.GaussianState` will be a good representative.

from stonesoup.types.state import GaussianState
prior = GaussianState([[0], [1], [0], [1]], np.diag([1.5, 0.5, 1.5, 0.5]), timestamp=start_time)

# %%
# With this, we'll now loop through our measurements, predicting and updating at each time step.
# Now, we predict the next state from the 'prior' state with the Kalman predictor, group this with
# an incoming measurement to make a :class:`~.SingleHypothesis` type, and use the Kalman updater to
# make our 'posterior' state estimate (and its corresponding covariances).
# The resulting estimations at each iteration are appended to a :class:`~.Track` type.

from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.track import Track

track = Track()
for measurement in measurements:
    prediction = predictor.predict(prior, timestamp=measurement.timestamp)
    hypothesis = SingleHypothesis(prediction, measurement)  # Group a prediction and measurement
    post = updater.update(hypothesis)
    track.append(post)
    prior = track[-1]

# %%
# Plot the resulting track.
ax.plot([state.state_vector[0] for state in track],
        [state.state_vector[2] for state in track],
        marker=".")
fig

# %%
# Adding error ellipses at each estimate (representing gaussian distributions to 1 standard
# deviation).

from matplotlib.patches import Ellipse
for state in track:
    w, v = np.linalg.eig(measurement_model.matrix()@state.covar@measurement_model.matrix().T)
    max_ind = np.argmax(v[0, :])
    orient = np.arctan2(v[max_ind, 1], v[max_ind, 0])
    ellipse = Ellipse(xy=(state.state_vector[0], state.state_vector[2]),
                      width=np.sqrt(w[0])*2, height=np.sqrt(w[1])*2,
                      angle=np.rad2deg(orient),
                      alpha=0.2)
    ax.add_artist(ellipse)
fig

# sphinx_gallery_thumbnail_number = 4

# %%
# There are situations in which a linear expression such as
# :math:`\hat{x}_{k} = F_{k}\hat{x}_{k-1} + B_{k}\vec{u}_{k}` that models the underlying system is
# not attainable.
# Determining posterior covariances and the Kalman gain rely on us expressing transitions and
# measurements in a linear/matrix form.
# The **Extended Kalman Filter** handles these situations by linearising the offending non-linear
# maps, as detailed in the next tutorial.
