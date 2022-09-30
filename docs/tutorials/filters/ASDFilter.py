#!/usr/bin/env python
# coding: utf-8

"""
================================================================
11 - Accumulated States Densities - Out-of-Sequence measurements
================================================================
"""
# %%
# Smoothing a filtered trajectory is an important task in live systems. Using
# Rauch–Tung–Striebel retrodiction after the normal filtering has a great effect on
# the filtered trajectories but it is not optimal because one has to calculate the
# retrodiction in an own step. In this point the Accumulated-State-Densities (ASDs) can help.
# In the ASDs the retrodiction is calculated in the prediction and update step.
# We use a multistate over time which can be pruned for better performance. Another advantage
# is the possibility to calculate Out-of-Sequence measurements in an optimal way.
# A more detailed introduction and the derivation of the formulas can be found in [#]_.
#

# %%
# First of all we plot the ground truth of one target moving on the Cartesian 2D plane.
# The target moves in a cubic function.

# %%
import matplotlib
from datetime import timedelta
from datetime import datetime
import numpy as np
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from matplotlib import pyplot as plt

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")

truth = GroundTruthPath()
start_time = datetime.now()
for n in range(1, 203):
    x = n
    y = 100 * (n - 100) ** 3
    varxy = np.array([[0.1, 0], [0, 0.1]])
    xy = np.random.multivariate_normal(np.array([x, y]), varxy)
    truth.append(GroundTruthState(np.array([[xy[0]], [xy[1]]]),
                                  timestamp=start_time + timedelta(seconds=n)))

# Plot the result
_ = ax.plot([state.state_vector[0, 0] for state in truth],
        [state.state_vector[1, 0] for state in truth],
        linestyle="--")

# %%
# Following we plot the measurements made of the ground truth. The measurements have
# an error matrix of variance 1 in both dimensions.

from scipy.stats import multivariate_normal
from stonesoup.types.detection import Detection

measurements = []
for state in truth:
    x, y = multivariate_normal.rvs(
        state.state_vector.ravel(), cov=np.diag([1, 1]))
    measurements.append(Detection(
        state.state_vector, timestamp=state.timestamp))

# Plot the result
ax.scatter([state.state_vector[0, 0] for state in measurements],
           [state.state_vector[1, 0] for state in measurements],
           color='b')
fig

# %%
# Now we have to setup a transition model for the prediction and the :class:`~.ASDPredictor`.

from stonesoup.models.transition.linear import \
    CombinedLinearGaussianTransitionModel, ConstantVelocity, \
    ConstantAcceleration
from stonesoup.predictor.kalman import ASDKalmanPredictor

transition_model = CombinedLinearGaussianTransitionModel(
    (ConstantAcceleration(0.1), ConstantAcceleration(0.1)))
predictor = ASDKalmanPredictor(transition_model)

# %%
# We have to do the same for the measurement model and the :class:`~.ASDKalmanUpdater`.

from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.updater.kalman import ASDKalmanUpdater

measurement_model = LinearGaussian(
    6,  # Number of state dimensions (position and velocity in 2D)
    (0, 3),  # Mapping measurement vector index to state index
    np.array([[1, 0],  # Covariance matrix for Gaussian PDF
              [0, 1]])
)
updater = ASDKalmanUpdater(measurement_model)

# %%
# We set up the state at position (0,0) with velocity and acceleration 0. We set max_nstep
# to 190 to see the complete trajectory.

from stonesoup.types.state import ASDGaussianState

prior = ASDGaussianState(multi_state_vector=[[0], [0], [0], [0], [0], [0]],
                         timestamps=start_time,
                         multi_covar=np.diag([1, 1, 1, 1, 1, 1]),
                         max_nstep=190)

# %%
# Last but not least we set up a track and execute the filtering. The first and last 10 steps
# are processed in sequence. All other measurements are divided in groups of 10 following in time.
# The latest one is processed first and the other 9 are used for filtering. In the end we plot the
# filtered trajectory

from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.track import Track

track = Track()
for i in range(0, len(measurements)):
    if i > 10 and i < 201:
        if i % 10 != 0:  # or i%10==3:
            continue
        if i % 10 == 0 and i > 0:
            # prediction and update of the newest measurement
            m = measurements[i]
            prediction = predictor.predict(prior, timestamp=m.timestamp)
            hypothesis = SingleHypothesis(prediction, m)
            # Used to group a prediction and measurement together
            post = updater.update(hypothesis)
            track.append(post)
            prior = track[-1]
            for j in range(9, 0, -1):
                # prediction and update for all OOS measurement. Beginning with the latest one.
                m = measurements[i - j]
                prediction = predictor.predict(prior, timestamp=m.timestamp)
                hypothesis = SingleHypothesis(prediction, m)
                # Used to group a prediction and measurement together
                post = updater.update(hypothesis)
                track.append(post)
                prior = track[-1]
    else:
        # the first 10 steps are for beginning of the ASD so that it is numerically stable
        m = measurements[i]
        prediction = predictor.predict(prior, timestamp=m.timestamp)
        hypothesis = SingleHypothesis(prediction, m)
        # Used to group a prediction and measurement together
        post = updater.update(hypothesis)
        track.append(post)
        prior = track[-1]

# Plot the resulting track
states = track[-1].state_list  # [state for state in track]

ax.plot([state.state_vector[0, 0] for state in states],
        [state.state_vector[3, 0] for state in states],
        marker=".")
fig

# %%
# References
# ----------
# .. [#] W. Koch and F. Govaers, On Accumulated State Densities with Applications to
#        Out-of-Sequence Measurement Processing in IEEE Transactions on Aerospace and Electronic Systems,
#        vol. 47, no. 4, pp. 2766-2778, OCTOBER 2011, doi: 10.1109/TAES.2011.6034663.

