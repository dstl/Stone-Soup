#!/usr/bin/env python
# coding: utf-8

"""
==========================================================
1 - An introduction to Stone Soup: using the Kalman filter
==========================================================
"""

# %%
# This notebook is designed to introduce some of the basic features of Stone Soup using a single
# target scenario and a Kalman filter as an example
#
# Background and notation
# -----------------------
#
# Let :math:`\mathbf{x}_{k} \in \mathbb{R}^n` be a state vector at some discrete point :math:`k`.
# The :math:`k` is most often interpreted as a timestep, but could be any sequential index
# (hence we don't use :math:`t`). The measurement is given by
# :math:`\mathbf{z}_{k} \in \mathbb{R}^m`.
#
# In Stone Soup, objects derived from the :class:`~.State` class carry information on a variety of
# states, whether they be hidden states, observations, ground truth. Minimally, a state will
# comprise a state vector and a timestamp. These can be extended. For example the
# :class:`~.GaussianState` is parameterised by a mean state vector and a covariance matrix. (We'll
# see this in action later.)
#
# Our goal is to infer :math:`\mathbf{x}_{k}`, given a sequence of measurements
# :math:`\mathbf{z}_{1}, ..., \mathbf{z}_{k}`, (which we'll write as :math:`\mathbf{z}_{1:k}`). In
# general :math:`\mathbf{z}` can include clutter and false alarms. We'll defer those complications
# to later tutorials and assume, for the moment, that all measurements are generated by a target.
#
# Prediction
# ^^^^^^^^^^
#
# We proceed under the Markovian assumption that :math:`p(\mathbf{x}_k) = \int_{-\infty} ^{\infty}
# p(\mathbf{x}_{k-1}) p(\mathbf{x}_k|\mathbf{x}_{k-1}) d \mathbf{x}_{k-1}`, meaning that the
# distribution over the state of an object at time :math:`k` can be predicted entirely from its
# state at previous time :math:`k-1`. If our understanding of :math:`\mathbf{x}_{k-1}` was informed
# by a series of measurements up
# to and including timestep :math:`k-1`, we can write
#
# .. math::
#           p(\mathbf{x}_k|\mathbf{z}_{1:k-1}) =
#           \int_{-\infty}^{\infty} p(\mathbf{x}_{k-1}|\mathbf{z}_{1:k-1})
#           p(\mathbf{x}_k|\mathbf{x}_{k-1}) d \mathbf{x}_{k-1}
#
# This is known as the *Chapman-Kolmogorov* equation. In Stone Soup we refer to this process as
# *prediction* and to an object that undertakes it as a :class:`~.Predictor`. A predictor requires
# a *state transition model*, namely a function which undertakes
# :math:`\mathbf{x}_{k|k-1} = f(\mathbf{x}_{k-1}, \mathbf{w}_k)`, where :math:`\mathbf{w}_k` is a
# noise term. Stone Soup has transition models derived from the :class:`~.TransitionModel` class.
#
# Update
# ^^^^^^
#
# We assume a sensor measurement is generated by some stochastic process represented by a function,
# :math:`\mathbf{z}_k = h(\mathbf{x}_k, \boldsymbol{\nu}_k)` where :math:`\boldsymbol{\nu}_k` is
# the noise.
#
# The goal of the update process is to generate the *posterior state estimate*,
# :math:`\mathbf{x}_k` from the prediction and the measurement. It does this by way of Bayes' rule,
#
# .. math::
#           p(\mathbf{x}_k | \mathbf{z}_{1:k}) =
#           \frac{ p(\mathbf{z}_{k} | \mathbf{x}_k) p(\mathbf{x}_k | \mathbf{z}_{1:k-1})}
#           {p(\mathbf{z}_k)}
#
# In Stone Soup, this is accomplised by the :class:`~.Updater` class. Updaters use a
# :class:`~.MeasurementModel` class which models the effect of :math:`h(\cdot)`.
#
# We then proceed recursively, the posterior distribution at :math:`k-1` becoming the prior
# for the next measurement timestep, and so on

# %%
# A nearly-constant velocity example
# ----------------------------------
#
# We're going to set up a simple scenario in which a target moves at constant velocity with the
# addition of some random noise, (referred to as a *nearly constant velocity* model).
#
# As is customary in Python scripts we begin with some imports. (These ones allow us access to
# mathematical, timing  and plotting functions.)
import numpy as np
from datetime import datetime, timedelta

# Figure to plot truth (and future data)
from matplotlib import pyplot as plt

# %%
# Simulate a target
# ^^^^^^^^^^^^^^^^^
#
# We consider a 2d Cartesian scheme where the state vector is
# :math:`[x \ \dot{x} \ y \ \dot{y}]^T`. To start we'll create a simple truth path, sampling at 1
# second intervals. We'll do this by employing one of Stone Soup's native transition models.
#
# These inputs are required:
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
                                               ConstantVelocity

# And the clock starts
start_time = datetime.now()

# %%
# We note that it can sometimes be useful to fix our random number generator in order to probe a
# particular example repeatedly. That option is available by uncommenting the next lines.

# import random
# random.seed(1991)

# %%
# The :class:`~.ConstantVelocity` class creates a one-dimensional nearly constant velocity model
# with specified noise. The :class:`~.CombinedLinearGaussianTransitionModel` class takes a number
# of 1d models and combines them in a linear Gaussian model of arbitrary dimension. We want a 2d
# simulation, so we'll do:
transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.05),
                                                          ConstantVelocity(0.05)])

# %%
# A 'truth path' is created starting at 0,0 moving to the NE
truth = GroundTruthPath([GroundTruthState([0, 1, 0, 1], timestamp=start_time)])

for k in range(1, 21):
    truth.append(GroundTruthState(
        transition_model.function(truth[k-1], noise=True, time_interval=timedelta(seconds=1)),
        timestamp=start_time+timedelta(seconds=k)))

# %%
# Thus the ground truth is generated and we can plot the result
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.axis('equal')
ax.plot([state.state_vector[0] for state in truth],
        [state.state_vector[2] for state in truth],
        linestyle="--")

# %%
# In the linear Gaussian transition model the transition function is linear with additive
# Gauss-distributed noise.
#
# :math:`\mathbf{x}_k  = F_k \mathbf{x}_{k-1} + \mathbf{w}_k`, :math:`\mathbf{w}_k \sim
# \mathcal{N}(0,Q)`, with
#
# .. math::
#           F_{k} &= \begin{bmatrix}
#                     1 & \triangle t & 0 & 0\\
#                     0 & 1 & 0 & 0\\
#                     0 & 0 & 1 & \triangle t\\
#                     0 & 0 & 0 & 1\\
#                     \end{bmatrix}
#
# and
#
# .. math::
#           Q_k &= \begin{bmatrix}
#                   \frac{\triangle t^3}{3} & \frac{\triangle t^2}{2} & 0 & 0\\
#                     \frac{\triangle t^2}{2} & \triangle t & 0 & 0\\
#                     0 & 0 & \frac{\triangle t^3}{3} & \frac{\triangle t^2}{2}\\
#                     0 & 0 & \frac{\triangle t^2}{2} & \triangle t\\
#                      \end{bmatrix} \sigma
#
# where :math:`\sigma` is the magnitude of the noise per unit time given to the constant velocity
# model.

# %%
# We can check the :math:`F_k` and :math:`Q_k` matrices (generated over a 1s period).
transition_model.matrix(time_interval=timedelta(seconds=1))

# %%
transition_model.covar(time_interval=timedelta(seconds=1))

# %%
# At this point you can play with the various parameters and see how it effects the simulated
# output.

# %%
# Simulate measurements
# ^^^^^^^^^^^^^^^^^^^^^
#
# We'll use one of Stone Soup's measurement models in order to generate
# measurements from the ground truth. For the moment we assume a 'linear' sensor which detects the
# position, but not velocity, of a target, such that
#
# :math:`\mathbf{z}_k = H_k \mathbf{x}_k + \boldsymbol{\nu}_k`,
# :math:`\boldsymbol{\nu}_k \sim \mathcal{N}(0,R)`, with
#
# .. math::
#           H_k &= \begin{bmatrix}
#                     1 & 0 & 0 & 0\\
#                     0  & 0 & 1 & 0\\
#                       \end{bmatrix}
#
# and
#
# .. math::
#
#           R &= \begin{bmatrix}
#                   1 & 0\\
#                     0 & 1\\
#                \end{bmatrix} \omega
#
# where :math:`\omega` is set to 0.75 initially (but again, feel free to play around).

# %%
# We're going to need a :class:`~.Detection` type to
# store the detections, and a :class:`~.LinearGaussian` measurement model.
from stonesoup.types.detection import Detection
from stonesoup.models.measurement.linear import LinearGaussian

# %%
# The linear Gaussian measurement model is set up by indicating the number of dimensions in the
# state vector and the dimensions that are measured (so specifying :math:`H_k`) and the noise
# covariance matrix :math:`R`.
measurement_model = LinearGaussian(
    ndim_state=4,  # Number of state dimensions (position and velocity in 2D)
    mapping=(0, 2),  # Mapping measurement vector index to state index
    noise_covar=np.array([[0.75, 0],  # Covariance matrix for Gaussian PDF
                          [0, 0.75]])
    )

# %%
# Check the output is as we expect
measurement_model.matrix()

# %%
# Generate the measurements
measurements = []
for state in truth:
    measurement = measurement_model.function(state, noise=True)
    measurements.append(Detection(measurement, timestamp=state.timestamp))

# Plot the result
ax.scatter([state.state_vector[0] for state in measurements],
           [state.state_vector[1] for state in measurements],
           color='b')
fig

# %%
# At this stage you should have a moderately linear ground truth path (dotted line) with a series
# of simulated measurements overplotted (blue circles). Take a moment to fiddle with the numbers in
# :math:`Q` and :math:`R` to see what it does to the path and measurements.

# %%
# Construct a Kalman filter
# ^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We're now ready to build a tracker. We'll use a Kalman filter as it's conceptually the simplest
# to start with. The Kalman filter is described extensively elsewhere [#]_, [#]_, so for the
# moment we just assert that the prediction step proceeds as:
#
# .. math::
#           \mathbf{x}_{k|k-1} &= F_{k}\mathbf{x}_{k-1} + B_{k}\mathbf{u}_{k}\\
#                 P_{k} &= F_{k}P_{k-1}F_{k}^T + Q_{k},
#
# :math:`B_k \mathbf{u}_k` is a control term which we'll ignore for now (we assume we don't
# influence the target directly).
#
# The update step is:
#
# .. math::
#       \mathbf{x}_{k} = \mathbf{x}_{k|k-1} + K_k(\mathbf{z}_k - H_{k}\mathbf{x}_{k|k-1})\\
#       P_{k} = P_{k|k-1} - K_k H_{k} P_{k|k-1}\\
#
# where,
#
# .. math::
#       K_k = P_{k|k-1} H_{k}^T S_k^{-1}\\
#       S_k = H_{k} P_{k|k-1} H_{k}^T + R_{k}
#
# :math:`\mathbf{z}_k - H_{k}\mathbf{x}_{k|k-1}` is known as the *innovation* and :math:`S_k` the
# *innovation covariance*; :math:`K_k` is the *Kalman gain*.

# %%
# Constructing a predictor and updater in Stone Soup is simple. In a nice division of
# responsibility, a :class:`~.Predictor` takes a :class:`~.TransitionModel` as input and
# an :class:`~.Updater` takes a :class:`~.MeasurementModel` as input. Note that for now we're using
# the same models used to generate the ground truth and the simulated measurements. This won't
# usually be possible and it's an interesting exercise to explore what happens when these
# parameters are mismatched.
from stonesoup.predictor.kalman import KalmanPredictor
predictor = KalmanPredictor(transition_model)

from stonesoup.updater.kalman import KalmanUpdater
updater = KalmanUpdater(measurement_model)

# %%
# Running the Kalman filter
# ^^^^^^^^^^^^^^^^^^^^^^^^^
# Now we have the components, we can execute the Kalman filter estimator on the simulated data.
#
# In order to start, we'll need to create the first prior estimate. We're going to use the
# :class:`~.GaussianState` we mentioned earlier. As the name suggests, this parameterises the state
# as :math:`\mathcal{N}(\mathbf{x}_0, P_0)`. By happy chance the initial values are chosen to match
# the truth quite well. You might want to manipulate these to see what happens.
from stonesoup.types.state import GaussianState
prior = GaussianState([[0], [1], [0], [1]], np.diag([1.5, 0.5, 1.5, 0.5]), timestamp=start_time)

# %%
# In this instance data association is done somewhat implicitly. There is one prediction and
# one detection per timestep so no need to think too deeply. Stone Soup discourages such
# (undesirable) practice and requires that a :class:`~.Prediction` and :class:`~.Detection` are
# associated explicitly. This is done by way of a :class:`~.Hypothesis`, the most simple of which
# is a :class:`~.SingleHypothesis` which associates a single predicted state with a single
# detection. There is much more detail on how the :class:`~.Hypothesis` class is used in later
# tutorials.
from stonesoup.types.hypothesis import SingleHypothesis

# %%
# With this, we'll now loop through our measurements, predicting and updating at each timestep.
# Uncontroversially, a Predictor has :meth:`predict` function and an Updater an :meth:`update` to
# do this. Storing the information is facilitated by the top-level :class:`~.Track` class which
# holds a sequence of states.
from stonesoup.types.track import Track
track = Track()
for measurement in measurements:
    prediction = predictor.predict(prior, timestamp=measurement.timestamp)
    hypothesis = SingleHypothesis(prediction, measurement)  # Group a prediction and measurement
    post = updater.update(hypothesis)
    track.append(post)
    prior = track[-1]

# %%
# Plot the resulting track, including uncertainty ellipses
ax.plot([state.state_vector[0] for state in track],
        [state.state_vector[2] for state in track],
        marker=".")

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

# sphinx_gallery_thumbnail_number = 3

# %%
# Key points
# ----------
# 1. Stone Soup is built on a variety of types of :class:`~.State` object. These can be used to
#    represent hidden states, observations, estimates, ground truth, and more.
# 2. Bayesian recursion is undertaken by the successive application of three classes of object.
#    These are the :class:`~.Predictor`, the :class:`~.Associator`, and the :class:`~.Updater`.
#    Broadly speaking Predictors apply a :class:`~.TransitionModel`, Associators use a
#    :class:`~.Hypothesiser` to associate prediction with a measurement, and Updaters use this
#    association together with the :class:`~.MeasurementModel` to calculate the posterior state
#    estimate.

# %%
# References
# ----------
# .. [#] Kalman 1960, A New Approach to Linear Filtering and Prediction Problems, Transactions of
#        the ASME, Journal of Basic Engineering, 82 (series D), 35
#        (https://pdfs.semanticscholar.org/bb55/c1c619c30f939fc792b049172926a4a0c0f7.pdf?_ga=2.51363242.2056055521.1592932441-1812916183.1592932441)
# .. [#] Anderson & Moore 2012, Optimal filtering,
#        (http://users.cecs.anu.edu.au/~john/papers/BOOK/B02.PDF)
