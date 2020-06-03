#!/usr/bin/env python
# coding: utf-8

"""
4 - Particle filter tutorial
============================
"""

# %%
# **Approximation via random sampling**
#
# Obviously, if we could totally encapsulate a target's state probability distribution with some
# description, send it through some transition model that perfectly describes the target's movement
# and end-up with a distribution that completely describes the possible resultant states, we would
# be happy.
# The Extended and Unscented Kalman Filters handle non-linearity to a certain degree of accuracy.
# The issue tends to lie in approximation of the state covariance. What if we tried a method that
# didn't require us to convert the covariances?

# %%
# Instead of considering a state probability distribution as a whole, lets take a random sample
# of points (states) within that distribution, send them individually through the
# transition/measurement process, look at their resulting 'positions' and make sense of them from
# there.
# There are a few advantages to this: Once we have taken our random sample, we no longer need to
# consider the covariance for the distribution, and only need to transit the particles' states
# through the process; the underlying distribution need not be gaussian (as with EKF, UKF and the
# normal Kalman Filter algorithm); we can model any non-linearities in their entirety.
#
# Initially, the particles can be weighted equally, and normalised such that they sum to :math:`1`.
# Through the update step, particles will have their weights adjusted such that those lying closer
# to an incoming measurement will have greater weight.
#
# The posterior estimate is then made by the approximation:
#
# .. math::
#       p(\textbf{x}_{k}|\textbf{z}_{1:k}) \approx
#       \sum_{i} w_{k}^i \delta (\textbf{x}_{k} - \textbf{x}_{k}^i)
#
# where :math:`p(\textbf{x}_{k}|\textbf{z}_{1:k})` is the resulting distribution
# :math:`p(\textbf{x}_{k}|\textbf{z}_{1:k})` of the possible states of :math:`\textbf{x}_{k}` given
# observations :math:`\textbf{z}_{1:k} = {\textbf{z}_{1}, ..., \textbf{z}_{k}}` from 'time'
# :math:`1` to :math:`k` and :math:`w_{k}^i` are weights such that :math:`\sum_{i} w_{k}^i = 1`.
#
# Simply put:
#
# - Each particle goes through the predict/update process as with the Kalman Filters.
# - The predict step 'moves' the particles.
# - The update step will recalculate their weights for the next iteration.
#
# One significant drawback of this is that we are going to be sending a lot of particles through
# multiple steps, resulting in a decent computational complexity.
# Also, we must be careful that we select an appropriate particle/sample sparsity to get a good
# description of the prior distribution.
# After a few iterations, most particles will have moved far enough away that they have
# :math:`\sim 0` weight, hence provide no contribution to our estimates. But they are still being
# processed through at each predict and update step, which is quite a computational waste. Hence a
# re-sampling method would be useful to apply every so often (many exist, and are designed to
# redistribute these particles to areas where the posterior probability is higher).

# %%

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

# Plot the ground truth.

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
# Create sensor with bearing, range measurement model.

import numpy as np

from stonesoup.models.measurement.nonlinear import CartesianToBearingRange
from stonesoup.types.detection import Detection

sensor_x = 0
sensor_y = 0

measurement_model = CartesianToBearingRange(
    ndim_state=4,
    mapping=(0, 2),
    noise_covar=np.diag([np.radians(0.1), 0.1]),  # bad bearing, good range uncertainties
    translation_offset=np.array([[sensor_x], [sensor_y]])
)

measurements = []
for state in truth:
    measurement = measurement_model.function(state, noise=True)
    measurements.append(Detection(measurement, timestamp=state.timestamp))

# %%
# Plot detections.

from stonesoup.functions import pol2cart

x, y = pol2cart(
    np.hstack(state.state_vector[1, 0] for state in measurements),
    np.hstack(state.state_vector[0, 0] for state in measurements))
ax.scatter(x + sensor_x, y + sensor_y, color='b')
fig

# %%
# Create a :class:`~.ParticlePredictor`, which takes a collection of particles, with a given
# transition model, and predicts them forward as with the :class:`~.KalmanPredictor`.

from stonesoup.predictor.particle import ParticlePredictor
predictor = ParticlePredictor(transition_model)

# %%
# We will need to re-sample from the distribution when more particles end-up with zero weights. So
# we make the addition of a :class:`~.SystematicResampler`, which redistributes a given set of
# particles as described earlier. The logic is left to the specified re-sampler to determine when
# a new sample should be taken (in the case of :class:`~.SystematicResampler` that is at every
# iteration).

from stonesoup.resampler.particle import SystematicResampler
resampler = SystematicResampler()

# %%
# The resampling process is contained within the update stage. Therefore the
# :class:`~.ParticleUpdater` takes both a measurement model and a resampler as its arguments.

from stonesoup.updater.particle import ParticleUpdater
updater = ParticleUpdater(measurement_model, resampler)

# %%
# To start, we'll need to create a prior estimate of where we think our target will be, but in this
# case we'll need this to be a set of :class:`~.Particle`. For this, we sample from Gaussian
# distribution (using same parameters we had in the previous examples).

from scipy.stats import multivariate_normal

from stonesoup.types.particle import Particle
from stonesoup.types.numeric import Probability  # Similar to a float type
from stonesoup.types.state import ParticleState

number_particles = 200

# Sample from the prior Gaussian distribution
samples = multivariate_normal.rvs(np.array([0, 1, 0, 1]),
                                  np.diag([1, 1, 1, 1]),
                                  size=number_particles)

particles = [
    Particle(sample.reshape(-1, 1), weight=Probability(1/number_particles)) for sample in samples]

# Create prior particle state.
prior = ParticleState(particles, timestamp=start_time)

# %%
# We now run the predict and update steps, propagating the collection of particles and re-sampling
# when required.

from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.track import Track

track = Track()
for measurement in measurements:
    prediction = predictor.predict(prior, timestamp=measurement.timestamp)
    hypothesis = SingleHypothesis(prediction, measurement)
    post = updater.update(hypothesis)
    track.append(post)
    prior = track[-1]

# Plot the resulting track
ax.plot([state.state_vector[0, 0] for state in track],
        [state.state_vector[2, 0] for state in track],
        marker=".")
fig

# %%
# Plotting the sample points at each iteration.

for state in track:
    data = np.array([particle.state_vector for particle in state.particles])
    ax.plot(data[:, 0], data[:, 2], linestyle='', marker=".", markersize=1, alpha=0.5)
fig

# sphinx_gallery_thumbnail_number = 4

# %%
# That covers the basics of single sensor, single target tracking.
