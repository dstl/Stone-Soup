#!/usr/bin/env python
# coding: utf-8

"""
=======================
Ensemble Filter Example
=======================
"""

# %%
# The Ensemble Kalman Filter (EnKF) is a hybrid of the Kalman updating scheme and the Monte Carlo
# approach of the particle filter. The EnKF provides an alternative to the Kalman Filter (and
# extensions of) which is specifically designed for high-dimensional states.
#
# EnKF can be applied to both non-linear and non-Gaussian state-spaces due to being completely
# based on sampling.
#
# Multiple versions of EnKF are implemented in Stone Soup - all make use of the same prediction
# step, but implement different versions of the update step. Namely, the updaters are:
#
# - :class:`~.EnsembleUpdater`
# - :class:`~.LinearisedEnsembleUpdater`
# - :class:`~.RecursiveLinearisedEnsembleUpdater`
# - :class:`~.RecursiveUpdater`
#
# The :class:`~.EnsembleUpdater` is deliberately structured to resemble the Vanilla Kalman Filter,
# :meth:`update` first calls :meth:`predict_measurement` function which
# proceeds by calculating the predicted measurement, innovation covariance
# and measurement cross-covariance. Note however, these are not propagated
# explicitly, they are derived from the sample covariance of the ensemble itself.
#
# The :class:`~.LinearisedEnsembleUpdater` is an implementation of 'The Linearized EnKF Update'
# algorithm from "Ensemble Kalman Filter with Bayesian Recursive Update" by Kristen Michaelson,
# Andrey A. Popov and Renato Zanetti. Similar to the EnsembleUpdater, but uses a different form
# of Kalman gain. This alternative form of the EnKF calculates a separate kalman gain for each
# ensemble member. This alternative Kalman gain calculation involves linearization of the
# measurement. An additional step is introduced to perform inflation.
#
# The :class:`~.RecursiveLinearisedEnsembleUpdater` is an implementation of 'The Bayesian
# Recursive Update Linearized EnKF' algorithm from "Ensemble Kalman Filter with Bayesian
# Recursive Update" by Kristen Michaelson, Andrey A. Popov and Renato Zanetti. It is an
# extended version of the LinearisedEnsembleUpdater that recursively iterates over the
# update step for a given number of iterations. This recursive version is designed to
# minimise the error caused by linearisation.
#
# The :class:`~.RecursiveEnsembleUpdater` is an extended version of the
# :class:`~.EnsembleUpdater` which recursively iterates over the update step.

# %%
# Example using stonesoup
# -----------------------
# Package imports
# ^^^^^^^^^^^^^^^


import numpy as np
from datetime import datetime, timedelta


# %%
# Start time of simulation
# ^^^^^^^^^^^^^^^^^^^^^^^^


start_time = datetime.now()
np.random.seed(1991)


# %%
# Generate and plot ground truth
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
    ConstantVelocity
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.plotter import Plotterly

transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.05),
                                                          ConstantVelocity(0.05)])
truth = GroundTruthPath([GroundTruthState([0, 1, 0, 1], timestamp=start_time)])

for k in range(1, 21):
    truth.append(GroundTruthState(
        transition_model.function(truth[k - 1], noise=True, time_interval=timedelta(seconds=1)),
        timestamp=start_time + timedelta(seconds=k)))

# %%

plotter = Plotterly()
plotter.plot_ground_truths(truth, [0, 2])
plotter.fig

# %%
# Generate and plot detections
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^


from stonesoup.models.measurement.nonlinear import CartesianToBearingRange

sensor_x = 50  # Placing the sensor off-centre
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
    measurements.append(Detection(measurement, timestamp=state.timestamp,
                                  measurement_model=measurement_model))

# %%

plotter.plot_measurements(measurements, [0, 2])
plotter.fig

# %%
# Set up predictor and updater
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# In this EnKF example we must use the :class:`~.EnsemblePredictor`, and choose to use the
# standard :class:`~.EnsembleUpdater`. Note that we could instanciate any of the other (ensemble)
# updaters mentioned in this example, in place of the :class:`~.EnsembleUpdater`.


from stonesoup.predictor.ensemble import EnsemblePredictor
from stonesoup.updater.ensemble import EnsembleUpdater

predictor = EnsemblePredictor(transition_model)
updater = EnsembleUpdater(measurement_model)


# %%
# Prior state
# ^^^^^^^^^^^
# For the simulation we must provide a prior state. The Ensemble Filter in stonesoup requires
# this to be an :class:`~.EnsembleState`. We generate an :class:`~.EnsembleState` by calling
# :meth:`~.generate_ensemble`. The prior state stores the prior state vector - see below that
# for the :class:`~.EnsembleState`, the `state_vector` must be a :class:`~.StateVectors` object.


from stonesoup.types.state import EnsembleState

ensemble = EnsembleState.generate_ensemble(
    mean=np.array([[0], [1], [0], [1]]),
    covar=np.diag([1.5, 0.5, 1.5, 0.5]), num_vectors=100)
prior = EnsembleState(state_vector=ensemble, timestamp=start_time)

# %%

type(prior.state_vector)


# %%
# Run the EnKF
# ^^^^^^^^^^^^
# Here we run the Ensemble Kalman Filter and plot the results. By marking flag `particle=True`,
# we plot each member of the ensembles. As usual, the ellipses represent the uncertainty in the
# tracks.

from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.track import Track

track = Track()
for measurement in measurements:
    prediction = predictor.predict(prior, timestamp=measurement.timestamp)
    hypothesis = SingleHypothesis(prediction, measurement)  # Group a prediction and measurement
    post = updater.update(hypothesis)
    track.append(post)
    prior = track[-1]

plotter.plot_tracks(track, [0, 2], uncertainty=True, particle=True)
plotter.fig
