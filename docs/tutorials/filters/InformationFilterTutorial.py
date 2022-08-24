#!/usr/bin/env python

"""
===========================
Information filter tutorial
===========================
"""

# %%
# This notebook is designed to introduce the Information Filter using a single target scenario as an example.
#
# Background and notation:
# ------------------------
#
# The information filter can be used when there is large, or infinite uncertainty about the initial
# state of an object. To compute the predicting and updating steps, the infinite covariance is
# converted to its inverse, using both the Precision matrix and the Information state.
# 
# We begin by creating a constant velocity model with q = 0.05.
# A ‘truth path’ is created starting at (20,20) moving to the NE at one distance unit per (time) step
# in each dimension. We propagate this with the transition model to generate a ground truth path.
#
# Firstly, we run the general imports, create the start time and build the Ground Truth constant velocity model.
# This follows the same procedure as shown in Tutorial 1 - Kalman Filter.
#
#


# %%
import numpy as np

from datetime import datetime, timedelta
start_time = datetime.now()

np.random.seed(1991)


# setting up ground truth

from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.models.transition.linear import (CombinedLinearGaussianTransitionModel,
                                               ConstantVelocity)

transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.05),ConstantVelocity(0.05)])

# Creating the initial truth state.

truth = GroundTruthPath([GroundTruthState([20,1,20,1],timestamp=start_time)])

# Generating the Ground truth path using a transition model.

for k in range(1,21):
    truth.append(GroundTruthState(
    transition_model.function(truth[k-1],noise=True,time_interval=timedelta(seconds=1)),
    timestamp=start_time + timedelta(seconds=k)))
    
# %%
# Importing the :class:`~.Plotterly` class from Stone Soup, we can plot the results.
# Note that the mapping argument is [0, 2] because those are the :math:`x` and :math:`y` position
# indices from our state vector.


# %%
# Plotting Ground Truths:
# ^^^^^^^^^^^^^^^^^^^^^^^

from stonesoup.plotter import Plotterly
plotter = Plotterly()
plotter.plot_ground_truths(truth,[0,2])
plotter.fig

# %%
# Taking Measurements:
# ^^^^^^^^^^^^^^^^^^^^
#
# As per the original Kalman tutorial, we’ll use one of Stone Soup’s measurement models in order to
# generate measurements from the ground truth. We shall assume a ‘linear’ sensor which detects the
# position only (not the velocity) of a target.
# 
# Omega is set to 5.
# 
# The linear Gaussian measurement model is set up by indicating the number of dimensions in the state
# vector and the dimensions that are measured (specifying :math:`{H}_{k}`) and the noise covariance matrix :math:`R`.
#


from stonesoup.types.detection import Detection
from stonesoup.models.measurement.linear import LinearGaussian

measurement_model = LinearGaussian(ndim_state = 4, # Number of state dimensions (position and velocity in 2D)
                                   mapping=(0,2), # Mapping measurement vector index to state index
                                   noise_covar=np.array([[5,0], # Covariance matrix for Gaussian PDF
                                                        [0,5]])
                                  )


measurements = []

for state in truth:
    measurement = measurement_model.function(state,noise=True)
    measurements.append(Detection(measurement,timestamp=state.timestamp,measurement_model = measurement_model))


# %%
# We plot the measurements using the Plotterly class in Stone Soup. Again specifying the :math:`x`, :math:`y` position
# indicies from the state vector.


plotter.plot_measurements(measurements,[0,2])
plotter.fig

# %%
# Running the Information Filter:
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We must first import the :class:`~.InformationKalmanPredictor`, and :class:`~.InformationKalmanUpdater` from the
# corresponding libraries.
#
# As before, the Predictor must be passed a Transition model, and the updater a Measurement model.
# 
# It is important to note that, unlike the Kalman Filter in Tutorial 1, the Information Filter requires
# the prior estimate to be in the form of an :class:`~.InformationState` (not a :class:`~.GaussianState`).
# The InformationState can be imported from stonesoup.types.state, and takes arguments: Information
# state, Precision Matrix and a timestamp.
# 
# The Precision Matrix is defined as : :math:`{Y}_{k-1}` = :math:`[{P}_{k-1}]^{-1}`.
# That is, the inverse of the covariance matrix.
# The information state is defined as : :math:`{y}_{k-1}` = :math:`[{P}_{k-1}]^{-1}` :math:`{x}_{k-1}`.
# That is the matrix multiplication of the Precision Matrix and the prior state in :math:`{x}_{k-1}`.
# 
# Using the same prior state as the original Kalman filtering example, we must firstly convert the
# covariance to be the Precision matrix, :math:`{Y}_{k-1}`, and calculate
# the :class:`~.InformationState`, :math:`{y}_{k-1}`.
#

from stonesoup.predictor.information import InformationKalmanPredictor
from stonesoup.updater.information import InformationKalmanUpdater


# Creating Information predictor and updater objects
predictor = InformationKalmanPredictor(transition_model)
updater = InformationKalmanUpdater(measurement_model)

# %%
# As before, we use the :class:`~.SingleHypothesis` class. The explicitly associates a single
# predicted state to a single detection.

from stonesoup.types.state import InformationState

# Precision matrix - the inverse of the covariance matrix
covar = np.diag([1.5, 0.5, 1.5, 0.5])
precision_matrix = np.linalg.inv(covar)


# yk_1 = Pk_1^-1 @ x_k-1
state = [20,1,20,1]
information_state = precision_matrix @ state


# Must use information state with precision matrix instead of Gaussian state
prior = InformationState(information_state, precision_matrix, timestamp=start_time)

from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.track import Track

track = Track()

for measurement in measurements:
    prediction = predictor.predict(prior,timestamp=measurement.timestamp)
    hypothesis = SingleHypothesis(prediction, measurement)
    post = updater.update(hypothesis)
    track.append(post)
    prior = track[-1]

# %%
# Plotting:
# ^^^^^^^^^
# Plotting the resulting track, including uncertainty ellipses.

# sphinx_gallery_thumbnail_number = 3

plotter.plot_tracks(track,[0,2],uncertainty=True)
plotter.fig


# %%
# Comparison to Kalman Filter:
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# We can show that the track created by the information filter and that of the Kalman filter are equivalent.
# Below we re-run the predicting and updating process with the :class:`~.KalmanPredictor` and :class:`~.KalmanUpdater`

from stonesoup.types.state import GaussianState
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.updater.kalman import KalmanUpdater

# Prior estimate:
prior = GaussianState([[20], [1], [20], [1]], np.diag([1.5, 0.5, 1.5, 0.5]), timestamp=start_time)

# Creating kalman predictor and updater
predictor = KalmanPredictor(transition_model)
updater = KalmanUpdater(measurement_model)

# Creating the kalman track
Kalman_track = Track()

for measurement in measurements:
    prediction = predictor.predict(prior,timestamp=measurement.timestamp)
    hypothesis = SingleHypothesis(prediction, measurement)
    post = updater.update(hypothesis)
    Kalman_track.append(post)
    prior = Kalman_track[-1]

# %%
# Comparison of Kalman Information Filter Tracks:
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We can use the Gaussian Hellinger measure to return the Hellinger distance between a
# pair of :class:`~.GaussianState` multivariate objects. The distance is bounded between 0-1, and we can therefore show
# that the total distance between the two tracks is close to zero.
#
#

from stonesoup.measures import GaussianHellinger

gh_measure = GaussianHellinger()

total_distance = 0

for information_state, kalman_state in zip(track, Kalman_track):
    distance = gh_measure(information_state.gaussian_state, kalman_state)

    total_distance += distance


print(total_distance)


# %%
# State Vector Comparison:
# ^^^^^^^^^^^^^^^^^^^^^^^^
#
# The state vector of the first position as an :class:`~.InformationState`.


print(track[0].state_vector)


# %%
# The state vector of the first position as a :class:`~.GaussianState`. We use the .gaussian_state property to
# convert to Gaussian form.
#
# We can obtain the covariance by taking the inverse of the Precision Matrix.
# We can also calculate the Gaussian state mean by multiplying by the covariance. In order to derive
# :math:`{x}_{k-1}`, we multiply by the covariance matrix.
#
# :math:`{x}_{k-1}` = :math:`[{P}_{k-1}]` :math:`{y}_{k-1}`.
#

print(track[0].gaussian_state.state_vector)

# %%



