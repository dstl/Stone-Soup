#!/usr/bin/env python
# coding: utf-8

"""
Classification Using Hidden Markov Model
========================================
This is a demonstration using the implemented Hidden Markov model to classify multiple targets.

We will attempt to classify 3 targets in an undefined region.
Our sensor will be all-seeing, and provide us with indirect observations of the targets such that,
using the implemented hidden Markov Model (HMM), we should hopefully successfully classify exactly
3 targets correctly.
"""

# %%
# All Stone Soup imports will be given in order of usage.

from datetime import datetime, timedelta
import numpy as np

# %%
# Ground Truth
# ^^^^^^^^^^^^
# The targets may take one of three discrete hidden classes: 'bike', 'car' and 'bus'.
# It will be assumed that the targets cannot transition from one class to another, hence an
# identity transition matrix is given to the :class:`~.CategoricalTransitionModel`.
#
# A :class:`~.CategoricalState` class is used to store information on the classification/category
# of the targets. The state vector will define a categorical distribution over the 3 possible
# classes, whereby each component defines the probability that a target is of the corresponding
# class. For example, the state vector (0.2, 0.3, 0.5), with category names ('bike', 'car', 'bus')
# indicates that a target has a 20% probability of being class 'bike', a 30% probability of being
# class 'car' etc.
# It does not make sense to have a true target being a distribution over the possible classes, and
# therefore the true categorical states will have binary state vectors indicating a specific class
# (i.e. a '1' at one state vector index, and '0's elsewhere).
# The :class:`~.CategoricalGroundTruthState` class inherits directly from the base
# :class:`~.CategoricalState` class.
#
# While the category will remain the same, a :class:`~.CategoricalTransitionModel` is used here
# for the sake of demonstration.
#
# The category and timings for one of the ground truth paths will be printed.

from stonesoup.models.transition.categorical import CategoricalTransitionModel
from stonesoup.types.groundtruth import CategoricalGroundTruthState
from stonesoup.types.groundtruth import GroundTruthPath

category_transition = CategoricalTransitionModel(transition_matrix=np.eye(3),
                                                 transition_covariance=0.1 * np.eye(3))

start = datetime.now()

hidden_classes = ['bike', 'car', 'bus']

# Generating ground truth
ground_truths = list()
for i in range(1, 4):
    state_vector = np.zeros(3)  # create a vector with 3 zeroes
    state_vector[np.random.choice(3, 1, p=[1/3, 1/3, 1/3])] = 1  # pick a random class out of the 3
    ground_truth_state = CategoricalGroundTruthState(state_vector,
                                                     timestamp=start,
                                                     category_names=hidden_classes)

    ground_truth = GroundTruthPath([ground_truth_state], id=f"GT{i}")

    for _ in range(10):
        new_vector = category_transition.function(ground_truth[-1],
                                                  noise=True,
                                                  time_interval=timedelta(seconds=1))
        new_state = CategoricalGroundTruthState(
            new_vector,
            timestamp=ground_truth[-1].timestamp + timedelta(seconds=1),
            category_names=hidden_classes
        )

        ground_truth.append(new_state)
    ground_truths.append(ground_truth)

for states in np.vstack(ground_truths).T:
    print(f"{new_state.timestamp:%H:%M:%S}", end="")
    for state in states:
        print(f" -- {new_state.category}", end="")
    print()

# %%
# Measurement
# ^^^^^^^^^^^
# Using a hidden markov model, it is assumed the hidden class of a target cannot be directly
# observed, and instead indirect observations are taken. In this instance, observations of the
# targets' sizes are taken ('small' or 'large'), which have direct implications as to the targets'
# hidden classes, and this relationship is modelled by the `emission matrix` of the
# :class:`~.CategoricalMeasurementModel`, which is used by the :class:`~.CategoricalSensor` to
# provide :class:`~.CategoricalDetection` types.
# We will model this such that a 'bike' has a very small chance of being observed as a 'big'
# target. Similarly, a 'bus' will tend to appear as 'large'. Whereas, a 'car' has equal chance of
# being observed as either.

from stonesoup.models.measurement.categorical import CategoricalMeasurementModel
from stonesoup.sensor.categorical import CategoricalSensor

E = np.array([[0.99, 0.01],  # P(small | bike)  P(large | bike)
              [0.5, 0.5],
              [0.01, 0.99]])
model = CategoricalMeasurementModel(ndim_state=3,
                                    emission_matrix=E,
                                    emission_covariance=0.1 * np.eye(2),
                                    mapping=[0, 1, 2])

eo = CategoricalSensor(measurement_model=model,
                       category_names=['small', 'large'])

# Generating measurements
measurements = list()
for states in np.vstack(ground_truths).T:
    measurements_at_time = eo.measure(states)
    timestamp = next(iter(states)).timestamp
    measurements.append((timestamp, measurements_at_time))

    print(f"{timestamp:%H:%M:%S} -- {[meas.category for meas in measurements_at_time]}")

# %%
# Tracking Components
# ^^^^^^^^^^^^^^^^^^^

# %%
# Predictor
# ---------
# A :class:`~.HMMPredictor` specifically uses :class:`~.CategoricalTransitionModel` types to
# predict.
from stonesoup.predictor.categorical import HMMPredictor

predictor = HMMPredictor(category_transition)

# %%
# Updater
# -------
from stonesoup.updater.categorical import HMMUpdater

updater = HMMUpdater()

# %%
# Hypothesiser
# ------------
# %%
# A :class:`~.CategoricalHypothesiser` is used for calculating categorical hypotheses.
# It utilises the :class:`~.ObservationAccuracy` measure: a multi-dimensional extension of an
# 'accuracy' score, essentially providing a measure of the similarity between two categorical
# distributions.
from stonesoup.hypothesiser.categorical import CategoricalHypothesiser

hypothesiser = CategoricalHypothesiser(predictor=predictor, updater=updater)

# %%
# Data Associator
# ---------------
# We will use a standard :class:`~.GNNWith2DAssignment` data associator.
from stonesoup.dataassociator.neighbour import GNNWith2DAssignment

data_associator = GNNWith2DAssignment(hypothesiser)

# %%
# Prior
# -----
# As we are tracking in a categorical state space, we should initiate with a categorical state for
# the prior. Equal probability is given to all 3 of the possible hidden classes that a target
# might take (the category names are also provided here).
from stonesoup.types.state import CategoricalState

prior = CategoricalState([1 / 3, 1 / 3, 1 / 3], category_names=hidden_classes)

# %%
# Initiator
# ---------
# For each unassociated detection, a new track will be initiated. In this instance we use a
# :class:`~.SimpleCategoricalInitiator`, which specifically handles categorical state priors.
from stonesoup.initiator.categorical import SimpleCategoricalInitiator

initiator = SimpleCategoricalInitiator(prior_state=prior, measurement_model=None)

# %%
# Deleter
# -------
# We can use a standard :class:`~.UpdateTimeStepsDeleter`.
from stonesoup.deleter.time import UpdateTimeStepsDeleter

deleter = UpdateTimeStepsDeleter(2)

# %%
# Tracker
# -------
# We can use a standard :class:`~.MultiTargetTracker`.
from stonesoup.tracker.simple import MultiTargetTracker

tracker = MultiTargetTracker(initiator, deleter, measurements, data_associator, updater)

# %%
# Tracking
# ^^^^^^^^

tracks = set()
for time, ctracks in tracker:
    tracks.update(ctracks)

print(f"Number of tracks: {len(tracks)}")
for track in tracks:
    certainty = track.state_vector[np.argmax(track.state_vector)][0] * 100
    print(f"id: {track.id} -- category: {track.category} -- certainty: {certainty}%")
    for state in track:
        meas_string = f"associated measurement: {state.hypothesis.measurement.category}"
        print(f"{state.timestamp} -- {state.category} -- {meas_string}")
    print()

# %%
# Metric
# ^^^^^^
# Determining tracking accuracy.
# In calculating how many targets were classified correctly, only tracks with the highest
# classification certainty are considered.

excess_tracks = len(tracks) - len(ground_truths)  # target value = 0
sorted_tracks = sorted(tracks,
                       key=lambda track: track.state_vector[np.argmax(track.state_vector)][0],
                       reverse=True)
best_tracks = sorted_tracks[:3]
true_classifications = {ground_truth.category for ground_truth in ground_truths}
track_classifications = {track.category for track in best_tracks}

num_correct_classifications = len(true_classifications & track_classifications)

print(f"Excess tracks: {excess_tracks}")
print(f"No. correct classifications: {num_correct_classifications}")
