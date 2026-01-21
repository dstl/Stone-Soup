#!/usr/bin/env python

"""
===================================================================================
1 - An Introduction to Multiple Model Algorithms: GPB1, GPB2, and IMM in Stone Soup
===================================================================================
"""

# %%
# This tutorial introduces the fundamental concepts of the multiple model framework in Stone Soup.
# There are many multiple model algorithms in the literature, this tutorial focuses on the
# generalised pseudo-Bayesian of order 1 (GPB1), the generalised pseudo-Bayesian of order 2 (GPB2),
# and the interacting multiple model (IMM) algorithms. These algorithms are crucial for tracking
# manoeuvring targets where the underlying dynamic model of the target can switch over time.
# We will explore their theoretical foundations and demonstrate their implementation within the
# Stone Soup framework, highlighting how they address the complexities of state estimation in
# dynamic environments.

# %%
# Manoeuvring Target Scenario
# ---------------------------

from stonesoup.types.mixture import GaussianMixture
from stonesoup.types.state import ModelAugmentedWeightedGaussianState
from stonesoup.types.matrix import TransitionMatrix
from stonesoup.predictors import KalmanPredictors
from stonesoup.augmentor import ModelAugmentor
from stonesoup.reducer import IdentityReducer, ModelReducer

import numpy as np

from stonesoup.models.transition.linear import (CombinedLinearGaussianTransitionModel as CLGTM,
                                                ConstantVelocity as CV,
                                                KnownTurnRate as CT)

# %%
# Ground truth and detections
# ---------------------------

from datetime import datetime, timedelta

from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.models.transition.linear import (CombinedLinearGaussianTransitionModel,
                                                ConstantVelocity, KnownTurnRate as ConstantTurn)
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.types.numeric import Probability
from stonesoup.plotter import Plotter
from stonesoup.updater.kalman import KalmanUpdater

start_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
np.random.seed(1)

# %%
noise = 1e-2
rate1 = 0.01634
rate2 = 0.0232
gtcv = CombinedLinearGaussianTransitionModel([ConstantVelocity(noise),
                                              ConstantVelocity(noise)])
gtctl2 = CombinedLinearGaussianTransitionModel([ConstantTurn(np.array([noise, noise]), rate2)])
gtctr1 = CombinedLinearGaussianTransitionModel([ConstantTurn(np.array([noise, noise]), -rate1)])

# %%
truths = set()
segment_lengths = [150, 200, 40, 165, 45]
segment_lengths = [int(x) for x in segment_lengths]
models = [gtcv, gtctr1, gtcv, gtctl2, gtcv]
segment_ids = [j for n, i in enumerate(segment_lengths) for j in [n]*i]

truth = GroundTruthPath([GroundTruthState([3700, -30, 1700, -30], timestamp=start_time)])
for k, model in enumerate(segment_ids, 1):
    truth.append(GroundTruthState(
        models[model].function(truth[-1], noise=False, time_interval=timedelta(seconds=1)),
        timestamp=start_time + timedelta(seconds=k)))
truths.add(truth)

# %%
import numpy as np
from itertools import cycle
from stonesoup.types.detection import TrueDetection
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.multihypothesis import MultipleHypothesis
from stonesoup.types.track import Track

from datetime import timedelta


def generate_measurements(truths,
                          measurement_model=None,
                          as_set=True):
    if measurement_model is None:
        measurement_model = LinearGaussian(
                              ndim_state=4,
                              mapping=(0, 2),
                              noise_covar=np.array([[0.75, 0], [0, 0.75]]))
    all_measurements = []
    measurements = []
    for k in range(np.max([len(t) for t in truths])):
        measurement_set = set()
        for truth in truths:
            measurement = measurement_model.function(truth[k], noise=True)
            if as_set:
                measurement_set.add(TrueDetection(state_vector=measurement,
                                                  groundtruth_path=truth,
                                                  timestamp=truth[k].timestamp,
                                                  measurement_model=measurement_model))
            else:
                measurement_ = TrueDetection(state_vector=measurement,
                                             groundtruth_path=truth,
                                             timestamp=truth[k].timestamp,
                                             measurement_model=measurement_model)
                measurements.append(measurement_)
        all_measurements.append((truth[k].timestamp, measurement_set))
    return all_measurements, measurements


# %%
all_measurements, measurements = generate_measurements(
    truths,
    measurement_model=LinearGaussian(ndim_state=4,
                                     mapping=(0, 2),
                                     noise_covar=np.array([[0.725, 0], [0, 0.7025]])),
    as_set=True)

# %%
import matplotlib
matplotlib.rcParams.update({"font.size": 18})
gtplotter = Plotter()
ms = [m for _, m in all_measurements]
gtplotter.plot_ground_truths(truths, [0, 2], linewidth=2)
gtplotter.fig

# %%
from matplotlib import pyplot as plt

plt.figure(figsize=(12, 6))
plt.subplot(211)
plt.scatter([x.timestamp for x in truth], [x.state_vector[1] for x in truth])
plt.subplot(212)
plt.scatter([x.timestamp for x in truth], [x.state_vector[3] for x in truth])
plt.show()


# %%
# Kalman Filter as a Multiple Model Algorithm (:math:`M = 1`)
# -----------------------------------------------------------

start_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
cv = CLGTM([CV(1e-5), CV(1e-5)])
measurement_model_noise = 1e-3
transitioning_probabilities = TransitionMatrix(np.atleast_2d(1))
prior_state_vector = [[truth[0].state_vector[0]], [0], [truth[0].state_vector[2]], [0]]
prior_covar = np.diag([50**2, 3**2, 50**2, 3**2])

transition_models_list = [cv]

measurement_model = LinearGaussian(
    ndim_state=4, mapping=(0, 2),
    noise_covar=np.array([[measurement_model_noise, 0], [0, measurement_model_noise]]))
predictors = KalmanPredictors(transition_models_list)
updater = KalmanUpdater(measurement_model)
model_history = 0
measurement_history = 0
prior = ModelAugmentedWeightedGaussianState(
    state_vector=prior_state_vector,
    covar=prior_covar,
    timestamp=start_time,
    weight=Probability(1),
    model_histories=[],
    model_history_length=model_history)
priors = GaussianMixture([prior])


model_augmentor = ModelAugmentor(
    transition_probabilities=transitioning_probabilities,
    transition_models=transition_models_list,
    histories=model_history)
model_reducer = IdentityReducer(
    transition_probabilities=transitioning_probabilities,
    transition_model_list=transition_models_list,
    model_history_length=model_history)
measurement_reducer = IdentityReducer(
    transition_probabilities=transitioning_probabilities,
    transition_model_list=transition_models_list,
    model_history_length=model_history)

# %%
import numpy as np
from stonesoup.models.measurement.linear import LinearGaussian


def MultipleModelTracker(model_augmentor,
                         model_reducer,
                         predictors,
                         updater,
                         measurement_reducer,
                         priors,
                         all_measurements):
    track = Track([priors])
    for timestamp, measurementset in all_measurements:
        augmented_states = model_augmentor.augment(priors)
        priors = model_reducer.reduce(augmented_states, timestamp)
        all_posts = []
        for predictor, prior_ in zip(cycle(predictors.predictors), priors.components):
            prediction = predictor.predict(prior_, timestamp)
            posts = []
            hypotheses = MultipleHypothesis(
                [SingleHypothesis(prediction, m) for m in measurementset])
            for hypothesis in hypotheses:
                if not hypothesis:
                    posts.append(hypothesis.prediction)
                else:
                    post = updater.update(hypothesis)
                    posts.append(post)
            all_posts.extend(posts)
        all_posts = GaussianMixture(all_posts)
        posterior = measurement_reducer.reduce(all_posts, timestamp)
        track.append(posterior)
        priors = track[-1]
    return track


# %%
KF_track = MultipleModelTracker(model_augmentor, model_reducer, predictors, updater,
                                measurement_reducer, priors, all_measurements)

# %%
# OSPA, RMSE = run_metrics(KF_track, truths, start_time, "KF", 1)

# %%
plotter = Plotter()
plotter.plot_tracks(KF_track, [0, 2], color="orange", track_label="KF [CV]")
plotter.fig
# %%
# GPB1 (:math:`M=3`)
# ------------------

ctl = CLGTM([CT(np.array([1e-5, 1e-5]), 0.01)])
ctr = CLGTM([CT(np.array([1e-5, 1e-5]), -0.01)])
transition_models_list = [ctl, cv, ctr]
transitioning_probabilities = TransitionMatrix([0.25, 0.5, 0.25])

predictors = KalmanPredictors(transition_models_list)
updater = KalmanUpdater(measurement_model)
model_history = 0
measurement_history = 0
prior = ModelAugmentedWeightedGaussianState(
    state_vector=prior_state_vector,
    covar=prior_covar,
    timestamp=start_time,
    weight=Probability(1),
    model_histories=[],
    model_history_length=model_history)
priors = GaussianMixture([prior])

model_augmentor = ModelAugmentor(
    transition_probabilities=transitioning_probabilities,
    transition_models=transition_models_list,
    histories=model_history)
model_reducer = IdentityReducer(
    transition_probabilities=transitioning_probabilities,
    transition_model_list=transition_models_list,
    model_history_length=model_history)
measurement_reducer = ModelReducer(
    transition_probabilities=transitioning_probabilities,
    transition_model_list=transition_models_list,
    model_history_length=model_history)

GPB1_track = MultipleModelTracker(model_augmentor, model_reducer, predictors, updater,
                                  measurement_reducer, priors, all_measurements)

# %%
# GPB2 (:math:`M=3`)
# ------------------

transitioning_probabilities = TransitionMatrix([[0.90, 0.05, 0.05],
                                                [0.05, 0.90, 0.05],
                                                [0.05, 0.05, 0.90]])

predictors = KalmanPredictors(transition_models_list)
updater = KalmanUpdater(measurement_model)
model_history = 1
measurement_history = 0
prior = ModelAugmentedWeightedGaussianState(
    state_vector=prior_state_vector,
    covar=prior_covar,
    timestamp=start_time,
    weight=Probability(1),
    model_histories=[],
    model_history_length=model_history)
priors = GaussianMixture([prior, prior, prior])
model_augmentor = ModelAugmentor(
    transition_probabilities=transitioning_probabilities,
    transition_models=transition_models_list,
    histories=model_history)
model_reducer = IdentityReducer(
    transition_probabilities=transitioning_probabilities,
    transition_model_list=transition_models_list,
    model_history_length=model_history)
measurement_reducer = ModelReducer(
    transition_probabilities=transitioning_probabilities,
    transition_model_list=transition_models_list,
    model_history_length=model_history)


GPB2_track = MultipleModelTracker(model_augmentor, model_reducer, predictors, updater,
                                  measurement_reducer, priors, all_measurements)

# %%
# IMM (:math:`M=3`)
# -----------------

predictors = KalmanPredictors(transition_models_list)
updater = KalmanUpdater(measurement_model)
model_history = 1
measurement_history = 0
prior = ModelAugmentedWeightedGaussianState(
    state_vector=prior_state_vector,
    covar=prior_covar,
    timestamp=start_time,
    weight=Probability(1),
    model_histories=[],
    model_history_length=model_history)
priors = GaussianMixture([prior, prior, prior])


model_augmentor = ModelAugmentor(
    transition_probabilities=transitioning_probabilities,
    transition_models=transition_models_list,
    histories=model_history)
model_reducer = ModelReducer(
    transition_probabilities=transitioning_probabilities,
    transition_model_list=transition_models_list,
    model_history_length=model_history)
measurement_reducer = IdentityReducer(
    transition_probabilities=transitioning_probabilities,
    transition_model_list=transition_models_list,
    model_history_length=model_history)


IMM_track = MultipleModelTracker(model_augmentor, model_reducer, predictors, updater,
                                 measurement_reducer, priors, all_measurements)
