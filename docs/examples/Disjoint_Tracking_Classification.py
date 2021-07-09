#!/usr/bin/env python
# coding: utf-8

"""
Disjoint Tracking and Classification
====================================
This is a demonstration of a utilisation of the implemented Hidden Markov model and composite
tracking modules in order to categorise a target as well as track its kinematics.
"""

# %%
# All non-generic imports will be given in order of usage.

from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np

# %%
# Ground Truth, Categorical and Composite States
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# We will attempt to track and classify 3 targets.

# %%
# True Kinematics
# ---------------
# They will move in random directions from defined starting points.
from stonesoup.types.groundtruth import GroundTruthState
from stonesoup.models.transition.linear import ConstantVelocity,\
    CombinedLinearGaussianTransitionModel

start = datetime.now()

kinematic_state1 = GroundTruthState([0, 1, 0, 1], timestamp=start)  # x, vx, y, vy
kinematic_state2 = GroundTruthState([10, -1, 0, 1], timestamp=start)
kinematic_state3 = GroundTruthState([10, -1, 5, 1], timestamp=start)

kinematic_transition = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.1),
                                                              ConstantVelocity(0.1)])

# %%
# True Classifications
# --------------------
# A target may take one of three discrete hidden classes: 'bike', 'car' and 'bus'.
# It will be assumed that the targets cannot transition from one class to another, hence an
# identity transition matrix is given to the :class:`~.CategoricalTransitionModel` for all targets.
#
# A :class:`~.CategoricalState` class is used to store information on the classification/'category'
# of the targets. The `state_vector` of each will define a categorical distribution over the 3
# possible classes, whereby each component defines the probability that the target is of the
# corresponding class. For example, the state vector (0.2, 0.3, 0.5), with category names
# ('bike', 'car', 'bus') indicates that the target has a 20% probability of being class
# 'bike', a 30% probability of being class 'car' etc.
# It doesn't make sense to have a true target being a distribution over the possible classes, and
# therefore the true categorical states will have binary state vectors indicating a specific class
# for each target (i.e. a '1' at one state vector index, and '0's elsewhere).
# The :class:`~.CategoricalGroundTruthState` inherits directly from the base
# :class:`~.CategoricalState`.

from stonesoup.types.groundtruth import CategoricalGroundTruthState
from stonesoup.models.transition.categorical import CategoricalTransitionModel

hidden_classes = ['bike', 'car', 'bus']
gt_kwargs = {'timestamp': start, 'category_names': hidden_classes}
category_state1 = CategoricalGroundTruthState([0, 0, 1], **gt_kwargs)
category_state2 = CategoricalGroundTruthState([1, 0, 0], **gt_kwargs)
category_state3 = CategoricalGroundTruthState([0, 1, 0], **gt_kwargs)

category_transition = CategoricalTransitionModel(transition_matrix=np.eye(3),
                                                 transition_covariance=0.1*np.eye(3))

# %%
# Composite States
# ----------------
# Each target will have kinematics and a category to be inferred. These are contained within a
# :class:`~.CompositeState` type (in this instance the child class
# :class:`~.CompositeGroundTruthState`).

from stonesoup.types.groundtruth import CompositeGroundTruthState

initial_state1 = CompositeGroundTruthState([kinematic_state1, category_state1])
initial_state2 = CompositeGroundTruthState([kinematic_state2, category_state2])
initial_state3 = CompositeGroundTruthState([kinematic_state3, category_state3])

# %%
# Generating Ground Truth Paths
# -----------------------------
# Both the phsyical and categorical state of the targets need to be transition. While the category
# will remain the same, a transition model is used here for the sake of demonstration.

from stonesoup.types.groundtruth import GroundTruthPath

GT1 = GroundTruthPath([initial_state1], id='GT1')
GT2 = GroundTruthPath([initial_state2], id='GT2')
GT3 = GroundTruthPath([initial_state3], id='GT3')
ground_truth_paths = [GT1, GT2, GT3]

for GT in ground_truth_paths:
    for i in range(10):
        kinematic_sv = kinematic_transition.function(GT[-1][0],
                                                     noise=True,
                                                     time_interval=timedelta(seconds=1))
        kinematic = GroundTruthState(kinematic_sv,
                                     timestamp=GT[-1].timestamp + timedelta(seconds=1))

        category_sv = category_transition.function(GT[-1][1],
                                                   noise=True,
                                                   time_interval=timedelta(seconds=1))
        category = CategoricalGroundTruthState(category_sv,
                                               timestamp=GT[-1].timestamp + timedelta(seconds=1),
                                               category_names=hidden_classes)

        GT.append(CompositeGroundTruthState([kinematic, category]))

# Printing GT1
for state in GT1:
    vector = np.round(state[0].state_vector.flatten().astype(np.double), 2)
    print("%25s" % vector, ' -- ', state[1].category, ' -- ', state.timestamp)

# %%
# Plotting Ground Truths
# ----------------------
# Colour will be used in plotting as an indicator to category: red == 'bike', green == 'car',
# blue == 'bus'.
fig, axes = plt.subplots(1, 3, figsize=(10, 5))
fig.set_figheight(15)
fig.set_figwidth(15)
fig.subplots_adjust(wspace=5)
fig.tight_layout()

for ax in axes:
    ax.set_aspect('equal', 'box')

for GT in ground_truth_paths:
    X = list()
    Y = list()
    col = list(GT[0][1].state_vector)
    for state in GT:
        pos = state[0].state_vector
        X.append(pos[0])
        Y.append(pos[2])
    axes[0].plot(X, Y, color=col, label=GT[-1][1].category)
axes[0].legend(loc='upper left')
axes[0].set(title='GT', xlabel='X', ylabel='Y')
axes[1].set_visible(False)
axes[2].set_visible(False)


def set_axes_limits():
    xmax = max(ax.get_xlim()[1] for ax in axes)
    ymax = max(ax.get_ylim()[1] for ax in axes)
    xmin = min(ax.get_xlim()[0] for ax in axes)
    ymin = min(ax.get_ylim()[0] for ax in axes)
    for ax in axes:
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)


set_axes_limits()

# %%
# Measurement
# ^^^^^^^^^^^
# A new sensor will be created, that can provide the information needed to both track and classify
# the targets.

# %%
# Composite Detection
# -------------------
# Detections relating to both the kinematics and classification will be needed. Therefore we will
# create a sensor that outputs :class:`~.CompositeDetection` types. The input `sensors` list will
# provide the contents of these compositions. For this example we will provide a
# :class:`RadarBearingRange` and a :class:`CategoricalSensor` for kinematics and classification
# respectively.
# :class:`~.CompositeDetection` types have a `mapping` attribute, which defines what sub-state
# index each sub-detection was created from. For example, with a composite state of form:
# (kinematic state, categorical state), and composite detection with mapping (1, 0), this would
# indicate that the 0th index sub-detection was attained from the categorical state, and the 1st
# index sub-detection from the kinematic state.
from typing import Set, Union, Sequence

from stonesoup.base import Property
from stonesoup.sensor.sensor import Sensor
from stonesoup.types.detection import CompositeDetection


class CompositeSensor(Sensor):
    sensors: Sequence[Sensor] = Property(doc="A list of sensors.")
    mapping: Sequence = Property(default=None,
                                 doc="Mapping of which component states in the composite truth "
                                     "state is measured.")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.mapping is None:
            self.mapping = list(np.arange(len(self.sensors)))

    def measure(self, ground_truths: Set[CompositeGroundTruthState],
                noise: Sequence[Union[np.ndarray, bool]] = True,
                **kwargs) -> Set[CompositeDetection]:

        if isinstance(noise, bool) or len(noise) == 1:
            noise = len(self.sensors) * [noise]

        detections = set()
        for truth in ground_truths:

            sub_detections = list()

            states = [truth.sub_states[i] for i in self.mapping]

            for state, sub_sensor, sub_noise in zip(states, self.sensors, noise):
                sub_detection = sub_sensor.measure(
                    ground_truths={state},
                    noise=sub_noise
                ).pop()  # sub-sensor returns a set
                sub_detections.append(sub_detection)

            detection = CompositeDetection(sub_states=sub_detections,
                                           groundtruth_path=truth,
                                           mapping=self.mapping)
            detections.add(detection)

        return detections


# %%
# Kinematic Measurement
# ---------------------
# Measurements of the target's kinematics will be attained via a :class:`~.RadarBearingRange`
# sensor model.
from stonesoup.sensor.radar.radar import RadarBearingRange

radar = RadarBearingRange(ndim_state=4,
                          position_mapping=[0, 2],
                          noise_covar=np.diag([np.radians(0.05), 0.1]))


# %%
# Categorical Measurement
# -----------------------
# Using the hidden markov model, it is assumed the hidden class of the target cannot be directly
# observed, and instead indirect observations are taken. In this instance, observations of the
# target's size are taken ('small' or 'large'), which have direct implications as to the target's
# hidden class, and this relationship is modelled by the `emission matrix` of the
# :class:`~.CategoricalMeasurementModel', which is used by the :class:`CategoricalSensor` to
# provide :class:`CategoricalDetection` types.
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
                                    emission_covariance=0.1*np.eye(2),
                                    mapping=[0, 1, 2])

eo = CategoricalSensor(measurement_model=model,
                       category_names=['small', 'large'])

# %%
# Composite Sensor
# ----------------
# Creating the composite sensor class.
sensor = CompositeSensor(sensors=[eo, radar], mapping=[1, 0])

# %%
# Generating Measurements
# -----------------------
all_measurements = list()

for gts1, gts2, gts3 in zip(GT1, GT2, GT3):
    measurements_at_time = sensor.measure({gts1, gts2, gts3})
    timestamp = gts1.timestamp
    all_measurements.append((timestamp, measurements_at_time))

# Printing some measurements
for i,  (time, measurements_at_time) in enumerate(all_measurements):
    if i > 2:
        break
    print(f"{time:%H:%M:%S}")
    for measurement in measurements_at_time:
        vector = np.round(measurement.state_vector.flatten().astype(np.double), 2)
        print("%25s" % vector, ' -- ', measurement[0].category)

# %%
# Plotting Measurements
# ---------------------
# Colour will be used to indicate measurement category: orange == 'small', light-blue == 'large'.
for time, measurements in all_measurements:
    for measurement in measurements:
        loc = measurement[1].state_vector
        obs = measurement[0].state_vector
        col = list(measurement[0].measurement_model.emission_matrix @ obs)

        phi = loc[0]
        rho = loc[1]
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        axes[1].scatter(x, y, color=col, marker='x', s=100, label=measurement[0].category)

a = axes[1].get_legend_handles_labels()
b = {l: h for h, l in zip(*a)}
c = [*zip(*b.items())]
d = c[::-1]
axes[1].legend(*d, loc='upper left')

axes[1].set(title='Measurements', xlabel='X', ylabel='Y')
axes[1].set_visible(True)
set_axes_limits()
fig

# %%
# Tracking Components
# ^^^^^^^^^^^^^^^^^^^

# %%
# Predictor
# ---------
# Though not used by the tracking components here, a :class:`~.CompositePredictor` will predict
# the component states of a composite state forward, according to a list of sub-predictors.
#
# A :class:`~.HMMPredictor` specifically uses :class:`CategoricalTransitionModel` types to predict.
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.predictor.categorical import HMMPredictor
from stonesoup.predictor.composite import CompositePredictor

kinematic_predictor = KalmanPredictor(kinematic_transition)
category_predictor = HMMPredictor(category_transition)

predictor = CompositePredictor([kinematic_predictor, category_predictor])

# %%
# Updater
# -------
# The :class:`~.CompositeUpdater`composite updater will update each component sub-state according
# to a list of corresponding sub-updaters. It has no method to create measurement predictions.
# This is instead handled on instantiation of :class:`CompositeHypothesis` types: the expected
# arguments to the updater's `update` method.
from stonesoup.updater.kalman import ExtendedKalmanUpdater
from stonesoup.updater.categorical import HMMUpdater
from stonesoup.updater.composite import CompositeUpdater

kinematic_updater = ExtendedKalmanUpdater()
category_updater = HMMUpdater()

updater = CompositeUpdater(sub_updaters=[kinematic_updater, category_updater])

# %%
# Hypothesiser
# ------------
# The hypothesiser is a :class:'~.CompositeHypothesiser' type. It is in the data association step
# that tracking and classification are combined: for each measurement, a hypothesis is created for
# both a track's kinematic and categorical components. A :class:'~.CompositeHypothesis` type is
# created, which contains these sub-hypotheses, whereby its weight is equal to the product of the
# sub-hypotheses' weights. These sub-hypotheses should be probabilistic.
#
# The :class:`CompositeHypothesiser` uses a list of sub-hypothesisers to create these
# sub-hypotheses, hence the sub-hypothesisers should also be probabilistic.
# In this example we will define a hypothesiser that simply changes kinematic distance weights in
# to probabilities for hypothesising the kinematic sub-state of the track.
from stonesoup.measures import Mahalanobis
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.types.hypothesis import SingleProbabilityHypothesis
from stonesoup.types.multihypothesis import MultipleHypothesis


class ProbabilityHypothesiser(DistanceHypothesiser):
    def hypothesise(self, track, detections, timestamp, **kwargs):
        multi_hypothesis = super().hypothesise(track, detections, timestamp, **kwargs)
        single_hypotheses = multi_hypothesis.single_hypotheses
        prob_single_hypotheses = list()
        for hypothesis in single_hypotheses:
            prob_hypothesis = SingleProbabilityHypothesis(hypothesis.prediction,
                                                          hypothesis.measurement,
                                                          1/hypothesis.distance,
                                                          hypothesis.measurement_prediction)
            prob_single_hypotheses.append(prob_hypothesis)
        return MultipleHypothesis(prob_single_hypotheses, normalise=False, total_weight=1)


kinematic_hypothesiser = ProbabilityHypothesiser(predictor=kinematic_predictor,
                                                 updater=kinematic_updater,
                                                 measure=Mahalanobis())

# %%
# A :class:'CategoricalHypothesiser' is used for calculating categorical hypotheses.
# It utilises the :class:`~.ObservationAccuracy` measure: a multi-dimensional extention of an
# 'accuracy' score, essentially providing a measure of the similarity between two categorical
# distributions.
from stonesoup.hypothesiser.categorical import CategoricalHypothesiser
from stonesoup.hypothesiser.composite import CompositeHypothesiser

category_hypothesiser = CategoricalHypothesiser(predictor=category_predictor,
                                                updater=category_updater)
hypothesiser = CompositeHypothesiser(
    sub_hypothesisers=[kinematic_hypothesiser, category_hypothesiser]
)

# %%
# Data Associator
# ---------------
# We will use a standard :class:`GNNWith2DAssignment` data associator.
from stonesoup.dataassociator.neighbour import GNNWith2DAssignment

data_associator = GNNWith2DAssignment(hypothesiser)

# %%
# Prior
# -----
# As we are tracking in a composite state space, we should initiate tracks with a
# :class:`CompositeState` type. The kinematic sub-state of the prior is a usual Gaussian state. For
# the categorical sub-state of the prior, equal probability is given to all 3 of the possible
# hidden classes that a target might take (the category names are also provided here).
from stonesoup.types.state import GaussianState, CategoricalState, CompositeState

kinematic_prior = GaussianState([0, 0, 0, 0], np.diag([10, 10, 10, 10]))
category_prior = CategoricalState([1/3, 1/3, 1/3], category_names=hidden_classes)
prior = CompositeState([kinematic_prior, category_prior])

# %%
# Initiator
# ---------
# The initiator is composite. For each unassociated detection, a new track will be initiated. In
# this instance we use a :class:`~.CompositeUpdateInitiator` type. A detection has both kinematic
# and categorical information to initiate the 2 state space sub-states from. However, in an
# instance where a detection only provides one of these, the missing sub-state for the track will
# be initiated as the given prior's sub-state (eg. if a detection provides only kinematic
# information of the target, the track will initiate its categorical sub-state as the category_
# prior defined earlier).
from stonesoup.initiator.simple import SimpleMeasurementInitiator
from stonesoup.initiator.categorical import SimpleCategoricalInitiator
from stonesoup.initiator.composite import CompositeUpdateInitiator

kinematic_initiator = SimpleMeasurementInitiator(prior_state=kinematic_prior,
                                                 measurement_model=None)
category_initiator = SimpleCategoricalInitiator(prior_state=category_prior,
                                                measurement_model=None)
initiator = CompositeUpdateInitiator(prior_state=prior,
                                     sub_initiators=[kinematic_initiator, category_initiator])

# %%
# Deleter
# -------
# We can use the standard :class:`~.UpdateTimeStepsDeleter`.
from stonesoup.deleter.time import UpdateTimeStepsDeleter

deleter = UpdateTimeStepsDeleter(2)

# %%
# Tracker
# -------
# We will use a standard :class:`~.MultiTargetTracker`.
from stonesoup.tracker.simple import MultiTargetTracker

tracker = MultiTargetTracker(initiator, deleter, all_measurements, data_associator, updater)

# %%
# Tracking
# ^^^^^^^^

tracks = set()
for time, ctracks in tracker:
    tracks.update(ctracks)

print(f'Number of tracks {len(tracks)}')
for track in tracks:
    print(f'id: {track.id}')
    for state in track:
        vector = np.round(state[0].state_vector.flatten().astype(np.double), 2)
        print("%25s" % vector, ' -- ', state[1].category, ' -- ', state.timestamp)

# %%
# Plotting Tracks
# ---------------
# Colour will be used to indicate a track's hidden category distribution. The rgb value is defined
# by the 'bike', 'car', and 'bus' probabilities. For example, a track with high probability of
# being a 'bike' will have a high 'r' value, and hence appear more red.
for track in tracks:
    for i, state in enumerate(track[1:], 1):
        loc0 = track[i-1][0].state_vector.flatten()
        loc1 = state[0].state_vector.flatten()
        X = [loc0[0], loc1[0]]
        Y = [loc0[2], loc1[2]]
        axes[2].plot(X, Y, label='track', color=list(state[1].state_vector))

axes[2].set(title='Tracks', xlabel='X', ylabel='Y')
axes[2].set_visible(True)
set_axes_limits()
fig

# %%
# sphinx_gallery_thumbnail_number = 3
