#!/usr/bin/env python
# coding: utf-8

"""
========================================================
General Multi Hypotheses tracking implementation example
========================================================
"""

# %%
# Multi hypotheses tracking (MHT) algorithm is considered one of the best algorithm
# for visual tracking, which consists in creating a tree of potential tracks for
# each target candidate (in a multi-target scenario) and prune such hypotheses
# in the data association phase. It is particularly efficient in maintain trajectories of
# multiple objects and handling uncertainties and ambiguities of tracks (e.g. presence of
# clutter).
# MHT, by definition, has several algorithms that fall under this definition, which
# we can list as Global Nearest Neighbour (GNN), Joint Probabilistic Data association
# (JPDA), Multi-frame assignment (MFA [#]_, see example here), Multi Bernoulli filter
# and Probabilistic multi hypotheses tracking (PMHT).
# In this example we employ the multi-frame assignment data associator for showing how
# to use the various components present in Stone Soup.
#
# This example follows this structure:
# 1. Create ground truth and detections;
# 2. Instantiate the tracking components and tracker;
# 3. Run the tracker and visualise the results.
#

# %%
# General imports
# ^^^^^^^^^^^^^^^
import numpy as np
from datetime import datetime, timedelta
from itertools import tee
from copy import deepcopy

# %%
# Stone soup imports
# ^^^^^^^^^^^^^^^^^^
from stonesoup.types.array import StateVector, CovarianceMatrix
from stonesoup.types.state import GaussianState
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity
from stonesoup.models.measurement.nonlinear import CartesianToBearingRange
from stonesoup.simulator.simple import MultiTargetGroundTruthSimulator, SimpleDetectionSimulator

# Simulation parameters
np.random.seed(1908)  # fix a random seed
start_time = datetime.now().replace(microsecond=0)
simulation_steps = 50
birth_rate = 0.2  # 20%
death_probability = 0.05  # 5%
timestep_size = timedelta(seconds=2)
prob_detection = 0.99
initial_state_mean = StateVector([[10], [0], [10], [0]])
initial_covariance = CovarianceMatrix(np.diag([5, 1, 5, 1]))

# clutter will be generated uniformly in this are around the target
clutter_area = np.array([[-1, 1], [-1, 1]])*50
clutter_rate = 0.5
surveillance_area = ((clutter_area[0][1] - clutter_area[0][0])*
                     (clutter_area[1][1] - clutter_area[1][0]))
clutter_spatial_density = clutter_rate/surveillance_area

# %%
# 1. Create ground truth and detections;
# --------------------------------------
# We have prepared all the general parameters for the simulation,
# including the clutter spatial density, target birth rate and
# death probability. We can, now, instantiate the transition model
# of the targets and the measurement model. In this example we employ
# :class:`~.CartesianToBearingRange` non-linear measurement model.
# Then we pass all these details to a :class:`~.MultiTargetGroundTruthSimulator`
# and use a :class:`~.SimpleDetectionSimulator`
# to obtain the target ground truth tracks, detections and clutter.
#

# Create an initial state
initial_state = GaussianState(state_vector=initial_state_mean,
                              covar=initial_covariance,
                              timestamp=start_time)

# Instantiate the transition model
transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.05),
                                                          ConstantVelocity(0.05)])

# Define a measuremnet model
measurement_model = CartesianToBearingRange(ndim_state=4,
                                            mapping=(0, 2),
                                            noise_covar=np.diag([np.radians(0.5), 3]))

# Instantiate the multi-target simulator
ground_truth_simulator = MultiTargetGroundTruthSimulator(
    transition_model=transition_model,
    initial_state=initial_state,
    timestep=timestep_size,
    number_steps=simulation_steps,
    birth_rate=birth_rate,
    death_probability=death_probability)

# Create a detector
detection_sim = SimpleDetectionSimulator(
    groundtruth=ground_truth_simulator,
    measurement_model=measurement_model,
    detection_probability=prob_detection,
    meas_range=clutter_area,
    clutter_rate=clutter_rate)


# Instantiate a set for detections/clutter and ground truths
detections = set()
ground_truth = set()

# Duplicate the detection simulator
plot, trackx = tee(detection_sim, 2)

# Iterate in the detection simulator to generate the measurements
for time, dets in plot:
    detections |= dets
    ground_truth |= ground_truth_simulator.groundtruth_paths

# Visualise the detections and tracks
from stonesoup.plotter import Plotterly

plotter = Plotterly()
plotter.plot_ground_truths(ground_truth, [0, 2])
plotter.plot_measurements(detections, [0, 2])

plotter.fig

# %%
# 2. Instantiate the tracking components and tracker;
# ---------------------------------------------------
# We need to prepare the tracker and its components. In this
# example we consider a Unscented kalman filter since we are
# dealing with non-linear measurements. We consider a
# :class:`~.UnscentedKalmanPredictor` and :class:`~.UnscentedKalmanUpdater`.
# As said previously, we consider a Multi-frame assignment data associator
# which wraps a :class:`~.PDAHypothesiser` probability hypothesiser into a
# :class:`~.MFAHypothesiser` to work with the :class:`~.MFADataAssociator`.
# To instantiate the tracks we can use :class:`~.GaussianMixtureInitiator` which
# job is to start the tracks with GaussianMixture models.
#

from stonesoup.predictor.kalman import UnscentedKalmanPredictor
from stonesoup.updater.kalman import UnscentedKalmanUpdater

predictor = UnscentedKalmanPredictor(transition_model)
updater = UnscentedKalmanUpdater(measurement_model)

from stonesoup.dataassociator.mfa import MFADataAssociator
from stonesoup.hypothesiser.mfa import MFAHypothesiser
from stonesoup.hypothesiser.probability import PDAHypothesiser

hypothesiser = PDAHypothesiser(predictor,
                               updater,
                               clutter_spatial_density,
                               prob_gate=0.9999,
                               prob_detect=prob_detection)

hypothesiser = MFAHypothesiser(hypothesiser)
data_associator = MFADataAssociator(hypothesiser,
                                    slide_window=3)

from stonesoup.deleter.time import UpdateTimeDeleter
deleter = UpdateTimeDeleter(timedelta(seconds=4),
                            delete_last_pred=False)

from stonesoup.initiator.simple import MultiMeasurementInitiator
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.dataassociator.neighbour import GNNWith2DAssignment
from stonesoup.measures import Mahalanobis
from stonesoup.types.state import TaggedWeightedGaussianState
from stonesoup.types.track import Track
from stonesoup.types.mixture import GaussianMixture
from stonesoup.types.numeric import Probability

from stonesoup.hypothesiser.gaussianmixture import GaussianMixtureHypothesiser
from stonesoup.initiator.simple import GaussianMixtureInitiator, SimpleMeasurementInitiator

# base_hypothesiser = DistanceHypothesiser(
#     predictor=predictor,
#     updater=updater,
#     measure=Mahalanobis(),
#     missed_distance=10,
#     include_all=False
# )
#
# hypothesiser_init = GaussianMixtureHypothesiser(
#     base_hypothesiser
# )

# GaussianMixture([TaggedWeightedGaussianState(initial_state.state_vector,
#                                          initial_state.covar,
#                                          timestamp=initial_state.timestamp,
#                                          weight=Probability(2),
#                                          tag=TaggedWeightedGaussianState.BIRTH,),
# ]),


initiator_p = MultiMeasurementInitiator(
    GaussianMixture([TaggedWeightedGaussianState(initial_state.state_vector,
                                initial_state.covar,
                                weight=Probability(1),
                                tag=[])]),
    measurement_model=None,
    deleter=deleter,
    data_associator=GNNWith2DAssignment(
        DistanceHypothesiser(predictor, updater, Mahalanobis(), missed_distance=5)),
    updater=updater,
    min_points=3
)

s_prior_state = GaussianMixture([TaggedWeightedGaussianState(initial_state.state_vector,
                                                 initial_state.covar,
                                                 weight=Probability(0.1), tag=[])])
initiator = GaussianMixtureInitiator(
    SimpleMeasurementInitiator(
        prior_state=s_prior_state,
        measurement_model=measurement_model
    )
)

# from stonesoup.initiator.simple import SimpleMeasurementInitiator
# birth_component = GaussianMixture([TaggedWeightedGaussianState(
#     state_vector=initial_state.state_vector,
#     covar=initial_state.covar,
#     weight=0.5,
#     tag=TaggedWeightedGaussianState.BIRTH,
#     timestamp=start_time
# )])

# initiator = SimpleMeasurementInitiator(birth_component,
#                                        measurement_model=measurement_model)


from stonesoup.tracker.simple import MultiTargetTracker, MultiTargetMixtureTracker

tracker = MultiTargetTracker(  # Runs the tracker
    initiator=initiator, #GaussianMixtureInitiator(initiator_p),
    deleter=deleter,
    detector=detection_sim,
    data_associator=data_associator,
    updater=updater)
from stonesoup.types.update import GaussianMixtureUpdate
prior1 = GaussianMixture([TaggedWeightedGaussianState(initial_state.state_vector,
                                                      initial_state.covar*1.5,
                                                      timestamp=initial_state.timestamp,
                                                      weight=Probability(0.5),
                                                             tag=[])])
prior2 = deepcopy(prior1)

tracks = {Track([prior1]), Track([prior2])} #, Track([prior1]), Track([prior1])}
#for (time, current_tracks) in tracker:
#    print(time)
#    tracks.update(current_tracks)

for time, detection in trackx: #enumerate(detections):  # loop over that
    association = data_associator.associate(tracks, detection, time)

    for track, hypotheses in association.items():
        components = []
        for hypothesis in hypotheses:
            if not hypothesis:
                components.append(hypothesis.prediction)
            else:
                update = updater.update(hypothesis)
                components.append(update)
        track.append(GaussianMixtureUpdate(components=components,
                                           hypothesis=hypotheses))

    tracks.update(track)

plotter.plot_tracks(tracks, [0, 2], track_label="EKF", line=dict(color="orange"))
plotter.fig.show()

# %%
# Conclusion
# ----------
# In this example we have presented how to set up a Multi-hypotheses tracking
# (MHT) simulation, by employing the existing components present in Stone Soup
# and perform the tracking in a cluttered multi-target scenario.

# References
# ----------
# .. [#] Xia, Y., Granström, K., Svensson, L., García-Fernández, Á.F., and Williams, J.L.,
#        2019. Multiscan Implementation of the Trajectory Poisson Multi-Bernoulli Mixture Filter.
#        J. Adv. Information Fusion, 14(2), pp. 213–235.

