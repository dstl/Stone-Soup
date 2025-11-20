#!/usr/bin/env python
# coding: utf-8

"""
==========================================================
Comparing different filters in the context of track fusion
==========================================================
"""

# %%
# This example shows a comparison between a Kalman filter algorithm and particle filter in the
# context of track fusion. This example is relevant to show how to get a unique track from partial
# tracks generated from set of different measurements obtained from independent sensors.
#
# This example simulates the case of a single target moving in a 2D Cartesian space with
# measurements obtained from two identical, for simplicity, radars with their own trackers.
# We present a comparison of the resulting composite track obtained by the two different
# partial tracks generated. Furthermore, we measure the track-to-truth accuracy of
# the final track obtained by different algorithms.
#
# This example follows this structure:
#
# 1. Initialise the sensors and the target trajectory;
# 2. Initialise the filters components and create the trackers;
# 3. Run the trackers, generate the partial tracks and merge the composite track;
# 4. Evaluate the obtained tracks with the groundtruth trajectory.
#

# %%
# 1. Initialise the sensors and the target trajectory
# ---------------------------------------------------
# We start by creating two identical, in terms of performance, radars using  the
# :class:`~.RadarBearingRange` sensor placed on two separate sensor platforms of type
# :class:`~.FixedPlatform`.
# For the target we simulate a single object moving on a straight trajectory.
# This example is set up such that it is easy to understand how the algorithm components work.

# %%
# General imports
# ^^^^^^^^^^^^^^^

import numpy as np
from datetime import datetime, timedelta
from copy import deepcopy
from itertools import tee

# %%
# Stone Soup general imports
# ^^^^^^^^^^^^^^^^^^^^^^^^^^

from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
    ConstantVelocity
from stonesoup.types.state import GaussianState
from stonesoup.types.array import CovarianceMatrix
from stonesoup.simulator.simple import SingleTargetGroundTruthSimulator
from stonesoup.models.clutter.clutter import ClutterModel

# Instantiate the radars to collect measurements - Use a :class:`~.RadarBearingRange` radar.#
from stonesoup.sensor.radar.radar import RadarBearingRange

# Import the platform to place the sensors on
from stonesoup.platform.base import FixedPlatform

# Load the platform detection simulator
from stonesoup.simulator.platform import PlatformDetectionSimulator

# %%
# Simulation parameters setup
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^

start_time = datetime.now().replace(microsecond=0)
number_of_steps = 75  # Number of timestep of the simulation
np.random.seed(2000)  # Random seed for reproducibility
n_particles = 2**10

# Instantiate the target transition model
gnd_transition_model = CombinedLinearGaussianTransitionModel([
    ConstantVelocity(0.00), ConstantVelocity(0.00)])

# the transition model needs to have little more process noise for the Particle filter
transition_model = CombinedLinearGaussianTransitionModel([
    ConstantVelocity(0.5), ConstantVelocity(0.5)])

# Define the initial target state
initial_target_state = GaussianState([25, 0.5, 75, -0.25],
                                     np.diag([10, 1, 10, 1]),
                                     timestamp=start_time)

# Set up the ground truth simulation
groundtruth_simulation = SingleTargetGroundTruthSimulator(
    transition_model=gnd_transition_model,
    initial_state=initial_target_state,
    timestep=timedelta(seconds=1),
    number_steps=number_of_steps)

# %%
# Load a clutter model
# ^^^^^^^^^^^^^^^^^^^^

clutter_model = ClutterModel(
    clutter_rate=0.75,
    distribution=np.random.default_rng().uniform,
    dist_params=((0, 120), (-5, 105)))
# dist_params describe the area where the clutter is detected

# Define the clutter area and spatial density for the tracker
clutter_area = np.prod(np.diff(clutter_model.dist_params))
clutter_spatial_density = clutter_model.clutter_rate/clutter_area

# %%
# Instantiate the sensors, platforms and simulators
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# Let's assume that both radars have the same noise covariance for simplicity
# These radars will have a variance of 0.005 degrees in bearing and 5 meters in range
radar_noise = CovarianceMatrix(np.diag([np.deg2rad(0.05), 5]))

# Define the specifications of the two radars
radar1 = RadarBearingRange(
    ndim_state=4,
    position_mapping=(0, 2),
    noise_covar=radar_noise,
    clutter_model=clutter_model,
    max_range=3000)  # max_range can be removed and use the default value

# deep copy the first radar specs. Changes in the first one does not influence the second
radar2 = deepcopy(radar1)

# prepare the first sensor platform and add the sensor
sensor1_platform = FixedPlatform(
    states=GaussianState([10, 0, 80, 0],
                         np.diag([1, 0, 1, 0])),
    position_mapping=(0, 2),
    sensors=[radar1])

# Instantiate the second sensor platform with the second sensor
sensor2_platform = FixedPlatform(
    states=GaussianState([75, 0, 10, 0],
                         np.diag([1, 0, 1, 0])),
    position_mapping=(0, 2),
    sensors=[radar2])

# create copy of the simulators, in this way all trackers will receive the same detections
gt_sims = tee(groundtruth_simulation, 2)

# create the radar simulators
radar_simulator1 = PlatformDetectionSimulator(
    groundtruth=gt_sims[0],
    platforms=[sensor1_platform])

radar_simulator2 = PlatformDetectionSimulator(
    groundtruth=gt_sims[1],
    platforms=[sensor2_platform])

radar1plot, radar1KF, radar1PF = tee(radar_simulator1, 3)
radar2plot, radar2KF, radar2PF = tee(radar_simulator2, 3)

# %%
# Visualise the detections from the sensors
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Before preparing the different trackers components, we can visualise the target ground truth and
# its detections from the two sensors.
# In this way we can appreciate how the measurements are different and can lead to separate tracks.

from stonesoup.plotter import Plotterly
from stonesoup.metricgenerator.tracktotruthmetrics import SIAPMetrics
from stonesoup.measures import Euclidean
from stonesoup.plotter import MetricPlotter
from stonesoup.dataassociator.tracktotrack import TrackToTruth
from stonesoup.metricgenerator.manager import MultiManager

# Lists to hold the detections from each sensor
s1_detections = []
s2_detections = []

groundtruth_generator = groundtruth_simulation.groundtruth_paths_gen()

truths = set()
# Iterate over the time steps, extracting the detections and truths
for (time, sd1), (_, sd2) in zip(radar1plot, radar2plot):
    s1_detections.append(sd1)
    s2_detections.append(sd2)
    truths.update(next(groundtruth_generator)[1])  # consider only the path

# Plot the detections from the two radars
plotter = Plotterly()
plotter.plot_measurements(s1_detections, [0, 2], marker=dict(color='red'),
                          label='Sensor 1 measurements')
plotter.plot_measurements(s2_detections, [0, 2], marker=dict(color='blue'),
                          label='Sensor 2 measurements')
plotter.plot_sensors({sensor1_platform, sensor2_platform}, [0, 1],
                     marker=dict(color='black', symbol='1', size=10))
plotter.plot_ground_truths(truths, [0, 2])
plotter.fig

# %%
# 2. Initialise the trackers components
# -------------------------------------
# We have initialised the sensors and the target trajectory, we can see that the
# detections from the two sensors differ from one another, that will lead to two separate tracks.
#
# Now, we initialise the components of the two trackers, one using an Extended Kalman filter
# and a particle filter.
#

# %%
# Stone Soup tracker components
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#

# Let's consider a distance based hypothesiser
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.measures import Mahalanobis
from stonesoup.dataassociator.neighbour import GNNWith2DAssignment
from stonesoup.deleter.time import UpdateTimeStepsDeleter

# Load the kalman filter components
from stonesoup.updater.kalman import UnscentedKalmanUpdater
from stonesoup.predictor.kalman import UnscentedKalmanPredictor
from stonesoup.initiator.simple import SimpleMeasurementInitiator, GaussianParticleInitiator

# Load a single target tracker
from stonesoup.tracker.simple import SingleTargetTracker

# prepare the particle components
from stonesoup.predictor.particle import ParticlePredictor
from stonesoup.updater.particle import ParticleUpdater
from stonesoup.resampler.particle import ESSResampler

# instantiate the Extended Kalman filter predictor and updater
KF_predictor = UnscentedKalmanPredictor(transition_model)
KF_updater = UnscentedKalmanUpdater(measurement_model=None)

# define the hypothesiser
hypothesiser_KF = DistanceHypothesiser(
    predictor=KF_predictor,
    updater=KF_updater,
    measure=Mahalanobis(),
    missed_distance=10)

# define the distance data associator
data_associator_KF = GNNWith2DAssignment(hypothesiser_KF)

# define a track time deleter
deleter = UpdateTimeStepsDeleter(3)

# create a track initiator placed on the target track origin
initiator = SimpleMeasurementInitiator(
    prior_state=initial_target_state,
    measurement_model=None)

# Instantiate the predictor, particle resampler and particle filter updater
PF_predictor = ParticlePredictor(transition_model)
resampler = ESSResampler()
PF_updater = ParticleUpdater(measurement_model=None,
                             resampler=resampler)

hypothesiser_PF = DistanceHypothesiser(
    predictor=PF_predictor,
    updater=PF_updater,
    measure=Mahalanobis(),
    missed_distance=9)

# define the data associator
data_associator_PF = GNNWith2DAssignment(hypothesiser_PF)

# Initiator with a prior Gaussian state with the target track origin
prior_state = SimpleMeasurementInitiator(
    prior_state=GaussianState([25, 0.5, 70, -0.25],
                              np.diag([5, 2, 5, 2])**2))

# Particle filter initiator
PF_initiator = GaussianParticleInitiator(
    initiator=prior_state,
    number_particles=n_particles)

# %%
# Stone soup track fusion components
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# At this stage we have all the components needed to perform the tracking using both Kalman and
# particle filters.
#
# To perform the track fusion, we employ the covariance intersection algorithm implemented in the
# :class:`~.ChernoffUpdater` class, treating the tracks as :class:`~.GaussianMixture` detections.


# Load the ChernoffUpdater components for track fusion
from stonesoup.updater.chernoff import ChernoffUpdater
from stonesoup.updater.pointprocess import PHDUpdater
from stonesoup.mixturereducer.gaussianmixture import GaussianMixtureReducer
from stonesoup.hypothesiser.gaussianmixture import GaussianMixtureHypothesiser
from stonesoup.tracker.pointprocess import PointProcessMultiTargetTracker
from stonesoup.types.state import TaggedWeightedGaussianState
from stonesoup.feeder.track import Tracks2GaussianDetectionFeeder
from stonesoup.feeder.multi import MultiDataFeeder

# create the Chernoff Updater
ch_updater = ChernoffUpdater(measurement_model=None)

# Instantiate the PHD Updater, including the probability of
# detection and probability of survival
updater = PHDUpdater(updater=ch_updater,
                     clutter_spatial_density=clutter_spatial_density,
                     prob_detection=0.99,
                     prob_survival=0.95)

# Create a base hypothesiser using the Chernoff updater and
# the kalman predictor
base_hypothesiser = DistanceHypothesiser(
    predictor=KF_predictor,
    updater=ch_updater,
    measure=Mahalanobis(),
    missed_distance=15)

# Instantiate the Gaussian Mixture hypothesiser
hypothesiser = GaussianMixtureHypothesiser(base_hypothesiser,
                                           order_by_detection=True)

# Gaussian mixture reducer to prune and merge the various tracks
ch_reducer = GaussianMixtureReducer(
    prune_threshold=1e-10,
    pruning=True,
    merge_threshold=100,
    merging=True)

# Create the covariance for the birth of the tracks,
# large on the x,y location and smaller on the velocities
birth_covar = CovarianceMatrix(np.diag([50, 5, 50, 5]))

# Define the Chernoff updated birth components for the tracks
ch_birth_component = TaggedWeightedGaussianState(
    state_vector=[25, 0.5, 70, -0.25],  # Initial target state
    covar=birth_covar**2,
    weight=1,
    tag=TaggedWeightedGaussianState.BIRTH,
    timestamp=start_time)

# Instantiate the Track fusion tracker using the Point Process Tracker
track_fusion_tracker = PointProcessMultiTargetTracker(
    detector=None,
    hypothesiser=hypothesiser,
    updater=updater,
    reducer=deepcopy(ch_reducer),
    birth_component=deepcopy(ch_birth_component),
    extraction_threshold=0.95)

# Copy the track fusion tracker
track_fusion_tracker2 = deepcopy(track_fusion_tracker)

# %%
# 3. Run the trackers, generate the partial tracks and merge the final composite track
# ------------------------------------------------------------------------------------
# So far, we have shown how to instantiate the various tracker components as well as the fusion
# tracker. Now, we run the trackers to generate the tracks, and perform the track fusion.

# Create the tracks for the particle filters, Kalman and merged ones
PF_track1, PF_track2, KF_track1, KF_track2 = set(), set(), set(), set()
PF_fused_track, KF_fused_track = set(), set()

# we assign a unique tracker the detections from a specific radar simulator
KF_tracker_1 = SingleTargetTracker(initiator=initiator,
                                   detector=radar1KF,
                                   updater=KF_updater,
                                   data_associator=data_associator_KF,
                                   deleter=deleter)

KF_tracker_2 = SingleTargetTracker(initiator=initiator,
                                   detector=radar2KF,
                                   updater=KF_updater,
                                   data_associator=data_associator_KF,
                                   deleter=deleter)

PF_tracker_1 = SingleTargetTracker(initiator=PF_initiator,
                                   detector=radar1PF,
                                   updater=PF_updater,
                                   data_associator=data_associator_PF,
                                   deleter=deleter)

PF_tracker_2 = SingleTargetTracker(initiator=PF_initiator,
                                   detector=radar2PF,
                                   updater=PF_updater,
                                   data_associator=data_associator_PF,
                                   deleter=deleter)

PartialTrackPF1, TrackFusionPF1 = tee(PF_tracker_1, 2)
PartialTrackPF2, TrackFusionPF2 = tee(PF_tracker_2, 2)
PartialTrackKF1, TrackFusionKF1 = tee(KF_tracker_1, 2)
PartialTrackKF2, TrackFusionKF2 = tee(KF_tracker_2, 2)

# Create the detector feeding the tracker algorithms
track_fusion_tracker.detector = Tracks2GaussianDetectionFeeder(
    MultiDataFeeder([TrackFusionKF1, TrackFusionKF2]))

track_fusion_tracker2.detector = Tracks2GaussianDetectionFeeder(
    MultiDataFeeder([TrackFusionPF1, TrackFusionPF2]))

# Create an iterator for the trackers
iter_fusion_tracker = iter(track_fusion_tracker)
iter_fusion_tracker2 = iter(track_fusion_tracker2)

PartialTrackPF1_iter = iter(PartialTrackPF1)
PartialTrackPF2_iter = iter(PartialTrackPF2)
PartialTrackKF1_iter = iter(PartialTrackKF1)
PartialTrackKF2_iter = iter(PartialTrackKF2)

# Loop over the timestep doing the track fusion and partial tracks
for _ in range(number_of_steps):
    for _ in range(2):
        _, tracks = next(iter_fusion_tracker)
        KF_fused_track.update(tracks)
        del tracks
        _, tracks = next(iter_fusion_tracker2)
        PF_fused_track.update(tracks)

    _, PF_sensor_track1 = next(PartialTrackPF1_iter)
    PF_track1.update(PF_sensor_track1)

    _, PF_sensor_track2 = next(PartialTrackPF2_iter)
    PF_track2.update(PF_sensor_track2)

    _, KF_sensor_track1 = next(PartialTrackKF1_iter)
    KF_track1.update(KF_sensor_track1)

    _, KF_sensor_track2 = next(PartialTrackKF2_iter)
    KF_track2.update(KF_sensor_track2)

truths = set(groundtruth_simulation.groundtruth_paths)

# %%
# Let's visualise the various tracks and detections in the cases
# using the Kalman and particle filters.

plotter.plot_tracks(PF_track1, [0, 2], line=dict(color="orange"), label='PF partial track 1')
plotter.plot_tracks(PF_track2, [0, 2], line=dict(color="gold"), label='PF partial track 2')
plotter.plot_tracks(PF_fused_track, [0, 2], line=dict(color="red"), label='PF fused track')
plotter.plot_tracks(KF_fused_track, [0, 2], line=dict(color="blue"), label='KF fused track')
plotter.plot_tracks(KF_track1, [0, 2], line=dict(color="cyan"), label='KF partial track 1')
plotter.plot_tracks(KF_track2, [0, 2], line=dict(color="skyblue"), label='KF partial track 2')

plotter.fig

# %%
# 4. Evaluate the obtained tracks with the groundtruth trajectory.
# ----------------------------------------------------------------
# At this stage we have almost completed our example. We have created the detections from the
# radars, performed the tracking and the fusion of the tracks. We now employ the
# :class:`~.MetricManager` to generate summary statistics on the accuracy of the tracks by comparing
# them with the ground truth trajectory.
#
# If we consider the SIAP metrics, we can appreciate that the fused tracks have a lower error
# compared to the partial tracks obtained with the single instruments.
#

# Load the SIAP metric
siap_kf_truth = SIAPMetrics(position_measure=Euclidean((0, 2)),
                            velocity_measure=Euclidean((1, 3)),
                            generator_name='SIAP_KF_fused-truth',
                            tracks_key='KF_fused_tracks',
                            truths_key='truths'
                            )
siap_kf1_truth = SIAPMetrics(position_measure=Euclidean((0, 2)),
                             velocity_measure=Euclidean((1, 3)),
                             generator_name='SIAP_KF1-truth',
                             tracks_key='KF_1_tracks',
                             truths_key='truths'
                             )
siap_kf2_truth = SIAPMetrics(position_measure=Euclidean((0, 2)),
                             velocity_measure=Euclidean((1, 3)),
                             generator_name='SIAP_KF2-truth',
                             tracks_key='KF_2_tracks',
                             truths_key='truths'
                             )

siap_pf_truth = SIAPMetrics(position_measure=Euclidean((0, 2)),
                            velocity_measure=Euclidean((1, 3)),
                            generator_name='SIAP_PF_fused-truth',
                            tracks_key='PF_fused_tracks',
                            truths_key='truths'
                            )
siap_pf1_truth = SIAPMetrics(position_measure=Euclidean((0, 2)),
                             velocity_measure=Euclidean((1, 3)),
                             generator_name='SIAP_PF1-truth',
                             tracks_key='PF_1_tracks',
                             truths_key='truths'
                             )

siap_pf2_truth = SIAPMetrics(position_measure=Euclidean((0, 2)),
                             velocity_measure=Euclidean((1, 3)),
                             generator_name='SIAP_PF2-truth',
                             tracks_key='PF_2_tracks',
                             truths_key='truths'
                             )

# Define a data associator between the tracks and the truths
associator = TrackToTruth(association_threshold=30)

# Create the metric manager
metric_manager = MultiManager([siap_kf_truth,
                               siap_kf1_truth,
                               siap_kf2_truth,
                               siap_pf_truth,
                               siap_pf1_truth,
                               siap_pf2_truth],
                              associator)

# Add the tracks to the metric manager
metric_manager.add_data({'KF_1_tracks': KF_track1,
                         'KF_2_tracks': KF_track2,
                         'PF_1_tracks': PF_track1,
                         'PF_2_tracks': PF_track2,
                         'KF_fused_tracks': KF_fused_track,
                         'PF_fused_tracks': PF_fused_track,
                         'truths': truths}, overwrite=False)

# Loaded the plotter for the various metrics.
metrics = metric_manager.generate_metrics()
graph = MetricPlotter()

graph.plot_metrics(metrics, generator_names=['SIAP_KF_fused-truth',
                                             'SIAP_PF_fused-truth',
                                             'SIAP_KF1-truth',
                                             'SIAP_KF2-truth',
                                             'SIAP_PF1-truth',
                                             'SIAP_PF2-truth'],
                   color=['blue', 'red', 'cyan', 'azure', 'orange', 'gold'])
graph.fig

# %%
# Conclusions
# -----------
# In this example we have shown how it is possible to merge the tracks generated by independent
# trackers ran on sets of detections obtained from separate sensors. Finally, we compared how the
# Kalman and the particle filters behave in these cases, making track-to-truth comparisons.

# sphinx_gallery_thumbnail_number = 2
