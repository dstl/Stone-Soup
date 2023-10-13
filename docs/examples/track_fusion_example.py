#!/usr/bin/env python
# coding: utf-8

"""
==========================================================
Comparing different filters in the context of track fusion
==========================================================
"""

# %%
# This example shows a comparison between a Kalman filter and
# a particle filter in the context of track fusion. This example is
# relevant to show how to get an unique track
# from partial tracks generated from a set of
# different measurements obtained from independent sensors.
# This example simulates the case of a single target moving in
# a 2D cartesian space with measurements obtained from two
# identical, for simplicity, radars and trackers. We
# compare the resulting composite track obtained by the
# two different tracks generated. Furthermore, we
# measure the performances of the tracks obtained
# by two different Kalman and particle filters.
#
# This example follows this structure:
# 1) Initialise the sensors and the target trajectory;
# 2) Initialise the filters components and create the tracker;
# 3) Run the trackers, generate the partial tracks and the final
# composite track;
# 4) Compare the obtained tracks with the groundtruth trajectory.
#

# %%
# 1) Initialise the sensors and the target trajectory
# ---------------------------------------------------
# We start creating two identical, in terms of performances,
# radars using :class:`~.RadarBearingRange` placed on two
# separate :class:`~.FixedPlatform`. For the target we
# simulate a single object moving on a straight trajectory.
# The example setup is simple to it is easier to understand
# how the Stone soup components are working.

# %%
# General imports
# ^^^^^^^^^^^^^^^

import numpy as np
from datetime import datetime
from datetime import timedelta
from copy import deepcopy

# %%
# Stone Soup general imports
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
#
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
    ConstantVelocity
from stonesoup.types.state import GaussianState
from stonesoup.types.array import CovarianceMatrix
from stonesoup.simulator.simple import SingleTargetGroundTruthSimulator

# %%
# Simulation parameters setup
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
start_time = datetime.now().replace(microsecond=0) # For simplicity fix a date
number_of_steps = 75  # Number of timestep for the simulation
np.random.seed(1908)  # Random seed for reproducibility
n_particles= 2**8

# Instantiate the target transition model
gnd_transition_model = CombinedLinearGaussianTransitionModel([
    ConstantVelocity(0.00), ConstantVelocity(0.00)])

# the transition model needs to have little more noise for the PF
transition_model = CombinedLinearGaussianTransitionModel([
    ConstantVelocity(0.2), ConstantVelocity(0.2)])

# Define the initial target state
initial_target_state = GaussianState([25, 0.5, 75, -0.25],
                                     np.diag([5, 1, 5, 1]) ** 2,
                                     timestamp=start_time)


# Set up the ground truth simulation
groundtruth_simulation = SingleTargetGroundTruthSimulator(
    transition_model=gnd_transition_model,
    initial_state=initial_target_state,
    timestep=timedelta(seconds=2),
    number_steps=number_of_steps)


# %%
# Load a clutter model
# ^^^^^^^^^^^^^^^^^^^^
from stonesoup.models.clutter.clutter import ClutterModel

clutter_model = ClutterModel(
    clutter_rate=0.6,
    distribution=np.random.default_rng().uniform,
    dist_params=((0,120), (-5,105)))
# dist_params describe the area where the clutter is detected

# Define the clutter area and spatial density for the tracker
clutter_area = np.prod(np.diff(clutter_model.dist_params))
clutter_spatial_density = clutter_model.clutter_rate/clutter_area

# %%
# Instantiate the sensors, platforms and simulators
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Instantiate the radars to collect measurements - Use a BearingRange radar
from stonesoup.sensor.radar.radar import RadarBearingRange
# Import the platform to place the sensors on
from stonesoup.platform.base import FixedPlatform
# Load the platform detection simulator - Let's use a simulator for each radar
# Instantiate the simulators
from stonesoup.simulator.platform import PlatformDetectionSimulator

# Let's assume that both radars have the same noise covariance for simplicity
# These radars will have the +/-0.005 degrees accuracy in bearing and +/- 2 meters in range
radar_noise = CovarianceMatrix(np.diag([np.deg2rad(0.005), 2.0**2]))

# Define the specifications of the two radars
radar1 = RadarBearingRange(
    ndim_state= 4,
    position_mapping= (0, 2),
    noise_covar= radar_noise,
    clutter_model= clutter_model,
    max_range= 3000)  # max_range can be removed and use the default value

# deep copy the first radar specs. Changes in the first one does not influence the second
radar2 = deepcopy(radar1)

# Instantiate the first sensor platform and add the sensor
sensor1_platform = FixedPlatform(
    states=GaussianState([10, 0, 80, 0],
                         np.diag([1, 0, 1, 0])),
    position_mapping= (0, 2),
    sensors= [radar1])

# Instantiate the second one
sensor2_platform = FixedPlatform(
    states=GaussianState([75, 0, 10, 0],
                         np.diag([1, 0, 1, 0])),
    position_mapping= (0, 2),
    sensors= [radar2])

# create the radar simulators
radar_simulator1 = PlatformDetectionSimulator(
    groundtruth= groundtruth_simulation,
    platforms= [sensor1_platform])

radar_simulator2 = PlatformDetectionSimulator(
    groundtruth= groundtruth_simulation,
    platforms= [sensor2_platform])

# create copy of the simulators
from itertools import tee
_, *r1simulators = tee(radar_simulator1, 3)
_, *r2simulators = tee(radar_simulator2, 3)



# %%
# Visualise the detections from the sensors
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Before creating the different trackers components
# let's visualise the target and its detections from the
# two sensors. In this way we can appreciate
# how the measurements are different and can lead to two separate
# tracks.
#

# Load the stone soup plotter
from stonesoup.plotter import Plotterly

# Lists to hold the detections from each sensor
s1_detections = []
s2_detections = []

detections1 = deepcopy(radar_simulator1)
detections2 = deepcopy(radar_simulator2)

# # Extract the generator function for the detections
g1 = detections1.detections_gen()
g2 = detections2.detections_gen()


# Iterate over the time steps, extracting the detections and truths
for _ in range(number_of_steps):
    s1_detections.append(next(g1)[1])
    s2_detections.append(next(g2)[1])


# Plot the detections from the two radars
plotter = Plotterly()
plotter.plot_measurements(s1_detections, [0, 2], marker= dict(color='orange', symbol='305'),
                          measurements_label='Sensor 1 measurements')
plotter.plot_measurements(s2_detections, [0, 2], marker= dict(color='blue', symbol='0'),
                          measurements_label='Sensor 2 measurements')
plotter.plot_sensors({sensor1_platform,sensor2_platform}, [0,1], marker= dict(color='black', symbol= '1',
                                                                              size=10))
plotter.fig


# %%
# 2) Initialise the trackers components
# -------------------------------------
# We have initialised the sensors and the
# target path, we can see that the
# detections from the two sensors
# are slightly different one from the
# other, that will lead to two separate
# tracks. Now, we initialise the two trackers
# components, one using a extended Kalman filter
# and a particle filter.
#

# %%
# Stone soup tracker components
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#

# Let's consider a Distance based hypothesiser
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.measures import Mahalanobis
from stonesoup.dataassociator.neighbour import GNNWith2DAssignment
from stonesoup.deleter.time import UpdateTimeStepsDeleter

# Load the kalman filter components
from stonesoup.updater.kalman import ExtendedKalmanUpdater
from stonesoup.predictor.kalman import ExtendedKalmanPredictor
from stonesoup.initiator.simple import SimpleMeasurementInitiator, GaussianParticleInitiator

# Load a single target tracker
from stonesoup.tracker.simple import SingleTargetTracker

# prepare the particle components
from stonesoup.predictor.particle import ParticlePredictor
from stonesoup.updater.particle import ParticleUpdater
from stonesoup.resampler.particle import ESSResampler

# Lets define an helper function to minimise the number of times
# we have to initialise the same tracker object
def general_tracker(tracker_class, detector,
                    filter_updater, initiator, deleter,
                    data_associator):
    """
    Helper function to initialise the trackers
    """

    tracker = tracker_class(
        initiator= initiator,
        detector= detector,
        updater= filter_updater,
        data_associator= data_associator,
        deleter=deleter)
    return tracker


# instantiate the Kalman filter predictor
KF_predictor = ExtendedKalmanPredictor(transition_model)

# Instantiate the Kalman filter updater
KF_updater = ExtendedKalmanUpdater(measurement_model=None)

# create an track initiator placed on the target track origin
initiator = SimpleMeasurementInitiator(
    prior_state=initial_target_state,
    measurement_model=None)

# define the hypothesiser
hypothesiser_KF = DistanceHypothesiser(
    predictor=KF_predictor,
    updater=KF_updater,
    measure=Mahalanobis(),
    missed_distance=5
)

# define the distance data associator
data_associator_KF = GNNWith2DAssignment(hypothesiser_KF)

# define a track time deleter
deleter = UpdateTimeStepsDeleter(5)

# Instantiate the predictor, particle resampler and particle
# filter updater
PF_predictor = ParticlePredictor(transition_model)
resampler = ESSResampler()
PF_updater = ParticleUpdater(measurement_model=None,
                             resampler=resampler)

hypothesiser_PF = DistanceHypothesiser(
    predictor=PF_predictor,
    updater=PF_updater,
    measure=Mahalanobis(),
    missed_distance=10)

# define the data associator
data_associator_PF = GNNWith2DAssignment(hypothesiser_PF)

# To instantiate the track initiator we define a prior state
# as gaussian state with the target track origin
prior_state=  SimpleMeasurementInitiator(
    prior_state=GaussianState([25, 0.5, 70, -0.25],
                np.diag([10, 2, 10, 2]) ** 2))

# Particle filter initiator
PF_initiator = GaussianParticleInitiator(
    initiator=prior_state,
    number_particles=n_particles)

# %%
# At this stage we have all the components needed to
# perform the tracking using both Kalman and Particle
# filters. We need to create a way to perform the track fusion.
# To perform such fusion, we employ the covariance
# intersection algorithm
# adopting the :class:`~.ChernoffUpdater` class, and
# treating the tracks as measurements and we consider
# the measurements as GaussianMixture objects.

# Instantiate a dummy detector to read the detections
from stonesoup.buffered_generator import BufferedGenerator
from stonesoup.reader.base import DetectionReader

class DummyDetector(DetectionReader):
    def __init__(self, *args, **kwargs):
        self.current = kwargs['current']

    @BufferedGenerator.generator_method
    def detections_gen(self):
        yield self.current


# %%
# Stone soup track fusion components
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#

# Load the ChernoffUpdater components for track fusion
from stonesoup.updater.chernoff import ChernoffUpdater
from stonesoup.updater.pointprocess import PHDUpdater
from stonesoup.mixturereducer.gaussianmixture import GaussianMixtureReducer
from stonesoup.hypothesiser.gaussianmixture import GaussianMixtureHypothesiser
from stonesoup.tracker.pointprocess import PointProcessMultiTargetTracker
from stonesoup.types.state import TaggedWeightedGaussianState
from stonesoup.feeder.track import Tracks2GaussianDetectionFeeder

# create the Chernoff Updater
ch_updater = ChernoffUpdater(measurement_model=None)

# Instantiate the PHD Updater, including the probability of
# detection and probability of survival
updater = PHDUpdater(updater=ch_updater,
                     clutter_spatial_density=clutter_spatial_density,
                     prob_detection=0.9,
                     prob_survival=0.9)

# Create a base hypothesiser using the Chernoff updater and
# the kalman predictor
base_hypothesiser = DistanceHypothesiser(
    predictor=KF_predictor,
    updater=ch_updater,
    measure=Mahalanobis(),
    missed_distance=100)

# Instantiate the Gaussian Mixture hypothesiser
hypothesiser= GaussianMixtureHypothesiser(base_hypothesiser,
                                          order_by_detection=True)

# Gaussian mixture reducer to prune and merge the various tracks
ch_reducer = GaussianMixtureReducer(
    prune_threshold=1e-10,
    pruning=True,
    merge_threshold=100,
    merging=True)

# Create the covariance for the birth of the tracks,
# large on the x,y location and smaller on the velocities
birth_covar = CovarianceMatrix(np.diag([100, 1, 100, 1]))

# Define the Chernoff updated birth components for the tracks
ch_birth_component = TaggedWeightedGaussianState(
    state_vector=[25, 0.5, 70, -0.25],  # Initial target state
    covar=birth_covar**2,
    weight=0.5,
    tag =TaggedWeightedGaussianState.BIRTH,
    timestamp=start_time)

# Instantiate the Track fusion tracker using Point Process Tracker
track_fusion_tracker = PointProcessMultiTargetTracker(
    detector=None,
    hypothesiser=hypothesiser,
    updater=updater,
    reducer=deepcopy(ch_reducer),
    birth_component=deepcopy(ch_birth_component),
    extraction_threshold=0.9)

track_fusion_tracker2 = deepcopy(track_fusion_tracker)
# %%
# 3) Run the trackers, generate the partial tracks and the final composite track;
# -------------------------------------------------------------------------------
# So far, we have shown how to instantiate the various tracker components
# as well as the track fusion tracker. Now, we run the trackers to generate
# the tracks and we perform the track fusion. Furthermore, we want to measure
# how good are the track fusions in comparisons to the groundtruths, so we
# instantiate a metric manager to measure the various distances.
#

# Instantiate the metric manager
from stonesoup.metricgenerator.basicmetrics import BasicMetrics

basic_KF = BasicMetrics(generator_name='KF_fused', tracks_key='KF_fused_tracks',
                        truths_key='truths')

basic_KF1 = BasicMetrics(generator_name='KF1', tracks_key='KF_1_tracks',
                        truths_key='truths')

basic_KF2 = BasicMetrics(generator_name='KF2', tracks_key='KF_2_tracks',
                        truths_key='truths')

basic_PF = BasicMetrics(generator_name='PF_fused', tracks_key='PF_fused_tracks',
                        truths_key='truths')
basic_PF1 = BasicMetrics(generator_name='PF1', tracks_key='PF_1_tracks',
                        truths_key='truths')
basic_PF2 = BasicMetrics(generator_name='PF2', tracks_key='PF_2_tracks',
                        truths_key='truths')

# Load the SIAP metric
from stonesoup.metricgenerator.tracktotruthmetrics import SIAPMetrics
from stonesoup.measures import Euclidean

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
from stonesoup.dataassociator.tracktotrack import TrackToTruth
associator = TrackToTruth(association_threshold=30)

from stonesoup.metricgenerator.manager import MultiManager
metric_manager = MultiManager([basic_KF,
                               basic_KF1,
                               basic_KF2,
                               basic_PF,
                               basic_PF1,
                               basic_PF2,
                               siap_kf_truth,
                               siap_kf1_truth,
                               siap_kf2_truth,
                               siap_pf_truth,
                               siap_pf1_truth,
                               siap_pf2_truth],
                              associator)

# Create the tracks for the particle filters, Kalman and merged ones
PF_track1, PF_track2, KF_track1, KF_track2 = set(), set(), set(), set()
PF_fused_track, KF_fused_track = set(), set()

# Instantiate the various trackers using the general_tracker function,
# we assign a unique tracker the detections from a specific radar simulator
KF_tracker_1 = general_tracker(SingleTargetTracker, r1simulators[0], KF_updater,
                                 initiator, deleter, data_associator_KF)

KF_tracker_2 = general_tracker(SingleTargetTracker, r2simulators[0], KF_updater,
                               initiator, deleter, data_associator_KF)

PF_tracker_1 = general_tracker(SingleTargetTracker, r1simulators[1], PF_updater,
                             PF_initiator, deleter, data_associator_PF)

PF_tracker_2 = general_tracker(SingleTargetTracker, r2simulators[1], PF_updater,
                               PF_initiator, deleter, data_associator_PF)

# Load the detection generator
g1 = radar_simulator1.detections_gen()
g2 = radar_simulator2.detections_gen()

# Each tracker will have the same data
gg1 = deepcopy(radar_simulator1).detections_gen()
gg2 = deepcopy(radar_simulator2).detections_gen()


# Loop on the simulation steps
for aa in range(number_of_steps):

    radar1_detections = next(g1)

    # Run the Particle Filter tracker on the radars measurements
    PF_tracker_1.detector = DummyDetector(current=radar1_detections)
    PF_tracker_1.__iter__()
    _, PF_sensor_track1 = next(PF_tracker_1)
    PF_track1.update(PF_sensor_track1)

    rd1 = next(gg1)
    # Run the Kalman Filter tracker on the radars measurements
    KF_tracker_1.detector = DummyDetector(current=rd1)
    KF_tracker_1.__iter__()
    _, KF_sensor_track1 = next(KF_tracker_1)
    KF_track1.update(KF_sensor_track1)

    radar2_detections = next(g2)

    PF_tracker_2.detector = DummyDetector(current=radar2_detections)
    PF_tracker_2.__iter__()
    pf_time, PF_sensor_track2 = next(PF_tracker_2)
    PF_track2.update(PF_sensor_track2)

    rd2 = next(gg2)

    KF_tracker_2.detector = DummyDetector(current=rd2)
    KF_tracker_2.__iter__()
    kf_time, KF_sensor_track2 = next(KF_tracker_2)
    KF_track2.update(KF_sensor_track2)

    # We have now the track for each radar now let's perform the
    # track fusion
    for PF_track_meas in [PF_track1, PF_track2]:
        dummy_detector_PF = DummyDetector(current=[pf_time, PF_track_meas])
        track_fusion_tracker2.detector = Tracks2GaussianDetectionFeeder(dummy_detector_PF)
        track_fusion_tracker2.__iter__()
        _, tracks = next(track_fusion_tracker2)
        PF_fused_track.update(tracks)

    for KF_track_meas in [KF_track1, KF_track2]:
        dummy_detector_KF = DummyDetector(current=[kf_time, KF_track_meas])
        track_fusion_tracker.detector = Tracks2GaussianDetectionFeeder(dummy_detector_KF)
        track_fusion_tracker.__iter__()
        _, tracks = next(track_fusion_tracker)
        KF_fused_track.update(tracks)


# load the various tracks
metric_manager.add_data({'KF_1_tracks': KF_track1,
                         'KF_2_tracks': KF_track2,
                         'PF_1_tracks': PF_track1,
                         'PF_2_tracks': PF_track2}, overwrite=False)

metric_manager.add_data({'KF_fused_tracks': KF_fused_track,
                         'PF_fused_tracks': PF_fused_track,
                         }, overwrite=False)

truths = set()
truths = set(groundtruth_simulation.groundtruth_paths)
metric_manager.add_data({'truths': truths}, overwrite=False)
# %%
# Let's visualise the various tracks and detections in the cases
# using the Kalman and Particle filters.
plotter.plot_tracks(PF_track1, [0,2], line=dict(color="orange"), track_label='PF partial track')
plotter.plot_tracks(PF_track2, [0,2], line=dict(color="gold"), track_label='PF partial track')
plotter.plot_tracks(PF_fused_track, [0, 2], line=dict(color="red"), track_label='PF fused track')
plotter.plot_tracks(KF_fused_track, [0, 2], line=dict(color="blue"), track_label='KF fused track')
plotter.plot_tracks(KF_track1, [0, 2], line=dict(color="cyan"), track_label='KF partial track')
plotter.plot_tracks(KF_track2, [0, 2], line=dict(color="azure"), track_label='KF partial track')
plotter.plot_ground_truths(truths, [0, 2])

plotter.fig.show()


# %%
# 4) Compare the obtained tracks with the groundtruth trajectory.
# ---------------------------------------------------------------
# At this stage we have almost completed our example. We have created the
# detections from the radars, performed the tracking and the
# fusion of the tracks. Now we use the :class:`~.MetricManager`
# to generate summary statistics on the accuracy of the tracks
# in comparison to the groundtruth measurements.
#

# Loaded the plotter for the various metrics.
from stonesoup.plotter import MetricPlotter

metrics = metric_manager.generate_metrics()
graph = MetricPlotter()

graph.plot_metrics(metrics, generator_names=['SIAP_KF_fused-truth',
                                             'SIAP_PF_fused-truth',
                                             'SIAP_KF1-truth',
                                             'SIAP_KF2-truth',
                                             'SIAP_PF1-truth',
                                             'SIAP_PF2-truth'],
                   color=['blue', 'red', 'cyan', 'azure', 'orange', 'gold'])
graph.fig.show()

# %%
# Conclusions
# -----------
# In this example we have shown how it is possible to
# merge the tracks generated by independent trackers
# ran on sets of data obtained by separate sensors. We
# have, also, compared how the Kalman and the particle
# filters behave in these cases, making track to track
# comparisons.
