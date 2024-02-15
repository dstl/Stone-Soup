#!/usr/bin/env python
# coding: utf-8

"""
==========================================================
Performance comparison between Kalman and Particle Filters
==========================================================
"""

# %%
# In this example, we present the case of data fusion. In particular,
# we are looking at measurement fusion from two sensors. The context
# is a multi-target tracking scenario, where we want to compare
# the performances of separate filters: an unscented Kalman filter (UKF),
# an extended Kalman filter (EKF) and a particle filter (PF).
#
# The example layout follows:
#
# 1. define the targets trajectories and the sensors specifics for collecting the measurements;
# 2. define the various filter components and build the trackers;
# 3. perform the measurement fusion algorithm and run the trackers;
# 4. plot the tracks and the track performances using the metric manager tool available.
#

# %%
# 1. Define the targets trajectories and the sensors collecting the measurements
# ------------------------------------------------------------------------------
# Let's define the targets trajectories, assuming a simple case
# of a straight movement and using the same instrument specifics for both
# sensors. We consider two :class:`~.RadarBearingRange` radars
# collecting the detections of the targets, in Cartesian space.
# The first radar is placed onto a :class:`~.FixedPlatform`, while the
# second is on a :class:`~.MovingPlatform`.
# The targets follow a straight line trajectory for simplicity.
# For the targets we instantiate the origins and a transition model
# with a :class:`~.ConstantVelocity` model with noise equal to 0.
#

# %%
# General imports
# ^^^^^^^^^^^^^^^
import numpy as np
from datetime import datetime, timedelta

# %%
# Stone Soup general imports
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
    ConstantVelocity
from stonesoup.types.state import GaussianState
from stonesoup.types.array import CovarianceMatrix
from stonesoup.simulator.simple import MultiTargetGroundTruthSimulator

# %%
# Simulation parameters setup
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
start_time = datetime(2023, 8, 1, 10, 0, 0)  # For simplicity fix a date, or datetime.now()
number_of_steps = 50  # Number of time-steps for the simulation
np.random.seed(1908)  # Random seed for reproducibility
n_particles = 2**10  # Fix the number of particles

# %%
# Generate Ground Truths
# ^^^^^^^^^^^^^^^^^^^^^^

# Specify the ground truth transition model
gnd_transition_model = CombinedLinearGaussianTransitionModel([
    ConstantVelocity(0.00), ConstantVelocity(0.00)])

# Instantiate the target transition model with two dimensions and
# insert some noise for the prediction in particle filter
transition_model = CombinedLinearGaussianTransitionModel([
    ConstantVelocity(0.1), ConstantVelocity(0.1)])

# Define the initial target state
initial_target_state_1 = GaussianState([25, 1, 50, -0.5],
                                       np.diag([1, 0.1, 1, 0.1]) ** 2,
                                       timestamp=start_time)
# Define the initial target state
initial_target_state_2 = GaussianState([25, 1, -50, 0.5],
                                       np.diag([1, 0.1, 1, 0.1]) ** 2,
                                       timestamp=start_time)

# Create a ground truth simulator and specify the number of initial targets as
# 0, so that no new targets will be created aside from the two provided
ground_truth_simulator = MultiTargetGroundTruthSimulator(
    transition_model=gnd_transition_model,
    initial_state=GaussianState([10, 1, 0, 0.5],
                                 np.diag([5, 0.1, 5, 0.1]),
                                 timestamp=start_time),
    birth_rate=0.0,
    death_probability=0.0,
    number_steps=number_of_steps,
    preexisting_states=[initial_target_state_1.state_vector,
                        initial_target_state_2.state_vector],
    initial_number_targets=0)

# %%
# Generate clutter
# ^^^^^^^^^^^^^^^^
from stonesoup.models.clutter.clutter import ClutterModel

# Define the clutter model which will be the same for both sensors
# Keep the clutter rate low due to particle filter errors
clutter_model = ClutterModel(
    clutter_rate=1,
    distribution=np.random.default_rng().uniform,
    dist_params=((0, 150), (-105, 105)))

# Define the clutter area and spatial density for the tracker
clutter_area = np.prod(np.diff(clutter_model.dist_params))
clutter_spatial_density = clutter_model.clutter_rate/clutter_area

# %%
# Radar sensor and platform set-up
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Instantiate the radars to collect measurements - Use a :class:`~.RadarBearingRange`.
from stonesoup.sensor.radar.radar import RadarBearingRange

# Let's assume that both radars have the same noise covariance for simplicity
# These radars will have the +/-0.1 degrees accuracy in bearing and 5 meters in range
radar_noise = CovarianceMatrix(np.diag([np.deg2rad(0.1), 5]))

# Define the specifications of the two radars
radar1 = RadarBearingRange(
    ndim_state=4,
    position_mapping=(0, 2),
    noise_covar=radar_noise,
    clutter_model=clutter_model,
    max_range=3000)  # max_range can be ignored as well

radar2 = RadarBearingRange(
    ndim_state=4,
    position_mapping=(0, 2),
    noise_covar=radar_noise,
    clutter_model=clutter_model,
    max_range=3000)

# Import the platform to place the sensors on
from stonesoup.platform.base import FixedPlatform
from stonesoup.platform.base import MovingPlatform

# Instantiate the first sensor platform and add the sensor
sensor1_platform = FixedPlatform(
    states=GaussianState([10, 0, 5, 0],
                         np.diag([1, 0, 1, 0])),
    position_mapping=(0, 2),
    sensors=[radar1])

sensor2_platform = MovingPlatform(
    states=GaussianState([120, 0, -50, 1.5],
                         np.diag([1, 0, 5, 1])),
    position_mapping=(0, 2),
    velocity_mapping=(0, 2),
    transition_model=gnd_transition_model,
    sensors=[radar2])

# Load the platform detection simulator - Let's use a simulator for each track
# Instantiate the simulators
from stonesoup.simulator.platform import PlatformDetectionSimulator

radar_simulator1 = PlatformDetectionSimulator(
    groundtruth=ground_truth_simulator,
    platforms=[sensor1_platform])

radar_simulator2 = PlatformDetectionSimulator(
    groundtruth=ground_truth_simulator,
    platforms=[sensor2_platform])


# %%
# 2. Define the various filter components and build the trackers
# --------------------------------------------------------------
# We have presented the scenario with two separate moving targets
# and two sensors collecting the measurements. Now, we focus on
# building the various tracker components: we use a :class:`~.DistanceHypothesiser`
# hypothesiser using :class:`~.Mahalanobis` distance measure to assign detections to tracks.
# We consider an Unscented Kalman filter (UKF), an Extended Kalman filter
# (EKF) and a Particle filter (PF) using the same components specifications.
# For the deleter we consider an :class:`~.UpdateTimeDeleter`.
#

# %%
# Stone Soup imports
# ^^^^^^^^^^^^^^^^^^

# We use a Distance hypothesiser
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.measures import Mahalanobis

# Use a GNN 2D assignment, time deleter and initiator
from stonesoup.dataassociator.neighbour import GNNWith2DAssignment
from stonesoup.deleter.time import  UpdateTimeDeleter
from stonesoup.initiator.simple import MultiMeasurementInitiator, GaussianParticleInitiator

# Load the UKF components
from stonesoup.updater.kalman import UnscentedKalmanUpdater
from stonesoup.predictor.kalman import UnscentedKalmanPredictor

# Load the EKF components
from stonesoup.updater.kalman import ExtendedKalmanUpdater
from stonesoup.predictor.kalman import ExtendedKalmanPredictor

# prepare the Particle filter components
from stonesoup.predictor.particle import ParticlePredictor
from stonesoup.updater.particle import ParticleUpdater
from stonesoup.resampler.particle import ESSResampler

# Load the multitarget tracker
from stonesoup.tracker.simple import MultiTargetTracker

# Load a detection reader
from stonesoup.buffered_generator import BufferedGenerator
from stonesoup.reader.base import DetectionReader

# %%
# Design the trackers components
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# %%
# Detection reader and track deleter

# define a track deleter based on time measurements
deleter = UpdateTimeDeleter(timedelta(seconds=3), delete_last_pred=False)

# Create a dummy detector to parse the detections
class DummyDetector(DetectionReader):
    def __init__(self, *args, **kwargs):
        self.current = kwargs['current']

    @BufferedGenerator.generator_method
    def detections_gen(self):
        yield self.current

# Unscented Kalman Filter
# load the Unscented Kalman filter predictor and updater
UKF_updater = UnscentedKalmanUpdater(measurement_model=None)
UKF_predictor = UnscentedKalmanPredictor(transition_model)

# define the hypothesiser
hypothesiser_UKF = DistanceHypothesiser(
    predictor=UKF_predictor,
    updater=UKF_updater,
    measure=Mahalanobis(),
    missed_distance=5)

# define the distance data associator
data_associator_UKF = GNNWith2DAssignment(hypothesiser_UKF)

# create a track initiator placed on the target tracks origin
UKF_initiator = MultiMeasurementInitiator(
    prior_state=GaussianState([10, 0, 10, 0],
                              np.diag([1, 1, 1, 1])),
    measurement_model=None,
    deleter=deleter,
    updater=UKF_updater,
    data_associator=data_associator_UKF,
    min_points=5)

# %%
# Extended Kalman Filter

# load the Extended Kalman filter predictor and updater
EKF_predictor = ExtendedKalmanPredictor(transition_model)
EKF_updater = ExtendedKalmanUpdater(measurement_model=None)

# define the hypothesiser
hypothesiser_EKF = DistanceHypothesiser(
    predictor=EKF_predictor,
    updater=EKF_updater,
    measure=Mahalanobis(),
    missed_distance=5)

# define the distance data associator
data_associator_EKF = GNNWith2DAssignment(hypothesiser_EKF)

EKF_initiator = MultiMeasurementInitiator(
    prior_state=GaussianState([10, 0, 10, 0],
                              np.diag([1, 1, 1, 1])),
    measurement_model=None,
    deleter=deleter,
    updater=EKF_updater,
    data_associator=data_associator_EKF,
    min_points=5)

# %%
# Particle Filter

# Instantiate the particle predictor, resampler and updater
PF_predictor = ParticlePredictor(transition_model)
resampler = ESSResampler(threshold=n_particles/2.)
PF_updater = ParticleUpdater(measurement_model=None,
                             resampler=resampler)

hypothesiser_PF = DistanceHypothesiser(
    predictor=PF_predictor,
    updater=PF_updater,
    measure=Mahalanobis(),
    missed_distance=10)

# define the data associator
data_associator_PF = GNNWith2DAssignment(hypothesiser_PF)

# To instantiate the track initiator we define a prior state as gaussian state with the target track origin
# For the initiator we consider a KF based data associator
initiator_particles = MultiMeasurementInitiator(
    GaussianState([10, 0, 10, 0],
                  np.diag([5, 0.1, 5, 0.1]) ** 2,
                  timestamp=start_time),
    measurement_model=None,
    deleter=deleter,
    data_associator=GNNWith2DAssignment(
        DistanceHypothesiser(EKF_predictor, EKF_updater, Mahalanobis(), missed_distance=3)),
    updater=EKF_updater,
    min_points=5)

# Particle filter initiator, use 1024 particles
PF_initiator = GaussianParticleInitiator(
    initiator=initiator_particles,
    number_particles=n_particles)


# Instantiate each of the Trackers, without specifying the detector
UKF_tracker = MultiTargetTracker(
    initiator=UKF_initiator,
    deleter=deleter,
    data_associator=data_associator_UKF,
    updater=UKF_updater,
    detector=None)

EKF_tracker = MultiTargetTracker(
    initiator=EKF_initiator,
    deleter=deleter,
    data_associator=data_associator_EKF,
    updater=EKF_updater,
    detector=None)

# Instantiate the Particle filter as well
PF_tracker = MultiTargetTracker(
    initiator=PF_initiator,
    deleter=deleter,
    data_associator=data_associator_PF,
    updater=PF_updater,
    detector=None)

# %%
# 3. Perform the measurement fusion algorithm and run the trackers
# ----------------------------------------------------------------
# We have instantiated all the relevant components for the filters,
# and now we can run the simulation to generate the various detections, clutter and track associations.
# The final tracks will be passed onto a metric generator
# plotter to measure the track accuracy.
# We start composing the various metrics statistics available.

# %%
# Stone Soup plotting imports
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^

# Load the plotter
from stonesoup.plotter import Plotterly

# Load the metric manager
from stonesoup.metricgenerator.basicmetrics import BasicMetrics

# Load the OSPA metric managers
from stonesoup.metricgenerator.ospametric import OSPAMetric

# Define a data associator between the tracks and the truths
from stonesoup.dataassociator.tracktotrack import TrackToTruth

# load a multi-metric manager
from stonesoup.metricgenerator.manager import MultiManager

# Loaded the plotter for the various metrics.
from stonesoup.plotter import MetricPlotter

# %%
# Metrics
# ^^^^^^^

# load the metrics for the filters
basic_UKF = BasicMetrics(generator_name='Unscented Kalman Filter', tracks_key='UKF_tracks',
                        truths_key='truths')
basic_EKF = BasicMetrics(generator_name='Extended Kalman Filter', tracks_key='EKF_tracks',
                         truths_key='truths')
basic_PF = BasicMetrics(generator_name='Particle Filter', tracks_key='PF_tracks',
                        truths_key='truths')

# OSPA
ospa_UKF_truth = OSPAMetric(c=40, p=1, generator_name='OSPA_UKF_truths',
                           tracks_key='UKF_tracks',  truths_key='truths')
ospa_EKF_truth = OSPAMetric(c=40, p=1, generator_name='OSPA_EKF_truths',
                            tracks_key='EKF_tracks',  truths_key='truths')
ospa_PF_truth = OSPAMetric(c=40, p=1, generator_name='OSPA_PF_truths',
                           tracks_key='PF_tracks',  truths_key='truths')

# Use the track associatior
associator = TrackToTruth(association_threshold=30)

# Use a metric manager to deal with the various metrics
metric_manager = MultiManager([basic_UKF,
                               basic_EKF,
                               basic_PF,
                               ospa_UKF_truth,
                               ospa_EKF_truth,
                               ospa_PF_truth],
                              associator)

# %%
# Run simulations
# ^^^^^^^^^^^^^^^

# define the Detections from the two sensors
s1_detections = []
s2_detections = []
radar_path = []

# use the generator function from the simulators
g1 = radar_simulator1.detections_gen()
g2 = radar_simulator2.detections_gen()

# Create the tracks
ukf_tracks = set()
ekf_tracks = set()
pf_tracks = set()
truths = set()

# list for all detections
full_detections = []

for t in range(number_of_steps):  # loop over the various time-steps
    detections_1 = next(g1)
    s1_detections.extend(detections_1[1])
    detections_2 = next(g2)
    s2_detections.extend(detections_2[1])
    full_detections.extend(detections_1[1])
    full_detections.extend(detections_2[1])

    for detections in [detections_1, detections_2]:

        # Run the Unscented Kalman tracker
        UKF_tracker.detector = DummyDetector(current=detections)
        UKF_tracker.__iter__()
        _, tracks = next(UKF_tracker)
        ukf_tracks.update(tracks)
        del tracks

        # Run the Extended Kalman filter
        EKF_tracker.detector = DummyDetector(current=detections)
        EKF_tracker.__iter__()
        _, tracks = next(EKF_tracker)
        ekf_tracks.update(tracks)
        del tracks

        # Run the Particle filter Tracker
        PF_tracker.detector = DummyDetector(current=detections)
        PF_tracker.__iter__()
        _, tracks = next(PF_tracker)
        pf_tracks.update(tracks)

# Add data to the metric manager
metric_manager.add_data({'UKF_tracks': ukf_tracks}, overwrite=False)
metric_manager.add_data({'PF_tracks': pf_tracks}, overwrite=False)
metric_manager.add_data({'EKF_tracks': ekf_tracks}, overwrite=False)

truths = set(ground_truth_simulator.groundtruth_paths)
metric_manager.add_data({'truths': truths,
                         'detections': full_detections}, overwrite=False)

# %%
# 4. Plot the tracks and the track performances
# ---------------------------------------------
# We have obtained the tracks and the ground truths from the trackers. It is time
# to visualise the tracks and load the metric manager to evaluate the performances.

plotter = Plotterly()
plotter.plot_measurements(s1_detections, [0, 2],
                         measurements_label='Radar 1 measurements'),
plotter.plot_measurements(s2_detections, [0, 2],
                         measurements_label='Radar 2 measurements')
plotter.plot_tracks(ukf_tracks, [0, 2], line= dict(color='green'), track_label='UKF tracks')
plotter.plot_tracks(ekf_tracks, [0, 2], line= dict(color='blue'), track_label='EKF tracks')
plotter.plot_tracks(pf_tracks, [0, 2], particle=False, line= dict(color='red'),
                    track_label='PF tracks')
plotter.plot_ground_truths(truths, [0, 2])
plotter.plot_sensors(sensor1_platform, [0, 1], marker=dict(color='black', symbol='129', size=15),
                     sensor_label='Fixed Platform')
plotter.plot_ground_truths(sensor2_platform, [0, 2], marker=dict(color='orange', symbol='cross',
                                                           size=25),
                     truths_label='Moving Platform')
plotter.fig

# %%
# Plot the metrics

metrics = metric_manager.generate_metrics()

graph = MetricPlotter()

graph.plot_metrics(metrics, generator_names=['OSPA_UKF_truths',
                                             'OSPA_EKF_truths',
                                             'OSPA_PF_truths'],
                   color=['green', 'blue', 'red'])
graph.axes[0].set(ylabel='OSPA metrics', title='OSPA distances over time')
graph.fig

# %%
# Conclusion
# ----------
# This concludes this example where we have shown how to perform measurement fusion using two
# sensors, and we have shown the performances of the tracks obtained by an Unscented Kalman filter,
# an Extended Kalman Filter and a Particle filter, using distance hypothesiser based data associator.
#
