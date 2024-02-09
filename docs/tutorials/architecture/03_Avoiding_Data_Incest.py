#!/usr/bin/env python
# coding: utf-8

"""
========================
3 - Avoiding Data Incest
========================
"""

import random
import copy
import numpy as np
from datetime import datetime, timedelta


# %%
# Introduction
# ------------
# This tutorial uses the Stone Soup Architecture module to provide an example of how data incest
# can occur in a poorly designed network.
#
# We design two architectures: a centralised (non-hierarchical) architecture, and a hierarchical
# alternative, and look to compare the fused results at the central node.
#
# Scenario generation
# -------------------

start_time = datetime.now().replace(microsecond=0)
np.random.seed(1990)
random.seed(1990)

# %%
# "Good" and "Bad" Sensors
# ^^^^^^^^^^^^^^^^^^^^^^^^
#
# We build a "good" sensor with low measurement noise (high measurement accuracy).

from stonesoup.types.state import StateVector
from stonesoup.sensor.radar.radar import RadarRotatingBearingRange
from stonesoup.types.angle import Angle

good_sensor = RadarRotatingBearingRange(
         position_mapping=(0, 2),
         noise_covar=np.array([[np.radians(0.5) ** 2, 0],
                               [0, 1 ** 2]]),
         ndim_state=4,
         position=np.array([[10], [20 - 40]]),
         rpm=60,
         fov_angle=np.radians(360),
         dwell_centre=StateVector([0.0]),
         max_range=np.inf,
         resolutions={'dwell_centre': Angle(np.radians(30))}
     )

good_sensor.timestamp = start_time

# %%
# We then build a third "bad" sensor with high measurement noise (low measurement accuracy).
# This will enable us to design an architecture and observe how the "bad" measurements are
# propagated and fused with the "good" measurements. The bad sensor has its noise covariance
# scaled by a factor of `sf` over the noise of the good sensors.

sf = 2
bad_sensor = RadarRotatingBearingRange(
            position_mapping=(0, 2),
            noise_covar=np.array([[sf * np.radians(0.5) ** 2, 0],
                                  [0, sf * 1 ** 2]]),
            ndim_state=4,
            position=np.array([[10], [3*20 - 40]]),
            rpm=60,
            fov_angle=np.radians(360),
            dwell_centre=StateVector([0.0]),
            max_range=np.inf,
            resolutions={'dwell_centre': Angle(np.radians(30))}
    )

bad_sensor.timestamp = start_time
all_sensors = {good_sensor, bad_sensor}

# %%
# Ground Truth
# ^^^^^^^^^^^^

from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
    ConstantVelocity
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from ordered_set import OrderedSet

# Generate transition model
transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.005),
                                                          ConstantVelocity(0.005)])

yps = range(0, 100, 10)  # y value for prior state
truths = OrderedSet()
ntruths = 1  # number of ground truths in simulation
time_max = 60  # timestamps the simulation is observed over
timesteps = [start_time + timedelta(seconds=k) for k in range(time_max)]

xdirection = 1
ydirection = 1

# Generate ground truths
for j in range(0, ntruths):
    truth = GroundTruthPath([GroundTruthState([0, xdirection, yps[j], ydirection],
                                              timestamp=timesteps[0])], id=f"id{j}")

    for k in range(1, time_max):
        truth.append(
            GroundTruthState(transition_model.function(truth[k - 1], noise=True,
                                                       time_interval=timedelta(seconds=1)),
                             timestamp=timesteps[k]))
    truths.add(truth)

    xdirection *= -1
    if j % 2 == 0:
        ydirection *= -1

# %%
# Build Tracker
# ^^^^^^^^^^^^^
#
# We use the same configuration of trackers and track-trackers as we did in the previous tutorial.

from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.updater.kalman import ExtendedKalmanUpdater
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.measures import Mahalanobis
from stonesoup.dataassociator.neighbour import GNNWith2DAssignment
from stonesoup.deleter.error import CovarianceBasedDeleter
from stonesoup.types.state import GaussianState
from stonesoup.initiator.simple import MultiMeasurementInitiator
from stonesoup.tracker.simple import MultiTargetTracker
from stonesoup.architecture.edge import FusionQueue

predictor = KalmanPredictor(transition_model)
updater = ExtendedKalmanUpdater(measurement_model=None)
hypothesiser = DistanceHypothesiser(predictor, updater, measure=Mahalanobis(), missed_distance=5)
data_associator = GNNWith2DAssignment(hypothesiser)
deleter = CovarianceBasedDeleter(covar_trace_thresh=3)
initiator = MultiMeasurementInitiator(
    prior_state=GaussianState([[0], [0], [0], [0]], np.diag([0, 1, 0, 1])),
    measurement_model=None,
    deleter=deleter,
    data_associator=data_associator,
    updater=updater,
    min_points=2,
    )

tracker = MultiTargetTracker(initiator, deleter, None, data_associator, updater)

# %%
# Track Tracker
# ^^^^^^^^^^^^^

from stonesoup.updater.wrapper import DetectionAndTrackSwitchingUpdater
from stonesoup.updater.chernoff import ChernoffUpdater
from stonesoup.feeder.track import Tracks2GaussianDetectionFeeder

track_updater = ChernoffUpdater(None)
detection_updater = ExtendedKalmanUpdater(None)
detection_track_updater = DetectionAndTrackSwitchingUpdater(None, detection_updater, track_updater)


fq = FusionQueue()

track_tracker = MultiTargetTracker(
    initiator, deleter, Tracks2GaussianDetectionFeeder(fq), data_associator, detection_track_updater)

# %%
# Non-Hierarchical Architecture
# -----------------------------
#
# We start by constructing the non-hierarchical, centralised architecture.
#
# Nodes
# ^^^^^

from stonesoup.architecture.node import SensorNode, FusionNode, SensorFusionNode

node_B1_tracker = copy.deepcopy(tracker)
node_B1_tracker.detector = FusionQueue()

node_A1 = SensorNode(sensor=bad_sensor,
                     label='Bad\nSensorNode')
node_B1 = SensorFusionNode(sensor=good_sensor,
                           label='Good\nSensorFusionNode',
                           tracker=node_B1_tracker,
                           fusion_queue=node_B1_tracker.detector)
node_C1 = FusionNode(tracker=track_tracker,
                     fusion_queue=fq,
                     latency=0,
                     label='Fusion Node')

# %%
# Edges
# ^^^^^
#
# Here we define the set of edges for the non-hierarchical (NH) architecture.

from stonesoup.architecture import InformationArchitecture
from stonesoup.architecture.edge import Edge, Edges

NH_edges = Edges([Edge((node_A1, node_B1), edge_latency=1),
              Edge((node_B1, node_C1), edge_latency=0),
              Edge((node_A1, node_C1), edge_latency=0)])

# %%
# Create the Non-Hierarchical Architecture
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# sphinx_gallery_thumbnail_path = '_static/sphinx_gallery/ArchTutorial_3.png'

NH_architecture = InformationArchitecture(NH_edges, current_time=start_time, use_arrival_time=True)
NH_architecture.plot(plot_style='hierarchical')  # Style similar to hierarchical
NH_architecture

# %%
# Run the Non-Hierarchical Simulation
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

for time in timesteps:
    NH_architecture.measure(truths, noise=True)
    NH_architecture.propagate(time_increment=1)

# %%
# Extract all detections that arrived at Non-Hierarchical Node C
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

from stonesoup.types.detection import TrueDetection

NH_detections = set()
for timestep in node_C1.data_held['unfused']:
    for datapiece in node_C1.data_held['unfused'][timestep]:
        if isinstance(datapiece.data, TrueDetection):
            NH_detections.add(datapiece.data)

# %%
# Plot the tracks stored at Non-Hierarchical Node C
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

from stonesoup.plotter import Plotterly


def reduce_tracks(tracks):
    return {
        type(track)([s for s in track.last_timestamp_generator()])
        for track in tracks}


plotter = Plotterly()
plotter.plot_ground_truths(truths, [0, 2])
plotter.plot_tracks(reduce_tracks(node_C1.tracks), [0, 2], track_label="Node C1",
                    line=dict(color='#00FF00'), uncertainty=True)
plotter.plot_sensors(all_sensors)
plotter.plot_measurements(NH_detections, [0, 2])
plotter.fig

# %%
# Hierarchical Architecture
# -------------------------
#
# We now create an alternative architecture. We recreate the same set of nodes as before, but
# with a new edge set, which is a subset of the edge set used in the non-hierarchical
# architecture.
#
# Regenerate Nodes identical to those in the non-hierarchical example
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

from stonesoup.architecture.node import SensorNode, FusionNode
fq2 = FusionQueue()

track_tracker2 = MultiTargetTracker(
    initiator, deleter, Tracks2GaussianDetectionFeeder(fq2), data_associator,
    detection_track_updater)

node_B2_tracker = copy.deepcopy(tracker)
node_B2_tracker.detector = FusionQueue()

node_A2 = SensorNode(sensor=bad_sensor,
                     label='Bad\nSensorNode')
node_B2 = SensorFusionNode(sensor=good_sensor,
                           label='Good\nSensorFusionNode',
                           tracker=node_B2_tracker,
                           fusion_queue=node_B2_tracker.detector)
node_C2 = FusionNode(tracker=track_tracker2,
                     fusion_queue=fq2,
                     latency=0,
                     label='Fusion Node')

# %%
# Create Edges forming a Hierarchical Architecture
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

H_edges = Edges([Edge((node_A2, node_C2), edge_latency=0),
                Edge((node_B2, node_C2), edge_latency=0)])

# %%
# Create the Hierarchical Architecture
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

H_architecture = InformationArchitecture(H_edges, current_time=start_time, use_arrival_time=True)
H_architecture

# %%
# Run the Hierarchical Simulation
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

for time in timesteps:
    H_architecture.measure(truths, noise=True)
    H_architecture.propagate(time_increment=1)

# %%
# Extract all detections that arrived at Hierarchical Node C
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

H_detections = set()
for timestep in node_C2.data_held['unfused']:
    for datapiece in node_C2.data_held['unfused'][timestep]:
        if isinstance(datapiece.data, TrueDetection):
            H_detections.add(datapiece.data)

# %%
# Plot the tracks stored at Hierarchical Node C
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

plotter = Plotterly()
plotter.plot_ground_truths(truths, [0, 2])
plotter.plot_tracks(reduce_tracks(node_C2.tracks), [0, 2], track_label="Node C2",
                    line=dict(color='#00FF00'), uncertainty=True)
plotter.plot_sensors(all_sensors)
plotter.plot_measurements(H_detections, [0,2])
plotter.fig

# %%
# Metrics
# -------
# At a glance, the results from the hierarchical architecture look similar to the results from
# the original centralised architecture. We will now calculate and plot some metrics to give an
# insight into the differences.
#
# Calculate SIAP metrics for Centralised Architecture
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

from stonesoup.metricgenerator.tracktotruthmetrics import SIAPMetrics
from stonesoup.measures import Euclidean
from stonesoup.dataassociator.tracktotrack import TrackToTruth
from stonesoup.metricgenerator.manager import MultiManager

NH_siap_EKF_truth = SIAPMetrics(position_measure=Euclidean((0, 2)),
                                velocity_measure=Euclidean((1, 3)),
                                generator_name='NH_SIAP_EKF-truth',
                                tracks_key='NH_EKF_tracks',
                                truths_key='truths'
                                )

associator = TrackToTruth(association_threshold=30)


# %%

NH_metric_manager = MultiManager([NH_siap_EKF_truth,
                                  ], associator)  # associator for generating SIAP metrics
NH_metric_manager.add_data({'NH_EKF_tracks': node_C1.tracks,
                            'truths': truths,
                            'NH_detections': NH_detections}, overwrite=False)
NH_metrics = NH_metric_manager.generate_metrics()


# %%

NH_siap_metrics = NH_metrics['NH_SIAP_EKF-truth']
NH_siap_averages_EKF = {NH_siap_metrics.get(metric) for metric in NH_siap_metrics
                        if metric.startswith("SIAP") and not metric.endswith(" at times")}

# %%
# Calculate Metrics for Hierarchical Architecture
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

H_siap_EKF_truth = SIAPMetrics(position_measure=Euclidean((0, 2)),
                             velocity_measure=Euclidean((1, 3)),
                             generator_name='H_SIAP_EKF-truth',
                             tracks_key='H_EKF_tracks',
                             truths_key='truths'
                             )

associator = TrackToTruth(association_threshold=30)


# %%

H_metric_manager = MultiManager([H_siap_EKF_truth
                                 ], associator)  # associator for generating SIAP metrics
H_metric_manager.add_data({'H_EKF_tracks':node_C2.tracks,
                         'truths': truths,
                         'H_detections': H_detections}, overwrite=False)
H_metrics = H_metric_manager.generate_metrics()


# %%

H_siap_metrics = H_metrics['H_SIAP_EKF-truth']
H_siap_averages_EKF = {H_siap_metrics.get(metric) for metric in H_siap_metrics
                     if metric.startswith("SIAP") and not metric.endswith(" at times")}

# %%
# Compare Metrics
# ^^^^^^^^^^^^^^^
#
# Below we plot a table of SIAP metrics for both architectures. This results in this comparison
# table show that the results from the hierarchical architecture outperform the results from the
# centralised architecture.
#
# Further below is plot of SIAP position accuracy over time for the duration of the simulation.
# Smaller values represent higher accuracy

from stonesoup.metricgenerator.metrictables import SiapDiffTableGenerator
SiapDiffTableGenerator(NH_siap_averages_EKF, H_siap_averages_EKF).compute_metric()


# %%

from stonesoup.plotter import MetricPlotter

combined_metrics = H_metrics | NH_metrics
graph = MetricPlotter()
graph.plot_metrics(combined_metrics, generator_names=['H_SIAP_EKF-truth',
                                                     'NH_SIAP_EKF-truth'],
                   metric_names=['SIAP Position Accuracy at times'],
                   color=['red', 'blue'])

# %%
# Explanation of Results
# ----------------------
#
# In the centralised architecture, measurements from the 'bad' sensor are passed to both the
# central fusion node (C), and the sensor fusion node (B). At node B, these measurements are
# fused with measurements from the 'good' sensor, and the output is sent to node C.
#
# At node C, the fused results from node B are once again fused with the measurements from node
# A, despite implicitly containing the information from sensor A already. Hence, we end up with a
# fused result that is biased towards the readings from sensor A.
#
# By altering the architecture through removing the edge from node A to node B, we are removing
# the incestual loop, and the resulting fusion at node C is just fusion of two disjoint sets of
# measurements. Although node C is still receiving the less accurate measurements, it is not
# biased towards the measurements from node A.
