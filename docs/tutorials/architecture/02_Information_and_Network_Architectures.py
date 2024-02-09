#!/usr/bin/env python
# coding: utf-8

"""
=========================================
2 - Information and Network Architectures
=========================================
"""

import random
from datetime import datetime, timedelta
import copy
import numpy as np


# %%
# Introduction
# ------------
#
# Following on from Tutorial 01: Introduction to Architectures in Stone Soup, this tutorial
# provides a more in-depth walk through of generating and simulating information and network
# architectures.
#
# We start by generating a set of sensors, and a ground truth from which the sensors will make
# measurements. We will make use of these in both information and network architecture examples
# later in the tutorial.


# %%
# Set up variables
# ^^^^^^^^^^^^^^^^

start_time = datetime.now().replace(microsecond=0)
np.random.seed(1990)
random.seed(1990)


# %%
# Create Sensors
# ^^^^^^^^^^^^^^

total_no_sensors = 6

from ordered_set import OrderedSet
from stonesoup.types.state import StateVector
from stonesoup.sensor.radar.radar import RadarRotatingBearingRange
from stonesoup.types.angle import Angle

sensor_set = OrderedSet()
for n in range(0, total_no_sensors):
    sensor = RadarRotatingBearingRange(
        position_mapping=(0, 2),
        noise_covar=np.array([[np.radians(0.5) ** 2, 0],
                              [0, 1 ** 2]]),
        ndim_state=4,
        position=np.array([[10], [n*20 - 40]]),
        rpm=60,
        fov_angle=np.radians(360),
        dwell_centre=StateVector([0.0]),
        max_range=np.inf,
        resolutions={'dwell_centre': Angle(np.radians(30))}
    )
    sensor_set.add(sensor)
for sensor in sensor_set:
    sensor.timestamp = start_time


# %%
# Create ground truth
# ^^^^^^^^^^^^^^^^^^^

from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
    ConstantVelocity
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState

# Generate transition model
transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.005),
                                                          ConstantVelocity(0.005)])

yps = range(0, 100, 10)  # y value for prior state
truths = OrderedSet()
ntruths = 3  # number of ground truths in simulation
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
# Information Architecture Example
# --------------------------------
#
# An information architecture represents the points in a network which propagate, open, process,
# and/or fuse data. This type of architecture does not consider the physical network, and hence
# does not represent any nodes whose only functionality is to pass data in between information
# processing nodes.

# %%
# Information Architecture nodes
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Firstly, we set up a set of Nodes that will feature in our information architecture. As the set
# of nodes will include FusionNodes. As a prerequisite of building a FusionNode, we must first
# build FusionTrackers and FusionQueues.


# %%
# Build Tracker
# ^^^^^^^^^^^^^

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
deleter = CovarianceBasedDeleter(covar_trace_thresh=7)
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
# Build Track Tracker
# ^^^^^^^^^^^^^^^^^^^
#
# The track tracker works by treating tracks as detections, in order to enable fusion between
# tracks and detections together.

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
# Build Information Architecture Nodes
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

from stonesoup.architecture.node import SensorNode, FusionNode
node_A = SensorNode(sensor=sensor_set[0], label='SensorNode A')
node_B = SensorNode(sensor=sensor_set[2], label='SensorNode B')

node_C_tracker = copy.deepcopy(tracker)
node_C_tracker.detector = FusionQueue()
node_C = FusionNode(tracker=node_C_tracker, fusion_queue=node_C_tracker.detector, latency=0, label='FusionNode C')

node_D = SensorNode(sensor=sensor_set[1], label='SensorNode D')
node_E = SensorNode(sensor=sensor_set[3], label='SensorNode E')

node_F_tracker = copy.deepcopy(tracker)
node_F_tracker.detector = FusionQueue()
node_F = FusionNode(tracker=node_F_tracker, fusion_queue=node_F_tracker.detector, latency=0)

node_H = SensorNode(sensor=sensor_set[4])

node_G = FusionNode(tracker=track_tracker, fusion_queue=fq, latency=0)


# %%
# Build Information Architecture Edges
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

from stonesoup.architecture import InformationArchitecture
from stonesoup.architecture.edge import Edge, Edges

edges=Edges([Edge((node_A, node_C), edge_latency=0.5),
                 Edge((node_B, node_C)),
                 Edge((node_D, node_F)),
                 Edge((node_E, node_F)),
                 Edge((node_C, node_G), edge_latency=0),
                 Edge((node_F, node_G), edge_latency=0),
                 Edge((node_H, node_G)),
                ])


# %%
# Build and plot Information Architecture
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# sphinx_gallery_thumbnail_path = '_static/sphinx_gallery/ArchTutorial_2.png'

information_architecture = InformationArchitecture(edges, current_time=start_time)
information_architecture

# %%
# Simulate measuring and propagating data over the network
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

for time in timesteps:
    information_architecture.measure(truths, noise=True)
    information_architecture.propagate(time_increment=1)

# %%

from stonesoup.plotter import Plotterly


def reduce_tracks(tracks):
    return {
        type(track)([s for s in track.last_timestamp_generator()])
        for track in tracks}


# %%
# Plot Tracks at the Fusion Nodes
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

plotter = Plotterly()

plotter.plot_ground_truths(truths, [0, 2])
plotter.plot_tracks(reduce_tracks(node_C.tracks), [0, 2], track_label=node_C.label,
                    line=dict(color='#00FF00'), uncertainty=True)
plotter.plot_tracks(reduce_tracks(node_F.tracks), [0, 2], track_label=node_F.label,
                    line=dict(color='#FF0000'), uncertainty=True)
plotter.plot_tracks(reduce_tracks(node_G.tracks), [0, 2], track_label=node_G.label,
                    line=dict(color='#0000FF'), uncertainty=True)
plotter.plot_sensors(sensor_set)
plotter.fig


# %%
# Network Architecture Example
# ----------------------------
# A network architecture represents the full physical network behind an information architecture. For an analogy, we
# might compare an edge in the information architecture to the connection between a sender and recipient of an email.
# Much of the time, we only care about the information architecture and not the actual mechanisms behind delivery of the
# email, which is similar in nature to the network architecture.
# Some nodes have the sole purpose of receiving and re-transmitting data on to other nodes in the network. We call
# these :class:`~.RepeaterNode`s. Additionally, any :class:`~.Node` present in the :class:`~.InformationArchitecture`
# must also be modelled in the :class:`~.NetworkArchitecture`.

# %%
# Network Architecture Nodes
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# In this example, we will add six new :class:`~.RepeaterNode`s to the existing structure to create our corresponding
# :class:`~.NetworkArchitecture`.

from stonesoup.architecture.node import RepeaterNode

repeaternode1 = RepeaterNode(label='RepeaterNode 1')
repeaternode2 = RepeaterNode(label='RepeaterNode 2')
repeaternode3 = RepeaterNode(label='RepeaterNode 3')
repeaternode4 = RepeaterNode(label='RepeaterNode 4')
repeaternode5 = RepeaterNode(label='RepeaterNode 5')
repeaternode6 = RepeaterNode(label='RepeaterNode 6')

node_A = SensorNode(sensor=sensor_set[0], label='SensorNode A')
node_B = SensorNode(sensor=sensor_set[2], label='SensorNode B')

node_C_tracker = copy.deepcopy(tracker)
node_C_tracker.detector = FusionQueue()
node_C = FusionNode(tracker=node_C_tracker, fusion_queue=node_C_tracker.detector, latency=0,
                    label='FusionNode C')

node_D = SensorNode(sensor=sensor_set[1], label='SensorNode D')
node_E = SensorNode(sensor=sensor_set[3], label='SensorNode E')

node_F_tracker = copy.deepcopy(tracker)
node_F_tracker.detector = FusionQueue()
node_F = FusionNode(tracker=node_F_tracker, fusion_queue=node_F_tracker.detector, latency=0)

node_H = SensorNode(sensor=sensor_set[4])

fq = FusionQueue()
track_tracker = MultiTargetTracker(
    initiator, deleter, Tracks2GaussianDetectionFeeder(fq),
    data_associator, detection_track_updater)

node_G = FusionNode(tracker=track_tracker, fusion_queue=fq, latency=0)


# %%
# Network Architecture Edges
# ^^^^^^^^^^^^^^^^^^^^^^^^^^

edges = Edges([Edge((node_A, repeaternode1), edge_latency=0.5),
               Edge((repeaternode1, node_C), edge_latency=0.5),
               Edge((node_B, repeaternode3)),
               Edge((repeaternode3, node_C)),
               Edge((node_A, repeaternode2), edge_latency=0.5),
               Edge((repeaternode2, node_C)),
               Edge((repeaternode1, repeaternode2)),
               Edge((node_D, repeaternode4)),
               Edge((repeaternode4, node_F)),
               Edge((node_E, repeaternode5)),
               Edge((repeaternode5, node_F)),
               Edge((node_C, node_G), edge_latency=0),
               Edge((node_F, node_G), edge_latency=0),
               Edge((node_H, repeaternode6)),
               Edge((repeaternode6, node_G))
               ])


# %%
# Network Architecture Functionality
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# A network architecture provides a representation of all nodes in a network - the corresponding
# information architecture is made up of a subset of these nodes.
#
# To aid in modelling this in Stone Soup, the NetworkArchitecture class has a property
# `information_arch`: an InformationArchitecture object representing the information
# architecture that is underpinned by the network architecture. This means that a
# NetworkArchitecture object requires two Edge lists: one set of edges representing the
# network architecture, and another representing links in the information architecture. To
# ease the setup of these edge lists, there are multiple options for how to instantiate a
# :class:`~.NetworkArchitecture`
#
# - Firstly, providing the :class:`~.NetworkArchitecture` with an `edge_list` for the network
#   architecture (an Edges object), and a pre-fabricated InformationArchitecture object, which must
#   be provided as property `information_arch`.
#
# - Secondly, by providing the NetworkArchitecture with two `edge_list` values: one for the network
#   architecture and one for the information architecture.
#
# - Thirdly, by providing just a set of edges for the network architecture. In this case,
#   the NetworkArchitecture class will infer which nodes in the network architecture are also
#   part of the information architecture, and form an edge set by calculating the fastest routes
#   (the lowest latency) between each set of nodes in the architecture. Warning: this method is for
#   ease of use, and may not represent the information architecture you are designing - it is
#   best to check by plotting the generated information architecture.
#


# %%
# Instantiate Network Architecture
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

from stonesoup.architecture import NetworkArchitecture
network_architecture = NetworkArchitecture(edges=edges, current_time=start_time)


# %%
# Simulate measurement and propagation across the Network Architecture
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

for time in timesteps:
    network_architecture.measure(truths, noise=True)
    network_architecture.propagate(time_increment=1)


# %%
# Plot the Network Architecture
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The plot below displays the Network Architecture we have built. This includes all Nodes,
# including those that do not feature in the Information Architecture (the repeater nodes).

network_architecture

# %%
# Plot the Network Architecture's Information Architecture
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Next we plot the information architecture that is underpinned by the network architecture. The
# nodes of the information architecture are a subset of the nodes from the network architecture.
# An edge in the Information Architecture could be equivalent
# to a physical edge between two nodes in the Network Architecture, but it could also be a
# representation of multiple edges and nodes that a data piece would be transferred through in
# order to pass from a sender to a recipient node.

network_architecture.information_arch

# %%
# Plot Tracks at the Fusion Nodes (Network Architecture)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

plotter = Plotterly()
plotter.plot_ground_truths(truths, [0, 2])
plotter.plot_tracks(reduce_tracks(node_C.tracks), [0, 2], track_label=node_C.label,
                    line=dict(color='#00FF00'), uncertainty=True)
plotter.plot_tracks(reduce_tracks(node_F.tracks), [0, 2], track_label=node_F.label,
                    line=dict(color='#FF0000'), uncertainty=True)
plotter.plot_tracks(reduce_tracks(node_G.tracks), [0, 2], track_label=node_G.label,
                    line=dict(color='#0000FF'), uncertainty=True)
plotter.plot_sensors(sensor_set)
plotter.fig
