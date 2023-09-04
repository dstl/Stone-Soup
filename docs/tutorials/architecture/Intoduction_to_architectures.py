#!/usr/bin/env python
# coding: utf-8

"""
===========================================
Introduction to Architectures in Stone Soup
===========================================
"""

import tempfile
import numpy as np
import random
from ordered_set import OrderedSet
from datetime import datetime, timedelta
import copy


# %%
# Introduction
# ------------
#
# :class:`~.stonesoup.architecture` provides functionality to build Information and Network
# architectures, to simulate sensing, propagation and fusion of data. Architectures are modelled
# by defining the nodes in the architecture, and edges that represent links between nodes.
#
# Nodes
# -----
#
# Nodes represent points in the architecture that process the in some way. Before advancing, a
# few definitions are required:
#
# - Relationships between nodes are defined as parent-child. In a directed graph, an edge from
# node A to node B informs that data is passed from the child node, A, to the parent node, B.
# - The children of node A, TODO:Set notation here,
#  is defined as the set of nodes which pass data to node A.
# - The parents of node A, TODO:Set notation here,
#  is defined as the set of nodes which node A passes data to.
#
# Different types of node can provide different functionality in the architecture. The following
# are available in stonesoup:
#
# - SensorNode: makes detections of targets and propagates data onwards through the architecture.
# - FusionNode: receives data from child nodes, and fuses to achieve a fused result. Fused result
# can be propagated onwards.
# - SensorFusionNode: has the functionality of both a SensorNode and a FusionNode.
# - RepeaterNode: carries out no processing of data. Propagates forwards data that it has
# received. Cannot be used in information architecture.
#
# Set up and Node Properties
# ^^^^^^^^^^^^^^^^^^^^^^^^^^

from stonesoup.architecture.node import Node

node_A = Node(label='Node A')
node_B = Node(label='Node B')
node_C = Node(label='Node C')

# %%
# The Node base class contains several properties. The `latency` property gives functionality to
# simulate processing latency at the node. The rest of the properties (label, position, colour,
# shape, font_size, node_dim), are optional and primarily used for graph plotting.

node_A.colour = '#006494'

node_A.shape = 'hexagon'

# %%
# SensorNodes and FusionNodes have additional properties that must be defined. A SensorNode must
# be given an additional `sensor` property - this must be a TODO Sensorclass.
#  A FusionNode has two additional properties: `tracker` and `fusion_queue`. `tracker` must be
#  a TODO Trackerclass - the tracker manages the fusion at the node. The `fusion_queue` property
#   is a #TODO FusionQueueclass by default - this manages the inflow of data from child nodes.
#
# Edges
# -----
# An edge represents a link between two nodes in an architecture. An TODO Edgeclass contains a
#  property `nodes`: a tuple of TODO Nodeclass objects where the first entry in the tuple is
#   the child node and the second is the parent. Edges in stonesoup are directional (data can
#   flow only in one direction), with data flowing from child to parent. Edge objects also
#   contain a `latency` property to enable simulation of latency caused by sending a message.

from stonesoup.architecture.edge import Edge

edge1 = Edge(nodes=(node_B, node_A))
edge2 = Edge(nodes=(node_C, node_A))

# %%
# TODO Edgesclass is a container class for #TODO Edgeclass objects. #TODO Edgesclass has an
#  `edges` property - a list of TODO Edgeclass objects. An #TODO Edgesclass object is required
#   to pass into an #TODO Architectureclass

from stonesoup.architecture.edge import Edges

edges = Edges(edges=[edge1, edge2])


# Architecture
# ------------
# Architecture classes manage the simualtion of data propagation across a network. Two
# architecture classes are available in stonesoup: #TODO InformationArchitectureclass and
# NetworkArchitectureclass. Information architecture simulates the architecture of how
# information is shared across the network, only considering nodes that create or modify
# information. Network architecture simulates the architecture of how data is propagated through
# a network. All nodes are considered including nodes that only propagate received data onwards.
#
# Architecture classes contain an `edges` property - this must be a #TODO Edgesclass object.
#  The `current_time` property of an Architecture class maintains the current time within the
#  simulation. By default this begins at the current time of the operating system.

from stonesoup.architecture import InformationArchitecture

arch = InformationArchitecture(edges=edges)
arch.plot(tempfile.gettempdir(), save_plot=False)


#TODO Add simple plot image here

# %%
# A Sensor Fusion Information Architecture example
# ------------------------------------------------
# Using the classes detailed above, we can build an example of an information architecture and
# simulate data detection, propagation and fusion.
#
# Generate ground truth
# ^^^^^^^^^^^^^^^^^^^^^

from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
    ConstantVelocity
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState

start_time = datetime.now().replace(microsecond=0)
np.random.seed(1990)
random.seed(1990)

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
# Create sensors
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
# Build Track-Tracker
# ^^^^^^^^^^^^^^^^^^^

from stonesoup.updater.wrapper import DetectionAndTrackSwitchingUpdater
from stonesoup.updater.chernoff import ChernoffUpdater
from stonesoup.feeder.track import Tracks2GaussianDetectionFeeder

track_updater = ChernoffUpdater(None)
detection_updater = ExtendedKalmanUpdater(None)
detection_track_updater = DetectionAndTrackSwitchingUpdater(None, detection_updater, track_updater)


fq = FusionQueue()

track_tracker = MultiTargetTracker(
    initiator, deleter, Tracks2GaussianDetectionFeeder(fq), data_associator,
    detection_track_updater)

# ## Build the Architecture
# With ground truths, sensors and trackers built, we can design an architecture and simulate
# running it with some data.
#
# Build nodes
# ^^^^^^^^^^^

from stonesoup.architecture.node import SensorNode, SensorFusionNode, FusionNode
from stonesoup.architecture.edge import FusionQueue

# Sensor Nodes
node_A = SensorNode(sensor=sensor_set[0], label='Node A')
node_B = SensorNode(sensor=sensor_set[1], label='Node B')
node_D = SensorNode(sensor=sensor_set[2], label='Node D')
node_E = SensorNode(sensor=sensor_set[3], label='Node E')
node_H = SensorNode(sensor=sensor_set[4], label='Node H')

# Fusion Nodes
node_C_tracker = copy.deepcopy(tracker)
node_C_tracker.detector = FusionQueue()
node_C = FusionNode(tracker=node_C_tracker, fusion_queue=node_C_tracker.detector, latency=0, label='Node C')

node_F_tracker = copy.deepcopy(tracker)
node_F_tracker.detector = FusionQueue()
node_F = FusionNode(tracker=node_F_tracker, fusion_queue=node_F_tracker.detector, latency=0, label='Node F')

node_G = FusionNode(tracker=track_tracker, fusion_queue=FusionQueue(), latency=0, label='Node G')

# Sensor Fusion Node
node_I_tracker = copy.deepcopy(tracker)
node_I_tracker.detector = FusionQueue()
node_I = SensorFusionNode(sensor=sensor_set[5], tracker=node_I_tracker,
                          fusion_queue=node_I_tracker.detector, label='Node I')

# %%
# Build Edges
# ^^^^^^^^^^^

edges = Edges([Edge((node_A, node_C)),
               Edge((node_B, node_C)),
               Edge((node_D, node_F)),
               Edge((node_E, node_F)),
               Edge((node_C, node_G), edge_latency=0),
               Edge((node_F, node_G), edge_latency=0),
               Edge((node_H, node_I)),
               Edge((node_I, node_G))])


# %%
# Build Information Architecture
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

arch = InformationArchitecture(edges)
arch.plot(tempfile.gettempdir())

