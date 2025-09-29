#!/usr/bin/env python
# coding: utf-8

"""
===============================================
1 - Introduction to Architectures in Stone Soup
===============================================
"""

# %%
# Introduction
# ------------
#
# The architecture package in Stone Soup provides functionality to build information and network
# architectures, enabling the user to simulate sensing, propagation and fusion of data.
# Architectures are modelled by defining the nodes in the architecture, and edges that represent
# connections between nodes.
#
# Nodes
# -----
#
# Nodes represent points in the architecture that collect, process (fuse), or simply forward on 
# data. Before advancing, a few definitions are required:
#
# - Relationships between nodes are defined as parent-child. In a directed graph, an edge from
#   node A to node B means that data is passed from the child node, A, to the parent node, B.
#
# - The children of node A, denoted :math:`children(A)`, is defined as the set of nodes B, where
#   there exists a direct edge from node B to node A (set of nodes that A receives data from).
#
# - The parents of node A, denoted :math:`parents(A)`, is defined as the set of nodes B, where
#   there exists a direct edge from node A to node B (set of nodes that A passes data to).
#
# Different types of node can provide different functionality in the architecture. The following
# are available in Stone Soup:
#
# - :class:`~.SensorNode`: makes detections of targets and propagates data onwards through the
#   architecture.
#
# - :class:`~.FusionNode`: receives data from child nodes, and fuses to achieve a fused result.
#   The fused result can be propagated onwards.
#
# - :class:`~.SensorFusionNode`: has the functionality of both a SensorNode and a FusionNode.
#
# - :class:`~.RepeaterNode`: does not create or fuse data, but only propagates it onwards. 
#   It is only used in network architectures.
#
# Set up and Node Properties
# ^^^^^^^^^^^^^^^^^^^^^^^^^^

from stonesoup.architecture.node import Node

node_A = Node(label='Node A')
node_B = Node(label='Node B')
node_C = Node(label='Node C')

# %%
# The :class:`~.Node` base class contains several properties. The `latency` property gives 
# functionality to simulate processing latency at the node. The rest of the properties (`label`, 
# `position`, `colour`, `shape`, `font_size`, `node_dim`), are used for graph plotting.

node_A.colour = '#006494'

node_A.shape = 'hexagon'

# %%
# :class:`~.SensorNode` and :class:`~.FusionNode` objects have additional properties that must be
# defined. A :class:`~.SensorNode` must be given an additional `sensor` property - this must be a
# :class:`~.Sensor`. A :class:`~.FusionNode` has two additional properties: `tracker` and
# `fusion_queue`.`tracker`  must both be :class:`~.Tracker`\s - the main tracker manages the 
# fusion at the node, while the `fusion_queue` property is a :class:`~.FusionQueue` by default - 
# this manages the inflow of data from child nodes.
#
# Edges
# -----
# An edge represents a link between two nodes in an architecture. An :class:`~.Edge` contains a
# property `nodes`: a tuple of :class:`~.Node` objects where the first entry in the tuple is
# the child node and the second is the parent. Edges in Stone Soup are directional (data can
# flow only in one direction), with data flowing from child to parent. Edge objects also
# contain a `latency` property to enable simulation of latency caused by sending a message, 
# separately to node latency. 

from stonesoup.architecture.edge import Edge

edge1 = Edge(nodes=(node_B, node_A))
edge2 = Edge(nodes=(node_C, node_A))

# %%
# :class:`~.Edges` is a container class for :class:`~.Edge` objects. :class:`~.Edges` has an
# `edges` property - a list of :class:`~.Edge` objects. An :class:`~.Edges` object is required
# to pass into an :class:`~.Architecture`.

from stonesoup.architecture.edge import Edges

edges = Edges(edges=[edge1, edge2])


# %%
# Architecture
# ------------
# Architecture classes manage the simulation of data propagation across a network. Two
# architecture classes are available in Stone Soup: :class:`~.InformationArchitecture` and
# :class:`~.NetworkArchitecture`. Information architecture simulates how
# information is shared across the network, only considering nodes that create or modify
# information. Network architecture simulates how data is actually
# propagated through a network. All nodes are considered including nodes that don't open or modify
# any data.
#
# A good analogy for the two is receiving a parcel via post. The "information architecture" is 
# from sender to receiver, the former who creates the parcel, and the latter who opens it. 
# However, between these are many unseen but crucial steps which form the "network architecture". 
# In this analogy, the postman never does anything with the parcel besides deliver it, so functions 
# like a Stone Soup :class:`~.RepeaterNode`. 
# 
#
# Architecture classes contain an `edges` property - this must be an :class:`~.Edges` object.
# The `current_time` property of an Architecture instance maintains the current time within the
# simulation. By default, this begins at the current time of the operating system.

# sphinx_gallery_thumbnail_path = '_static/sphinx_gallery/ArchTutorial_1.png'

from stonesoup.architecture import InformationArchitecture

arch = InformationArchitecture(edges=edges)
arch