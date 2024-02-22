import operator
import random
import warnings

import networkx as nx
from datetime import datetime

import numpy as np

from stonesoup.architecture import SensorNode, FusionNode, Edge, Edges, InformationArchitecture, \
    NetworkArchitecture
from stonesoup.architecture.node import SensorFusionNode, RepeaterNode
from stonesoup.base import Base, Property


class InformationArchitectureGenerator(Base):
    """
    Class that can be used to generate InformationArchitecture classes given a set of input
    parameters.
    """
    arch_type: str = Property(
        doc="Type of architecture to be modelled. Currently only 'hierarchical' and "
            "'decentralised' are supported.",
        default='decentralised')
    start_time: datetime = Property(
        doc="Start time of simulation to be passed to the Architecture class.",
        default=datetime.now())
    node_ratio: tuple = Property(
        doc="Tuple containing the number of each type of node, in the order of (sensor nodes, "
            "sensor fusion nodes, fusion nodes).",
        default=None)
    mean_degree: float = Property(
        doc="Average (mean) degree of nodes in the network.",
        default=None)
    sensors: list = Property(
        doc="A list of sensors that are used to create SensorNodes.",
        default=None)
    trackers: list = Property(
        doc="A list of trackers that are used to create FusionNodes.",
        default=None)
    iteration_limit: int = Property(
        doc="Limit for the number of iterations the generate_edgelist() method can make when "
            "attempting to build a suitable graph.",
        default=10000)
    allow_invalid_graph: bool = Property(
        doc="Bool where True allows invalid graphs to be returned without throwing an error. "
            "False by default",
        default=False)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_nodes = sum(self.node_ratio)
        self.n_sensor_nodes = self.node_ratio[0]
        self.n_fusion_nodes = self.node_ratio[2]
        self.n_sensor_fusion_nodes = self.node_ratio[1]

        self.n_edges = np.ceil(self.n_nodes * self.mean_degree * 0.5)

        if self.arch_type not in ['decentralised', 'hierarchical']:
            raise ValueError('arch_style must be "decentralised" or "hierarchical"')

    def generate(self):
        edgelist, degrees = self.generate_edgelist()

        nodes = self.assign_nodes(degrees)

        arch = self.generate_architecture(nodes, edgelist)

        return arch

    def generate_architecture(self, nodes, edgelist):
        edges = []
        for t in edgelist:
            edge = Edge((nodes[t[0]], nodes[t[1]]))
            edges.append(edge)

        arch_edges = Edges(edges)

        arch = InformationArchitecture(arch_edges, self.start_time)
        return arch

    def assign_nodes(self, degrees):
        reordered = []
        for node_no in degrees.keys():
            reordered.append((node_no, degrees[node_no]['degree']))

        reordered.sort(key=operator.itemgetter(1))
        order = []
        for t in reordered:
            order.append(t[0])

        components = ['s'] * self.n_sensor_nodes + \
                     ['sf'] * self.n_sensor_fusion_nodes + \
                     ['f'] * self.n_fusion_nodes

        nodes = {}
        n_sensors = 0
        n_trackers = 0
        for node_no, type in zip(order, components):
            if type == 's':
                node = SensorNode(sensor=self.sensors[n_sensors], label=str(node_no))
                n_sensors += 1
                nodes[node_no] = node

            if type == 'f':
                node = FusionNode(tracker=self.trackers[n_trackers],
                                  fusion_queue=self.trackers[n_trackers].detector.reader)
                n_trackers += 1
                nodes[node_no] = node

            if type == 'sf':
                node = SensorFusionNode(sensor=self.sensors[n_sensors], label=str(node_no),
                                        tracker=self.trackers[n_trackers],
                                        fusion_queue=self.trackers[n_trackers].detector.reader)
                n_sensors += 1
                n_trackers += 1
                nodes[node_no] = node

        return nodes

    def generate_edgelist(self):
        count = 0
        edges = []
        sources = []
        targets = []
        if self.arch_type == 'hierarchical':
            for i in range(1, self.n_nodes):
                target = random.randint(0, i - 1)
                source = i
                edge = (source, target)
                edges.append(edge)
                sources.append(source)
                targets.append(target)

        if self.arch_type == 'decentralised':
            network_found = False
            while network_found is False and count < self.iteration_limit:
                i = 0
                nodes_used = {0}
                edges = []
                sources = []
                targets = []
                while i < self.n_edges:
                    source, target = -1, -1
                    while source == target or (source, target) in edges or (
                    target, source) in edges:
                        source = random.randint(0, self.n_nodes-1)
                        target = random.choice(list(nodes_used))
                        edge = (source, target)
                    edges.append(edge)
                    nodes_used |= {source, target}
                    sources.append(source)
                    targets.append(target)
                    i += 1

                if len(nodes_used) == self.n_nodes:
                    network_found = True
                count += 1

            if not network_found:
                if self.allow_invalid_graph:
                    warnings.warn("Unable to find valid graph within iteration limit. Returned "
                                 "network does not meet requirements")
                else:
                    raise ValueError("Unable to find valid graph within iteration limit. Returned "
                                     "network does not meet requirements")

        degrees = {}
        for node in range(self.n_nodes):
            degrees[node] = {'source': sources.count(node), 'target': targets.count(node),
                             'degree': sources.count(node) + targets.count(node)}

        return edges, degrees

    @staticmethod
    def plot_edges(edges):
        G = nx.DiGraph()
        G.add_edges_from(edges)
        nx.draw(G)


class NetworkArchitectureGenerator(InformationArchitectureGenerator):
    """
    Class that can be used to generate NetworkArchitecture classes given a set of input
    parameters.
    """
    n_routes: tuple = Property(
        doc="Tuple containing a minimum and maximum value for the number of routes created in the "
            "network architecture to represent a single edge in the information architecture.",
        default=(1, 2))

    def generate_architecture(self, nodes, edgelist):
        edges = []

        for e in edgelist:
            # Choose number of routes between two information architecture nodes
            n = self.n_routes if len(self.n_routes) == 1 else \
                random.randint(self.n_routes[0], self.n_routes[1])

            for route in range(n):
                r = RepeaterNode()
                edge1 = Edge((nodes[e[0]], r))
                edge2 = Edge((r, nodes[e[1]]))
                edges.append(edge1)
                edges.append(edge2)

        arch_edges = Edges(edges)
        arch = NetworkArchitecture(edges=arch_edges, current_time=self.start_time)

        return arch
