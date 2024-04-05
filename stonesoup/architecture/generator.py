import copy
import operator
import random
import warnings

from datetime import datetime

import numpy as np

from stonesoup.architecture import SensorNode, FusionNode, Edge, Edges, InformationArchitecture, \
    NetworkArchitecture
from stonesoup.architecture.edge import FusionQueue
from stonesoup.architecture.node import SensorFusionNode, RepeaterNode
from stonesoup.base import Base, Property
from stonesoup.feeder.track import Tracks2GaussianDetectionFeeder
from stonesoup.sensor.sensor import Sensor
from stonesoup.tracker import Tracker


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
    base_sensor: Sensor = Property(
        doc="Sensor class object that will be duplicated to create multiple sensors. Position of "
            "this sensor is used with 'sensor_max_distance' to calculate a position for "
            "duplicated sensors.",
        default=None)
    sensor_max_distance: tuple = Property(
        doc="Max distance each sensor can be from base_sensor.position. Should be a tuple of "
            "length equal to len(base_sensor.position_mapping)",
        default=None)
    base_tracker: Tracker = Property(
        doc="Tracker class object that will be duplicated to create multiple trackers. "
            "Should have detector=None.",
        default=None)
    iteration_limit: int = Property(
        doc="Limit for the number of iterations the generate_edgelist() method can make when "
            "attempting to build a suitable graph.",
        default=10000)
    allow_invalid_graph: bool = Property(
        doc="Bool where True allows invalid graphs to be returned without throwing an error. "
            "False by default",
        default=False)
    n_archs: int = Property(
        doc="Tuple containing a minimum and maximum value for the number of routes created in the "
            "network architecture to represent a single edge in the information architecture.",
        default=2)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_nodes = sum(self.node_ratio)
        self.n_sensor_nodes = self.node_ratio[0]
        self.n_fusion_nodes = self.node_ratio[2]
        self.n_sensor_fusion_nodes = self.node_ratio[1]

        self.n_edges = np.ceil(self.n_nodes * self.mean_degree * 0.5)

        if self.sensor_max_distance is None:
            self.sensor_max_distance = tuple(np.zeros(len(self.base_sensor.position_mapping)))
        if self.arch_type not in ['decentralised', 'hierarchical']:
            raise ValueError('arch_style must be "decentralised" or "hierarchical"')

    def generate(self):
        edgelist, degrees = self._generate_edgelist()

        nodes, edgelist = self._assign_nodes(degrees, edgelist)

        archs = list()
        for architecture in nodes.keys():
            arch = self._generate_architecture(nodes[architecture], edgelist)
            archs.append(arch)
        return archs

    def _generate_architecture(self, nodes, edgelist):
        edges = []
        for t in edgelist:
            edge = Edge((nodes[t[0]], nodes[t[1]]))
            edges.append(edge)

        arch_edges = Edges(edges)

        arch = InformationArchitecture(arch_edges, self.start_time)
        return arch

    def _assign_nodes(self, degrees, edgelist):

        # Order nodes by target degree (number of other nodes passing data to it)
        reordered = []
        for node_no in degrees.keys():
            reordered.append((node_no, degrees[node_no]['target']))

        reordered.sort(key=operator.itemgetter(1))
        order = []
        for t in reordered:
            order.append(t[0])

        # Reorder so that nodes at the top of the information chain will not be sensor nodes if
        # possible
        top_nodes = [n for n in degrees.keys() if degrees[n]['source'] == 0]
        for n in top_nodes:
            order.remove(n)
            order.append(n)

        # Order of s/sf/f nodes
        components = ['s'] * self.n_sensor_nodes + \
                     ['sf'] * self.n_sensor_fusion_nodes + \
                     ['f'] * self.n_fusion_nodes

        # Create dictionary entry for each architecture copy
        nodes = {}
        for architecture in range(self.n_archs):
            nodes[architecture] = {}

        # Create Nodes
        for node_no, node_type in zip(order, components):

            if node_type == 's':
                pos = np.array(
                    [[p + random.uniform(-d, d)] for p, d in zip(self.base_sensor.position,
                                                                 self.sensor_max_distance)])
                for architecture in range(self.n_archs):
                    s = copy.deepcopy(self.base_sensor)
                    s.position = pos

                    node = SensorNode(sensor=s, label=str(node_no))
                    nodes[architecture][node_no] = node

            elif node_type == 'f':
                for architecture in range(self.n_archs):
                    t = copy.deepcopy(self.base_tracker)
                    fq = FusionQueue()
                    t.detector = Tracks2GaussianDetectionFeeder(fq)

                    node = FusionNode(tracker=t,
                                      fusion_queue=fq,
                                      label=str(node_no))

                    nodes[architecture][node_no] = node

            elif node_type == 'sf':
                pos = np.array(
                    [[p + random.uniform(-d, d)] for p, d in zip(self.base_sensor.position,
                                                                 self.sensor_max_distance)])

                for architecture in range(self.n_archs):
                    s = copy.deepcopy(self.base_sensor)
                    s.position = pos

                    t = copy.deepcopy(self.base_tracker)
                    fq = FusionQueue()
                    t.detector = Tracks2GaussianDetectionFeeder(fq)

                    node = SensorFusionNode(sensor=s,
                                            tracker=t,
                                            fusion_queue=fq,
                                            label=str(node_no))

                    nodes[architecture][node_no] = node

        new_edgelist = copy.copy(edgelist)

        if self.arch_type != 'hierarchical':
            for edge in edgelist:
                s = edge[0]
                t = edge[1]

                if isinstance(nodes[0][s], FusionNode) and type(nodes[0][t]) == SensorNode:
                    new_edgelist.remove((s, t))
                    new_edgelist.append((t, s))

        return nodes, new_edgelist

    def _generate_edgelist(self):
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
                    while source == target or (source, target) in edges or \
                            (target, source) in edges:
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


class NetworkArchitectureGenerator(InformationArchitectureGenerator):
    """
    Class that can be used to generate NetworkArchitecture classes given a set of input
    parameters.
    """
    n_routes: tuple = Property(
        doc="Tuple containing a minimum and maximum value for the number of routes created in the "
            "network architecture to represent a single edge in the information architecture.",
        default=(1, 2))

    def generate(self):
        edgelist, degrees = self._generate_edgelist()

        nodes, edgelist = self._assign_nodes(degrees, edgelist)

        nodes, edgelist = self._add_network(nodes, edgelist)

        archs = list()
        for architecture in nodes.keys():
            arch = self._generate_architecture(nodes[architecture], edgelist)
            archs.append(arch)
        return archs

    def _add_network(self, nodes, edgelist):
        network_edgelist = []
        i = 0
        for e in edgelist:
            # Choose number of routes between two information architecture nodes
            n = self.n_routes[0] if len(self.n_routes) == 1 else \
                random.randint(self.n_routes[0], self.n_routes[1])

            for route in range(n):
                r_lab = 'r' + str(i)
                network_edgelist.append((e[0], r_lab))
                network_edgelist.append((r_lab, e[1]))
                for architecture in nodes.keys():
                    r = RepeaterNode(label=r_lab)
                    nodes[architecture][r_lab] = r
                i += 1

        return nodes, network_edgelist

    def _generate_architecture(self, nodes, edgelist):
        edges = []
        for t in edgelist:
            edge = Edge((nodes[t[0]], nodes[t[1]]))
            edges.append(edge)

        arch_edges = Edges(edges)

        arch = NetworkArchitecture(arch_edges, self.start_time)
        return arch
