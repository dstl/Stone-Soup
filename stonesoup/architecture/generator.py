import copy
from datetime import datetime
import random

import networkx as nx
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
    Convenience class that can be used to generate one or multiple
    :class:`~.InformationArchitecture`s given a set of input parameters.
    The graph is generated randomly subject to the parameters such as `node_ratio` and
    `mean_degree`, rather than having to be user-defined.
    """
    arch_type: str = Property(
        doc="Type of architecture to be modelled. Currently only 'hierarchical' and "
            "'decentralised' are supported.",
        default='decentralised')
    start_time: datetime = Property(
        doc="Start time of simulation to be passed to the Architecture class.",
        default_factory=datetime.now)
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
        doc="How many architectures should be generated.",
        default=2)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_nodes = sum(self.node_ratio)
        self.n_sensor_nodes = self.node_ratio[0]
        self.n_fusion_nodes = self.node_ratio[2]
        self.n_sensor_fusion_nodes = self.node_ratio[1]

        self.n_edges = int(np.ceil(self.n_nodes * self.mean_degree * 0.5))

        if self.sensor_max_distance is None:
            self.sensor_max_distance = tuple(np.zeros(len(self.base_sensor.position_mapping)))
        if self.arch_type not in ['decentralised', 'hierarchical']:
            raise ValueError('arch_style must be "decentralised" or "hierarchical"')

    def generate(self):
        """
        Generate one or more InformationArchitecture objects based on the generator's parameters.

        Returns
        -------
        list
            List of generated InformationArchitecture objects.
        """
        edgelist, node_labels = self._generate_edgelist()

        nodes = self._assign_nodes(node_labels)

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

    def _assign_nodes(self, node_labels):

        nodes = {}
        for architecture in range(self.n_archs):
            nodes[architecture] = {}

        for label in node_labels:

            if label.startswith('f'):
                for architecture in range(self.n_archs):
                    t = copy.deepcopy(self.base_tracker)
                    fq = FusionQueue()
                    t.detector = Tracks2GaussianDetectionFeeder(fq)

                    node = FusionNode(tracker=t,
                                      fusion_queue=fq,
                                      label=label,
                                      latency=0)

                    nodes[architecture][label] = node

            elif label.startswith('sf'):

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
                                            label=label,
                                            latency=0)

                    nodes[architecture][label] = node

            elif label.startswith('s'):
                pos = np.array(
                    [[p + random.uniform(-d, d)] for p, d in zip(self.base_sensor.position,
                                                                 self.sensor_max_distance)])
                for architecture in range(self.n_archs):
                    s = copy.deepcopy(self.base_sensor)
                    s.position = pos

                    node = SensorNode(sensor=s,
                                      label=label,
                                      latency=0)
                    nodes[architecture][label] = node

            else:
                for architecture in range(self.n_archs):
                    node = RepeaterNode(label=label,
                                        latency=0)
                    nodes[architecture][label] = node

        return nodes

    def _generate_edgelist(self):

        edges = []

        nodes = ['f' + str(i) for i in range(self.n_fusion_nodes)] + \
                ['sf' + str(i) for i in range(self.n_sensor_fusion_nodes)] + \
                ['s' + str(i) for i in range(self.n_sensor_nodes)]

        valid = False

        if self.arch_type == 'hierarchical':
            while not valid:
                edges = []
                n = self.n_fusion_nodes + self.n_sensor_fusion_nodes

                for i, node in enumerate(nodes):
                    if i == 0:
                        continue
                    elif i == 1:
                        source = nodes[0]
                        target = nodes[1]
                    else:
                        if node.startswith('s') and not node.startswith('sf'):
                            source = node
                            target = nodes[random.randint(0, n - 1)]
                        else:
                            source = node
                            target = nodes[random.randint(0, i - 1)]

                    # Create edge
                    edge = (source, target)
                    edges.append(edge)

                # Logic checks on graph
                g = nx.DiGraph(edges)
                for f_node in ['f' + str(i) for i in range(self.n_fusion_nodes)]:
                    if g.in_degree(f_node) == 0:
                        break
                else:
                    valid = True

        else:

            while not valid:

                for i in range(1, self.n_edges + 1):
                    source = target = -1
                    if i < self.n_nodes:
                        source = nodes[i]
                        target = nodes[random.randint(
                            0, min(i - 1, len(nodes) - self.n_sensor_nodes - 1))]

                    else:
                        while source == target \
                                or (source, target) in edges \
                                or (target, source) in edges:
                            source = nodes[random.randint(0, len(nodes) - 1)]
                            target = nodes[random.randint(0, len(nodes) - self.n_sensor_nodes - 1)]

                    edges.append((source, target))

                # Logic checks on graph
                g = nx.DiGraph(edges)
                for f_node in ['f' + str(i) for i in range(self.n_fusion_nodes)]:
                    if g.in_degree(f_node) == 0:
                        break
                else:
                    valid = True

        return edges, nodes


class NetworkArchitectureGenerator(InformationArchitectureGenerator):
    """
    Convenience class that can be used to generate one or multiple
    :class:`~.NetworkArchitecture`s given a set of input parameters.
    The graph is generated randomly subject to the parameters such as
    `node_ratio` and `mean_degree`, rather than having to be user-defined.
    """
    n_routes: tuple = Property(
        doc="Tuple containing a minimum and maximum value for the number of routes created in the "
            "network architecture to represent a single edge in the information architecture.",
        default=(1, 2))

    def generate(self):
        """
        Generate one or more NetworkArchitecture objects based on the generator's parameters.

        Returns
        -------
        list
            List of generated NetworkArchitecture objects.
        """
        edgelist, node_labels = self._generate_edgelist()

        edgelist, node_labels = self._add_network(edgelist, node_labels)

        nodes = self._assign_nodes(node_labels)

        archs = list()
        for architecture in nodes.keys():
            arch = self._generate_architecture(nodes[architecture], edgelist)
            archs.append(arch)
        return archs

    def _add_network(self, edgelist, nodes):
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
                nodes.append(r_lab)
                i += 1

        return network_edgelist, nodes

    def _generate_architecture(self, nodes, edgelist):
        edges = []
        for t in edgelist:
            edge = Edge((nodes[t[0]], nodes[t[1]]))
            edges.append(edge)

        arch_edges = Edges(edges)

        arch = NetworkArchitecture(arch_edges, self.start_time)
        return arch
