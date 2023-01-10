from ..base import Property
from .base import Type
from ..sensor.sensor import Sensor

from typing import Set, List, Collection
import networkx as nx
import plotly.graph_objects as go


class Node(Type):
    """Base node class"""


class SensorNode(Node):
    """A node corresponding to a Sensor. Fresh data is created here,
    and possibly processed as well"""
    sensor: Sensor = Property(doc="Sensor corresponding to this node")


class ProcessingNode(Type):
    """A node that does not measure new data, but does process data it receives"""
    # Latency property could go here


class RepeaterNode(Type):
    """A node which simply passes data along to others, without manipulating the data itself. """
    # Latency property could go here


class Architecture(Type):
    edge_list: Collection = Property(
        default=None,
        doc="A Collection of edges between nodes. For A to be connected to B we would have (A, B)"
            "be a member of this list. Default is None")
    node_set: Set[Node] = Property(
        default=None,
        doc="A Set of all nodes involved, each of which may be a sensor, processing node, "
            "or repeater node. If provided, used to check all Nodes given are included "
            "in edges of the graph. Default is None")
    force_connected: bool = Property(
        default=True,
        doc="If True, the undirected version of the graph must be connected, ie. all nodes should "
            "be connected via some path. Set this to False to allow an unconnected architecture. "
            "Default is True"
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(self.edge_list, Collection) and not isinstance(self.edge_list, List):
            self.edge_list = list(self.edge_list)
        if self.edge_list and len(self.edge_list) > 0:
            self.di_graph = nx.to_networkx_graph(self.edge_list, create_using=nx.DiGraph)
            if self.force_connected and not self.is_connected:
                raise ValueError("The graph is not connected. Use force_connected=False, "
                                 "if you wish to override this requirement")
        else:
            if self.node_set:
                raise TypeError("Edge list must be provided, if a node set is. ")
            self.di_graph = nx.DiGraph()
        if self.node_set:
            if not set(self.di_graph.nodes) == self.node_set:
                raise ValueError("Provided node set does not match nodes on graph")
        else:
            self.node_set = set(self.di_graph.nodes)

    def plot(self):

        return

    @property
    def density(self):
        """Returns the density of the graph, ie. the proportion of possible edges between nodes
        that exist in the graph"""
        num_nodes = len(self.node_set)
        num_edges = len(self.edge_list)
        architecture_density = num_edges/((num_nodes*(num_nodes-1))/2)
        return architecture_density

    @property
    def is_hierarchical(self):
        """Returns `True` if the :class:`Architecture` is hierarchical, otherwise `False`"""
        if len(list(nx.simple_cycles(self.di_graph))) > 0 or not self.is_connected:
            return False
        else:
            return True

    @property
    def is_connected(self):
        return nx.is_connected(self.to_undirected)

    @property
    def to_undirected(self):
        return self.di_graph.to_undirected()

    def __len__(self):
        return len(self.di_graph)


class InformationArchitecture(Architecture):
    """The architecture for how information is shared through the network. Node A is "
    "connected to Node B if and only if the information A creates by processing and/or "
    "sensing is received and opened by B without modification by another node. """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for node in self.node_set:
            if isinstance(node, RepeaterNode):
                raise TypeError("Information architecture should not contain any repeater nodes")


class NetworkArchitecture(Architecture):
    """The architecture for how data is propagated through the network. Node A is connected "
            "to Node B if and only if A sends its data through B. """


class CombinedArchitecture(Type):
    """Contains an information and a network architecture that pertain to the same scenario. """
    information_architecture: InformationArchitecture = Property(
        doc="The information architecture for how information is shared. ")
    network_architecture: NetworkArchitecture = Property(
        doc="The architecture for how data is propagated through the network. Node A is connected "
            "to Node B if and only if A sends its data through B. "
    )