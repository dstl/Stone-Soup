from ..base import Property
from .base import Type
from ..sensor.sensor import Sensor

from typing import Set, List, Collection, Union
import networkx as nx
import plotly.graph_objects as go


class ProcessingNode(Type):
    """A node that does not measure new data, but does process data it receives"""
    # Latency property could go here


class RepeaterNode(Type):
    """A node which simply passes data along to others, without manipulating the data itself. """
    # Latency property could go here


class Architecture(Type):
    node_set: Set[Union[Sensor, ProcessingNode, RepeaterNode]] = Property(
        default=None,
        doc="A Set of all nodes involved, each of which may be a sensor, processing node, "
            "or repeater node. ")
    edge_list: Collection = Property(
        default=None,
        doc="A Collection of edges between nodes. For A to be connected to B we would have (A, B)"
            "be a member of this list. ")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.node_set:
            self.node_set = set()
        if isinstance(self.edge_list, Collection) and not isinstance(self.edge_list, List):
            self.edge_list = list(self.edge_list)

    def plot(self):

        return

    @property
    def density(self):
        num_nodes = len(self.node_set)
        num_edges = len(self.edge_list)
        architecture_density = num_edges/((num_nodes*(num_nodes-1))/2)
        return architecture_density



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