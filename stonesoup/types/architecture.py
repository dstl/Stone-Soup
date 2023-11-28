from abc import abstractmethod
from ..base import Property
from .base import Type
from ..sensor.sensor import Sensor
from ..types.groundtruth import GroundTruthState
from ..types.detection import TrueDetection, Clutter, Detection
from ..types.hypothesis import Hypothesis

from typing import List, Collection, Tuple, Set, Union, Dict
import numpy as np
import networkx as nx
import graphviz
from string import ascii_uppercase as auc
from datetime import datetime


class Node(Type):
    """Base node class"""
    label: str = Property(
        doc="Label to be displayed on graph",
        default=None)
    position: Tuple[float] = Property(
        default=None,
        doc="Cartesian coordinates for node")
    colour: str = Property(
        default=None,
        doc='Colour to be displayed on graph')
    shape: str = Property(
        default=None,
        doc='Shape used to display nodes')
    data_held: Dict[datetime: Union[Detection, Hypothesis]] = Property(
        default=None,
        doc='Data or information held by this node')

    def __init__(self):
        if not self.data_held:
            self.data_held = []

    def update(self, time, data):
        if not isinstance(data, Detection) and not isinstance(data, Hypothesis):
            raise TypeError("Data must be a Detection or Hypothesis")
        if not isinstance(time, datetime):
            raise TypeError("Time must be a datetime object")
        if data not in self.data_held:
            self.data_held[time] = data


class SensorNode(Node):
    """A node corresponding to a Sensor. Fresh data is created here"""
    sensor: Sensor = Property(doc="Sensor corresponding to this node")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.colour:
            self.colour = '#1f77b4'
        if not self.shape:
            self.shape = 'square'


class ProcessingNode(Node):
    """A node that does not measure new data, but does process data it receives"""
    # Latency property could go here
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.colour:
            self.colour = '#006400'
        if not self.shape:
            self.shape = 'hexagon'


class SensorProcessingNode(SensorNode, ProcessingNode):
    """A node that is both a sensor and also processes data"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.colour:
            self.colour = 'something'  # attr dict in Architecture.__init__ also needs updating
        if not self.shape:
            self.shape = 'something'


class RepeaterNode(Node):
    """A node which simply passes data along to others, without manipulating the data itself. """
    # Latency property could go here
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.colour:
            self.colour = '#ff7f0e'
        if not self.shape:
            self.shape = 'circle'


class Architecture(Type):
    edge_list: Collection = Property(
        default=None,
        doc="A Collection of edges between nodes. For A to be connected to B we would have (A, B)"
            "be a member of this list. Default is an empty list")
    current_time: datetime = Property(
        default=None,
        doc="The time which the instance is at for the purpose of simulation. "
            "This is increased by the propagate method, and defaults to the current system time")
    name: str = Property(
        default=None,
        doc="A name for the architecture, to be used to name files and/or title plots. Default is "
            "the class name")
    force_connected: bool = Property(
        default=True,
        doc="If True, the undirected version of the graph must be connected, ie. all nodes should "
            "be connected via some path. Set this to False to allow an unconnected architecture. "
            "Default is True")
    font_size: int = Property(
        default=8,
        doc='Font size for node labels')
    node_dim: tuple = Property(
        default=(0.5, 0.5),
        doc='Height and width of nodes for graph icons, default is (1, 1)')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(self.edge_list, Collection) and not isinstance(self.edge_list, List):
            self.edge_list = list(self.edge_list)
        if not self.edge_list:
            self.edge_list = []
        if not self.name:
            self.name = type(self).__name__
        if not self.current_time:
            self.current_time = datetime.now()

        self.di_graph = nx.to_networkx_graph(self.edge_list, create_using=nx.DiGraph)

        if self.force_connected and not self.is_connected and len(self) > 0:
            raise ValueError("The graph is not connected. Use force_connected=False, "
                             "if you wish to override this requirement")

        # Set attributes such as label, colour, shape, etc for each node
        last_letters = {'SensorNode': '', 'ProcessingNode': '', 'RepeaterNode': ''}
        for node in self.di_graph.nodes:
            if node.label:
                label = node.label
            else:
                label, last_letters = _default_label(node, last_letters)
                node.label = label
            attr = {"label": f"{label}", "color": f"{node.colour}", "shape": f"{node.shape}",
                    "fontsize": f"{self.font_size}", "width": f"{self.node_dim[0]}",
                    "height": f"{self.node_dim[1]}", "fixedsize": True}
            self.di_graph.nodes[node].update(attr)

    def descendants(self, node: Node):
        """Returns a set of all nodes to which the input node has a direct edge to"""
        if node not in self.all_nodes:
            raise ValueError("Node not in this architecture")
        descendants = set()
        for other in self.all_nodes:
            if (node, other) in self.edge_list:
                descendants.add(other)
        return descendants

    def ancestors(self, node: Node):
        """Returns a set of all nodes to which the input node has a direct edge from"""
        if node not in self.all_nodes:
            raise ValueError("Node not in this architecture")
        ancestors = set()
        for other in self.all_nodes:
            if (other, node) in self.edge_list:
                ancestors.add(other)
        return ancestors

    @abstractmethod
    def propagate(self, time_increment: float):
        raise NotImplementedError

    @property
    def all_nodes(self):
        return set(self.di_graph.nodes)

    @property
    def sensor_nodes(self):
        sensors = set()
        for node in self.all_nodes:
            if isinstance(node, SensorNode):
                sensors.add(node.sensor)
        return sensors

    def plot(self, dir_path, filename=None, use_positions=False, plot_title=False,
             bgcolour="lightgray", node_style="filled"):
        """Creates a pdf plot of the directed graph and displays it

        :param dir_path: The path to save the pdf and .gv files to
        :param filename: Name to call the associated files
        :param use_positions:
        :param plot_title: If a string is supplied, makes this the title of the plot. If True, uses
        the name attribute of the graph to title the plot. If False, no title is used.
        Default is False
        :param bgcolour: String containing the background colour for the plot.
        Default is "lightgray". See graphviz attributes for more information.
        One alternative is "white"
        :param node_style: String containing the node style for the plot.
        Default is "filled". See graphviz attributes for more information.
        One alternative is "solid"
        :return:
        """
        if use_positions:
            for node in self.di_graph.nodes:
                if not isinstance(node.position, Tuple):
                    raise TypeError("If use_positions is set to True, every node must have a "
                                    "position, given as a Tuple of length 2")
                attr = {"pos": f"{node.position[0]},{node.position[1]}!"}
                self.di_graph.nodes[node].update(attr)
        dot = nx.drawing.nx_pydot.to_pydot(self.di_graph).to_string()
        dot_split = dot.split('\n')
        dot_split.insert(1, f"graph [bgcolor={bgcolour}]")
        dot_split.insert(1, f"node [style={node_style}]")
        dot = "\n".join(dot_split)
        print(dot)
        if plot_title:
            if plot_title is True:
                plot_title = self.name
            elif not isinstance(plot_title, str):
                raise ValueError("Plot title must be a string, or True")
            dot = dot[:-2] + "labelloc=\"t\";\n" + f"label=\"{plot_title}\";" + "}"
        #  print(dot)
        if not filename:
            filename = self.name
        viz_graph = graphviz.Source(dot, filename=filename, directory=dir_path, engine='neato')
        viz_graph.view()

    @property
    def density(self):
        """Returns the density of the graph, ie. the proportion of possible edges between nodes
        that exist in the graph"""
        num_nodes = len(self.all_nodes)
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
        for node in self.all_nodes:
            if isinstance(node, RepeaterNode):
                raise TypeError("Information architecture should not contain any repeater nodes")

    def measure(self, ground_truths: Set[GroundTruthState], noise: Union[bool, np.ndarray] = True,
                **kwargs) -> Dict[SensorNode: Union[TrueDetection, Clutter]]:
        """ Similar to the method for :class:`~.SensorSuite`. Updates each node. """
        all_detections = dict()

        for sensor_node in self.sensor_nodes:

            all_detections[sensor_node] = sensor_node.sensor.measure(ground_truths, noise,
                                                                     **kwargs)

            attributes_dict = {attribute_name: sensor_node.sensor.__getattribute__(attribute_name)
                               for attribute_name in self.attributes_inform}

            for detection in all_detections[sensor_node]:
                detection.metadata.update(attributes_dict)

            for data in all_detections[sensor_node]:
                # The sensor acquires its own data instantly
                sensor_node.update(self.current_time, data)

        return all_detections

    def propagate(self, time_increment: float, failed_edges: Collection = None):
        """Performs a single step of the propagation of the measurements through the network"""
        self.current_time += datetime.timedelta(seconds=time_increment)
        for node in self.all_nodes:
            for descendant in self.descendants(node):
                if (node, descendant) in failed_edges:
                    # The network architecture / some outside factor prevents information from node
                    # being transferred to other
                    continue
                for data in node.data_held:
                    descendant.update(self.current_time, data)


class NetworkArchitecture(Architecture):
    """The architecture for how data is propagated through the network. Node A is connected "
            "to Node B if and only if A sends its data through B. """
    def propagate(self, time_increment: float):
        # Still have to deal with latency/bandwidth
        self.current_time += datetime.timedelta(seconds=time_increment)
        for node in self.all_nodes:
            for descendant in self.descendants(node):
                for data in node.data_held:
                    descendant.update(self.current_time, data)


class CombinedArchitecture(Type):
    """Contains an information and a network architecture that pertain to the same scenario. """
    information_architecture: InformationArchitecture = Property(
        doc="The information architecture for how information is shared. ")
    network_architecture: NetworkArchitecture = Property(
        doc="The architecture for how data is propagated through the network. Node A is connected "
            "to Node B if and only if A sends its data through B. ")

    def propagate(self, time_increment: float):
        # First we simulate the network
        self.network_architecture.propagate(time_increment)
        # Now we want to only pass information along in the information architecture if it
        # Was in the information architecture by at least one path.
        # Some magic here
        failed_edges = []  # return this from n_arch.propagate?
        self.information_architecture(time_increment, failed_edges)


def _default_label(node, last_letters):
    """Utility function to generate default labels for nodes, where none are given
    Takes a node, and a dictionary with the letters last used for each class,
    ie `last_letters['Node']` might return 'AA', meaning the last Node was labelled 'Node AA'"""
    node_type = type(node).__name__
    type_letters = last_letters[node_type]  # eg 'A', or 'AA', or 'ABZ'
    new_letters = _default_letters(type_letters)
    last_letters[node_type] = new_letters
    return node_type + ' ' + new_letters, last_letters


def _default_letters(type_letters) -> str:
    if type_letters == '':
        return 'A'
    count = 0
    letters_list = [*type_letters]
    # Move through string from right to left and shift any Z's up to A's
    while letters_list[-1 - count] == 'Z':
        letters_list[-1 - count] = 'A'
        count += 1
        if count == len(letters_list):
            return 'A' * (count + 1)
    # Shift current letter up by one
    current_letter = letters_list[-1 - count]
    letters_list[-1 - count] = auc[auc.index(current_letter) + 1]
    new_letters = ''.join(letters_list)
    return new_letters
