from ..base import Property
from .base import Type
from ..sensor.sensor import Sensor

from typing import List, Collection, Tuple
import networkx as nx
import graphviz
from string import ascii_uppercase as auc


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


class SensorNode(Node):
    """A node corresponding to a Sensor. Fresh data is created here,
    and possibly processed as well"""
    sensor: Sensor = Property(doc="Sensor corresponding to this node")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.colour:
            self.colour = '#1f77b4'
        if not self.shape:
            self.shape = 'square'
        # We'd only need to set font_size to something here
        # if we wanted a different default for each class


class ProcessingNode(Node):
    """A node that does not measure new data, but does process data it receives"""
    # Latency property could go here
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.colour:
            self.colour = '#006400'
        if not self.shape:
            self.shape = 'triangle'


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
            "be a member of this list. Default is None")
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(self.edge_list, Collection) and not isinstance(self.edge_list, List):
            self.edge_list = list(self.edge_list)
        if not self.edge_list:
            self.edge_list = []
        if not self.name:
            self.name = type(self).__name__

        self.di_graph = nx.to_networkx_graph(self.edge_list, create_using=nx.DiGraph)

        if self.force_connected and not self.is_connected and len(self) > 0:
            raise ValueError("The graph is not connected. Use force_connected=False, "
                             "if you wish to override this requirement")

        # Set attributes such as label, colour, shape, etc for each node
        last_letters = {'SensorNode': 'A', 'ProcessingNode': 'A', 'RepeaterNode': 'A'}
        for node in self.di_graph.nodes:
            if node.label:
                label = node.label
            else:
                label, last_letters = _default_label(node, last_letters)
            attr = {"label": f"{label}", "color": f"{node.colour}", "shape": f"{node.shape}",
                    "fontsize": f"{self.font_size}"}
            self.di_graph.nodes[node].update(attr)

    @property
    def node_set(self):
        return set(self.di_graph.nodes)

    def plot(self, dir_path, filename=None, use_positions=True, plot_title=False,
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
            "to Node B if and only if A sends its data through B. ")


def _default_label(node, last_letters):  # Moved as we don't use "self", so no need to be a class method
    """Utility function to generate default labels for nodes, where none are given
    Takes a node, and a dictionary with the letters last used for each class,
    ie `last_letters['Node']` might return 'AA', meaning the last Node was labelled 'Node AA'"""
    node_type = type(node).__name__  # Neater way than all that splitting and indexing
    type_letters = last_letters[node_type]  # eg 'A', or 'AA', or 'ABZ'
    new_letters = _default_letters(type_letters)
    last_letters[node_type] = new_letters
    return node_type + ' ' + new_letters, last_letters


def _default_letters(type_letters):
    last_letter = type_letters[-1]
    if last_letter == 'Z':
        # do something recursive
        new_letters = 'A' # replace me
    else:
        # the easier case
        current_letter = auc[auc.index(last_letter) + 1]
        # Not quite done...
        new_letters = 'A'  # replace me
    return new_letters  # A string like 'A', or 'AAB'

