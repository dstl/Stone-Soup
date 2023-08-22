from abc import abstractmethod
from ..base import Base, Property
from .node import Node, SensorNode, RepeaterNode, FusionNode
from .edge import Edges, DataPiece
from ..types.groundtruth import GroundTruthPath
from ..types.detection import TrueDetection, Clutter

from typing import List, Collection, Tuple, Set, Union, Dict
import numpy as np
import networkx as nx
import graphviz
from string import ascii_uppercase as auc
from datetime import datetime, timedelta
import threading


class Architecture(Base):
    edges: Edges = Property(
        doc="An Edges object containing all edges. For A to be connected to B we would have an "
            "Edge with edge_pair=(A, B) in this object.")
    current_time: datetime = Property(
        default=datetime.now(),
        doc="The time which the instance is at for the purpose of simulation. "
            "This is increased by the propagate method. This should be set to the earliest "
            "timestep from the ground truth")
    name: str = Property(
        default=None,
        doc="A name for the architecture, to be used to name files and/or title plots. Default is "
            "the class name")
    force_connected: bool = Property(
        default=True,
        doc="If True, the undirected version of the graph must be connected, ie. all nodes should "
            "be connected via some path. Set this to False to allow an unconnected architecture. "
            "Default is True")

    # Below is no longer required with changes to plot - didn't delete in case we want to revert
    # to previous method
    # font_size: int = Property(
    #     default=8,
    #     doc='Font size for node labels')
    # node_dim: tuple = Property(
    #     default=(0.5, 0.5),
    #     doc='Height and width of nodes for graph icons, default is (0.5, 0.5)')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.name:
            self.name = type(self).__name__
        if not self.current_time:
            self.current_time = datetime.now()

        self.di_graph = nx.to_networkx_graph(self.edges.edge_list, create_using=nx.DiGraph)

        if self.force_connected and not self.is_connected and len(self) > 0:
            raise ValueError("The graph is not connected. Use force_connected=False, "
                             "if you wish to override this requirement")

        # Set attributes such as label, colour, shape, etc for each node
        last_letters = {'SensorNode': '', 'FusionNode': '', 'SensorFusionNode': '',
                        'RepeaterNode': ''}
        for node in self.di_graph.nodes:
            if node.label:
                label = node.label
            else:
                label, last_letters = _default_label(node, last_letters)
                node.label = label
            attr = {"label": f"{label}", "color": f"{node.colour}", "shape": f"{node.shape}",
                    "fontsize": f"{node.font_size}", "width": f"{node.node_dim[0]}",
                    "height": f"{node.node_dim[1]}", "fixedsize": True}
            self.di_graph.nodes[node].update(attr)

    def parents(self, node: Node):
        """Returns a set of all nodes to which the input node has a direct edge to"""
        if node not in self.all_nodes:
            raise ValueError("Node not in this architecture")
        parents = set()
        for other in self.all_nodes:
            if (node, other) in self.edges.edge_list:
                parents.add(other)
        return parents

    def children(self, node: Node):
        """Returns a set of all nodes to which the input node has a direct edge from"""
        if node not in self.all_nodes:
            raise ValueError("Node not in this architecture")
        children = set()
        for other in self.all_nodes:
            if (other, node) in self.edges.edge_list:
                children.add(other)
        return children

    def sibling_group(self, node: Node):
        """Returns a set of siblings of the given node. The given node is included in this set."""
        siblings = set()
        for parent in self.parents(node):
            for child in self.children(parent):
                siblings.add(child)
        return siblings

    @property
    def shortest_path_dict(self):
        g = nx.DiGraph()
        for edge in self.edges.edge_list:
            g.add_edge(edge[0], edge[1])
        path = nx.all_pairs_shortest_path_length(g)
        dpath = {x[0]: x[1] for x in path}
        return dpath

    def _parent_position(self, node: Node):
        """Returns a tuple of (x_coord, y_coord) giving the location of a node's parent"""
        parents = self.parents(node)
        if len(parents) == 1:
            parent = parents.pop()
        else:
            raise ValueError("Node has more than one parent")
        return parent.position

    @property
    def top_nodes(self):
        top_nodes = list()
        for node in self.all_nodes:
            if len(self.parents(node)) == 0:
                # This node must be the top level node
                top_nodes.append(node)

        return top_nodes

    def number_of_leaves(self, node: Node):
        node_leaves = set()
        non_leaves = 0
        for leaf_node in self.leaf_nodes:
            try:
                shortest_path = self.shortest_path_dict[leaf_node][node]
                if shortest_path != 0:
                    node_leaves.add(leaf_node)
            except KeyError:
                non_leaves += 1

        if len(node_leaves) == 0:
            return 1
        else:
            return len(node_leaves)

    @property
    def leaf_nodes(self):
        leaf_nodes = set()
        for node in self.all_nodes:
            if len(self.children(node)) == 0:
                # This must be a leaf node
                leaf_nodes.add(node)
        return leaf_nodes

    @abstractmethod
    def propagate(self, time_increment: float):
        raise NotImplementedError

    @property
    def all_nodes(self):
        return set(self.di_graph.nodes)

    @property
    def sensor_nodes(self):
        sensor_nodes = set()
        for node in self.all_nodes:
            if isinstance(node, SensorNode):
                sensor_nodes.add(node)
        return sensor_nodes

    @property
    def fusion_nodes(self):
        processing = set()
        for node in self.all_nodes:
            if isinstance(node, FusionNode):
                processing.add(node)
        return processing

    def plot(self, dir_path, filename=None, use_positions=False, plot_title=False,
             bgcolour="lightgray", node_style="filled", show_plot=True):
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
        elif self.is_hierarchical:

            # Find top node and assign location
            top_nodes = self.top_nodes
            if len(top_nodes) == 1:
                top_node = top_nodes[0]
            else:
                raise ValueError("Graph with more than one top level node provided.")

            top_node.position = (0, 0)
            attr = {"pos": f"{top_node.position[0]},{top_node.position[1]}!"}
            self.di_graph.nodes[top_node].update(attr)

            # Set of nodes that have been plotted / had their positions updated
            plotted_nodes = set()
            plotted_nodes.add(top_node)

            # Set of nodes that have been plotted, but need to have parent nodes plotted
            layer_nodes = set()
            layer_nodes.add(top_node)

            # Initialise a layer count
            layer = -1
            while len(plotted_nodes) < len(self.all_nodes):

                # Initialse an empty set to store nodes to be considered in the next iteration
                next_layer_nodes = set()

                # Iterate through nodes on the current layer (nodes that have been plotted but have
                # child nodes that are not plotted)
                for layer_node in layer_nodes:

                    # Find children of the parent node
                    children = self.children(layer_node)

                    # Find number of leaf nodes that descend from the parent
                    n_parent_leaves = self.number_of_leaves(layer_node)

                    # Get parent x_loc
                    parent_x_loc = layer_node.position[0]

                    # Get location of left limit of the range that leaf nodes will be plotted in
                    l_x_loc = parent_x_loc - n_parent_leaves/2
                    left_limit = l_x_loc

                    for child in children:
                        # Calculate x_loc of the child node
                        x_loc = left_limit + self.number_of_leaves(child)/2

                        # Update the left limit
                        left_limit += self.number_of_leaves(child)

                        # Update the position of the child node
                        child.position = (x_loc, layer)
                        attr = {"pos": f"{child.position[0]},{child.position[1]}!"}
                        self.di_graph.nodes[child].update(attr)

                        # Add child node to list of nodes to be considered in next iteration, and
                        # to list of nodes that have been plotted
                        next_layer_nodes.add(child)
                        plotted_nodes.add(child)

                # Set list of nodes to be considered next iteration
                layer_nodes = next_layer_nodes

                # Update layer count for correct y location
                layer -= 1

        dot = nx.drawing.nx_pydot.to_pydot(self.di_graph).to_string()
        dot_split = dot.split('\n')
        dot_split.insert(1, f"graph [bgcolor={bgcolour}]")
        dot_split.insert(1, f"node [style={node_style}]")
        dot = "\n".join(dot_split)
        if plot_title:
            if plot_title is True:
                plot_title = self.name
            elif not isinstance(plot_title, str):
                raise ValueError("Plot title must be a string, or True")
            dot = dot[:-2] + "labelloc=\"t\";\n" + f"label=\"{plot_title}\";" + "}"
        if not filename:
            filename = self.name
        viz_graph = graphviz.Source(dot, filename=filename, directory=dir_path, engine='neato')
        if show_plot:
            viz_graph.view()

    @property
    def density(self):
        """Returns the density of the graph, ie. the proportion of possible edges between nodes
        that exist in the graph"""
        num_nodes = len(self.all_nodes)
        num_edges = len(self.edges)
        architecture_density = num_edges / ((num_nodes * (num_nodes - 1)) / 2)
        return architecture_density

    @property
    def is_hierarchical(self):
        """Returns `True` if the :class:`Architecture` is hierarchical, otherwise `False`"""
        # if len(list(nx.simple_cycles(self.di_graph))) > 0 or not self.is_connected:
        no_parents = 0
        one_parent = 0
        multiple_parents = 0
        for node in self.all_nodes:
            if len(self.parents(node)) == 0:
                no_parents += 1
            elif len(self.parents(node)) == 1:
                one_parent += 1
            elif len(self.parents(node)) > 1:
                multiple_parents += 1

        if multiple_parents == 0 and no_parents == 1:
            return True
        else:
            return False

    @property
    def is_centralised(self):
        n_parents = 0
        for node in self.all_nodes:
            if len(node.children) == 0:
                n_parents += 1

        if n_parents == 0:
            return True
        else:
            return False

    @property
    def is_connected(self):
        return nx.is_connected(self.to_undirected)

    @property
    def to_undirected(self):
        return self.di_graph.to_undirected()

    def __len__(self):
        return len(self.di_graph)

    @property
    def fully_propagated(self):
        """Checks if all data for each node have been transferred
        to its parents. With zero latency, this should be the case after running propagate"""
        for edge in self.edges.edges:
            if len(edge.unsent_data) != 0:
                return False

        return True


class InformationArchitecture(Architecture):
    """The architecture for how information is shared through the network. Node A is "
    "connected to Node B if and only if the information A creates by processing and/or "
    "sensing is received and opened by B without modification by another node. """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for node in self.all_nodes:
            if isinstance(node, RepeaterNode):
                raise TypeError("Information architecture should not contain any repeater nodes")
        for fusion_node in self.fusion_nodes:
            fusion_node.tracker.set_time(self.current_time)

    def measure(self, ground_truths: List[GroundTruthPath], noise: Union[bool, np.ndarray] = True,
                **kwargs) -> Dict[SensorNode, Set[Union[TrueDetection, Clutter]]]:
        """ Similar to the method for :class:`~.SensorSuite`. Updates each node. """
        all_detections = dict()

        # Get rid of ground truths that have not yet happened
        # (ie GroundTruthState's with timestamp after self.current_time)
        new_ground_truths = set()
        for ground_truth_path in ground_truths:
            # need an if len(states) == 0 continue condition here?
            new_ground_truths.add(ground_truth_path.available_at_time(self.current_time))

        for sensor_node in self.sensor_nodes:
            all_detections[sensor_node] = set()
            for detection in sensor_node.sensor.measure(new_ground_truths, noise, **kwargs):
                all_detections[sensor_node].add(detection)

            # Borrowed below from SensorSuite. I don't think it's necessary, but might be something
            # we need. If so, will need to define self.attributes_inform

            # attributes_dict = \
            # {attribute_name: sensor_node.sensor.__getattribute__(attribute_name)
            #                    for attribute_name in self.attributes_inform}
            #
            # for detection in all_detections[sensor_node]:
            #     detection.metadata.update(attributes_dict)

            for data in all_detections[sensor_node]:
                # The sensor acquires its own data instantly
                sensor_node.update(data.timestamp, data.timestamp,
                                   DataPiece(sensor_node, sensor_node, data, data.timestamp),
                                   'created')

        return all_detections

    def propagate(self, time_increment: float, failed_edges: Collection = None):
        """Performs the propagation of the measurements through the network"""
        for edge in self.edges.edges:
            if failed_edges and edge in failed_edges:
                edge._failed(self.current_time, time_increment)
                continue  # No data passed along these edges
            edge.update_messages(self.current_time)
            # fuse goes here?
            for data_piece, time_pertaining in edge.unsent_data:
                edge.send_message(data_piece, time_pertaining, data_piece.time_arrived)

        # for node in self.processing_nodes:
        #     node.process() # This should happen when a new message is received
        count = 0
        if not self.fully_propagated:
            count += 1
            self.propagate(time_increment, failed_edges)
            return

        for fuse_node in self.fusion_nodes:
            x = threading.Thread(target=fuse_node.fuse)
            x.start()

        self.current_time += timedelta(seconds=time_increment)
        for fusion_node in self.fusion_nodes:
            fusion_node.tracker.set_time(self.current_time)


class NetworkArchitecture(Architecture):
    """The architecture for how data is propagated through the network. Node A is connected "
            "to Node B if and only if A sends its data through B. """

    def propagate(self, time_increment: float):
        # Still have to deal with latency/bandwidth
        self.current_time += timedelta(seconds=time_increment)
        for node in self.all_nodes:
            for parent in self.parents(node):
                for data in node.data_held:
                    parent.update(self.current_time, data)


class CombinedArchitecture(Base):
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
        self.information_architecture.propagate(time_increment, failed_edges)


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
