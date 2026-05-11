from abc import abstractmethod
from collections.abc import Collection, Sequence
from datetime import datetime, timedelta
from operator import attrgetter
from typing import Union

import graphviz
import numpy as np
import networkx as nx
import pydot
from ordered_set import OrderedSet

from .edge import Edges, DataPiece, Edge
from .node import Node, SensorNode, RepeaterNode, FusionNode
from ._functions import _default_label_gen
from ..base import Base, Property
from ..types.detection import TrueDetection, Clutter
from ..types.groundtruth import GroundTruthPath


class Architecture(Base):
    """Abstract Architecture Base class. Subclasses must implement the
    :meth:`~Architecture.propogate` method.
    """

    edges: Edges = Property(
        doc="An Edges object containing all edges. For A to be connected to B we would have an "
            "Edge with edge_pair=(A, B) in this object.")
    current_time: datetime = Property(
        default_factory=datetime.now,
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
    use_arrival_time: bool = Property(
        default=False,
        doc="If True, the timestamp on data passed around the network will not be assigned when "
            "it is opened by the fusing node - simulating an architecture where time of recording "
            "is not registered by the sensor nodes"
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.name:
            self.name = type(self).__name__

        self.di_graph = nx.to_networkx_graph(self.edges.edge_list, create_using=nx.DiGraph)
        self._viz_graph = None

        if self.force_connected and not self.is_connected and len(self) > 0:
            raise ValueError("The graph is not connected. Use force_connected=False, "
                             "if you wish to override this requirement")

        node_label_gens = {}
        labels = {node.label.replace("\n", " ") for node in self.di_graph.nodes if node.label}
        for node in self.di_graph.nodes:
            if not node.label:
                label_gen = node_label_gens.setdefault(type(node), _default_label_gen(type(node)))
                while not node.label or node.label.replace("\n", " ") in labels:
                    node.label = next(label_gen)
            self.di_graph.nodes[node].update(self._node_kwargs(node))

    def recipients(self, node: Node):
        """
        Returns a set of all nodes to which the input node has a direct edge to.

        Parameters
        ----------
        node : Node
            Node of which to return the recipients of.

        Returns
        -------
        set
            Set of nodes that are recipients of the given node.

        Raises
        ------
        ValueError
            If the given node is not in the Architecture.
        """
        if node not in self.all_nodes:
            raise ValueError("Node not in this architecture")
        recipients = set()
        for other in self.all_nodes:
            if (node, other) in self.edges.edge_list:
                recipients.add(other)
        return recipients

    def senders(self, node: Node):
        """
        Returns a set of all nodes from which the input node has a direct edge.

        Parameters
        ----------
        node : Node
            Node of which to return the senders of.

        Returns
        -------
        set
            Set of nodes that are senders to the given node.

        Raises
        ------
        ValueError
            If the given node is not in the Architecture.
        """
        if node not in self.all_nodes:
            raise ValueError("Node not in this architecture")
        senders = set()
        for other in self.all_nodes:
            if (other, node) in self.edges.edge_list:
                senders.add(other)
        return senders

    @property
    def shortest_path_dict(self):
        """
        Returns a dictionary where dict[key1][key2] gives the distance of the shortest path
        from node1 to node2 if key1=node1 and key2=node2. If no path exists from node1 to node2,
        a KeyError is raised.

        Returns
        -------
        dict
            Nested dictionary where dict[node1][node2] gives the distance of the shortest
            path from node1 to node2.
        """
        # Cannot use self.di_graph as it is not adjusted when edges are removed after
        # instantiation of architecture.
        g = nx.DiGraph()
        for edge in self.edges.edge_list:
            g.add_edge(edge[0], edge[1])
        path = nx.all_pairs_shortest_path_length(g)
        dpath = {x[0]: x[1] for x in path}
        return dpath

    @property
    def top_level_nodes(self):
        """
        Returns a set of 'top level nodes' - nodes with no recipients (e.g., the single node
        at the top of a hierarchical architecture).

        Returns
        -------
        set
            Set of nodes that have no recipients.
        """
        top_nodes = set()
        for node in self.all_nodes:
            if len(self.recipients(node)) == 0:
                top_nodes.add(node)

        return top_nodes

    def number_of_leaves(self, node: Node):
        """
        Returns the number of leaf nodes which are connected to the given node by a path from
        the leaf node to the parameter node.

        Parameters
        ----------
        node : Node
            Node of which to calculate number of leaf nodes.

        Returns
        -------
        int
            Number of leaf nodes that are connected to a given node.
        """
        node_leaves = set()
        for leaf_node in self.leaf_nodes:
            if node == leaf_node:
                return 1
            try:
                self.shortest_path_dict[leaf_node][node]
            except KeyError:
                continue
            node_leaves.add(leaf_node)
        return len(node_leaves)

    @property
    def leaf_nodes(self):
        """
        Returns all the nodes in the Architecture which have no sender nodes (i.e., all nodes
        that do not receive any data from other nodes).

        Returns
        -------
        set
            Set of all leaf nodes that exist in the Architecture.
        """
        leaf_nodes = set()
        for node in self.all_nodes:
            if len(self.senders(node)) == 0:
                # This must be a leaf node
                leaf_nodes.add(node)
        return leaf_nodes

    @abstractmethod
    def propagate(self, time_increment: float):
        raise NotImplementedError

    @property
    def all_nodes(self):
        """
        Returns a set of all Nodes in the Architecture.

        Returns
        -------
        set
            Set of all nodes in the Architecture.
        """
        return set(self.di_graph.nodes)

    @property
    def sensor_nodes(self):
        """
        Returns a set of all SensorNodes in the Architecture.

        Returns
        -------
        set
            Set of nodes in the Architecture that have a Sensor.
        """
        sensor_nodes = set()
        for node in self.all_nodes:
            if isinstance(node, SensorNode):
                sensor_nodes.add(node)
        return sensor_nodes

    @property
    def fusion_nodes(self):
        """
        Returns a set of all FusionNodes in the Architecture.

        Returns
        -------
        set
            Set of nodes in the Architecture that perform data fusion.
        """
        fusion = set()
        for node in self.all_nodes:
            if isinstance(node, FusionNode):
                fusion.add(node)
        return fusion

    @property
    def repeater_nodes(self):
        """
        Returns a set of all RepeaterNodes in the Architecture.

        Returns
        -------
        set
            Set of nodes in the Architecture whose only role is to link two other nodes together.
        """

        repeater_nodes = set()
        for node in self.all_nodes:
            if isinstance(node, RepeaterNode):
                repeater_nodes.add(node)
        return repeater_nodes

    @staticmethod
    def _node_kwargs(node, use_position=True):
        node_kwargs = {
            'label': node.label,
            'shape': node.shape,
            'color': node.colour,
        }
        if node.font_size:
            node_kwargs['fontsize'] = node.font_size
        if node.node_dim:
            node_kwargs['width'] = node.node_dim[0]
            node_kwargs['height'] = node.node_dim[1]
        if use_position and node.position:
            if not isinstance(node.position, Sequence):
                raise TypeError("Node position, must be Sequence of length 2")
            node_kwargs["pos"] = f"{node.position[0]},{node.position[1]}!"
        return node_kwargs

    def plot(self, use_positions=False, plot_title=False,
             bgcolour="transparent", node_style="filled", font_name='helvetica', plot_style=None):
        """
        Creates a pdf plot of the directed graph and displays it.

        Parameters
        ----------
        use_positions : bool, optional
            Whether to use node positions (default is False).
        plot_title : str or bool, optional
            If a string is supplied, makes this the title of the plot. If True, uses
            the name attribute of the graph to title the plot. If False, no title is used.
            Default is False.
        bgcolour : str, optional
            Background colour for the plot. Default is "transparent".
        node_style : str, optional
            Node style for the plot. Default is "filled".
        font_name : str, optional
            Font name for node labels. Default is 'helvetica'.
        plot_style : str, optional
            Style to be used to plot the graph. Currently only option is 'hierarchical'.

        Returns
        -------
        graphviz.Source
            The graphviz Source object for the plot.
        """
        is_hierarchical = self.is_hierarchical or plot_style == 'hierarchical'
        if is_hierarchical:
            # Find top node and assign location
            top_nodes = self.top_level_nodes
            if len(top_nodes) == 1:
                top_node = top_nodes.pop()
            else:
                raise ValueError("Graph with more than one top level node provided.")

            # Initialise a layer count
            node_layers = [[top_node]]
            processed_nodes = {top_node}
            while self.all_nodes - processed_nodes:
                senders = [
                    sender
                    for node in node_layers[-1]
                    for sender in sorted(self.senders(node), key=attrgetter('label'))]
                if not senders:
                    break
                else:
                    node_layers.append(senders)
                    processed_nodes.update(senders)

        strict = nx.number_of_selfloops(self.di_graph) == 0 and not self.di_graph.is_multigraph()
        graph = pydot.Dot(graph_name='', strict=strict, graph_type='digraph', rankdir='BT')
        if isinstance(plot_title, str):
            graph.set_graph_defaults(label=plot_title, labelloc='t')
        elif isinstance(plot_title, bool) and plot_title:
            graph.set_graph_defaults(label=self.name, labelloc='t')
        elif not isinstance(plot_title, bool):
            raise ValueError("Plot title must be a string or bool")
        graph.set_graph_defaults(bgcolor=bgcolour)
        graph.set_node_defaults(fontname=font_name, style=node_style)

        if is_hierarchical:
            for n, layer_nodes in enumerate(node_layers):
                subgraph = pydot.Subgraph(rank='max' if n == 0 else 'same')
                for node in layer_nodes:
                    new_node = pydot.Node(
                        node.label.replace("\n", " "), **self._node_kwargs(node, use_positions))
                    subgraph.add_node(new_node)
                graph.add_subgraph(subgraph)
        else:
            graph.set_overlap('false')
            for node in self.all_nodes:
                new_node = pydot.Node(
                    node.label.replace("\n", " "), **self._node_kwargs(node, use_positions))
                graph.add_node(new_node)

        for edge in self.edges.edge_list:
            new_edge = pydot.Edge(
                edge[0].label.replace("\n", " "), edge[1].label.replace("\n", " "))
            graph.add_edge(new_edge)

        viz_graph = graphviz.Source(
            graph.to_string(), engine='dot' if is_hierarchical else 'neato')
        self._viz_graph = viz_graph
        return viz_graph

    def _repr_html_(self):
        if getattr(self, '_viz_graph', None) is None:
            self.plot()
        return self._viz_graph._repr_image_svg_xml()

    @property
    def density(self):
        """
        Returns the density of the graph, i.e. the proportion of possible edges between nodes
        that exist in the graph.

        Returns
        -------
        float
            Density of the architecture.
        """
        num_nodes = len(self.all_nodes)
        num_edges = len(self.edges)
        architecture_density = num_edges / ((num_nodes * (num_nodes - 1)) / 2)
        return architecture_density

    @property
    def is_hierarchical(self):
        """
        Returns True if the Architecture is hierarchical, otherwise False.
        An architecture is hierarchical if and only if there exists only one node with 0
        recipients and all other nodes have exactly 1 recipient.

        Returns
        -------
        bool
            Whether the architecture is hierarchical.
        """
        top_nodes = self.top_level_nodes
        if len(self.top_level_nodes) != 1:
            return False
        for node in self.all_nodes:
            if node not in top_nodes and len(self.recipients(node)) != 1:
                return False
        return True

    @property
    def is_centralised(self):
        """
        Returns True if the Architecture is centralised, otherwise False.
        An architecture is centralised if and only if there exists only one node with 0
        recipients, and there exists a path to this node from every other node in the
        architecture.

        Returns
        -------
        bool
            Whether the architecture is centralised.
        """
        top_nodes = self.top_level_nodes
        if len(top_nodes) != 1:
            return False
        top_node = top_nodes.pop()
        for node in self.all_nodes - {top_node}:
            try:
                _ = self.shortest_path_dict[node][top_node]
            except KeyError:
                return False
        return True

    @property
    def is_connected(self):
        """
        Returns True if the graph is connected, otherwise False.

        Returns
        -------
        bool
            Whether the graph is connected.
        """
        return nx.is_connected(self.to_undirected)

    @property
    def to_undirected(self):
        """
        Returns an undirected version of self.digraph.

        Returns
        -------
        networkx.Graph
            Undirected version of the directed graph.
        """
        return self.di_graph.to_undirected()

    def __len__(self):
        return len(self.di_graph)

    @property
    def fully_propagated(self):
        """
        Checks if all data for each node have begun transfer to its recipients.
        With zero latency, this should be the case after running propagate.

        Returns
        -------
        bool
            Whether all data have begun transfer to recipients.
        """
        for edge in self.edges.edges:
            if len(edge.unsent_data) != 0:
                return False
            elif len(edge.unpassed_data) != 0:
                return False

        return True

    def measure(self, ground_truths: list[GroundTruthPath], noise: Union[bool, np.ndarray] = True,
                **kwargs) -> dict[SensorNode, set[Union[TrueDetection, Clutter]]]:
        """
        Similar to the method for :class:`~.SensorSuite`. Updates each node.

        Parameters
        ----------
        ground_truths : list of GroundTruthPath
            List of ground truth paths.
        noise : bool or np.ndarray, optional
            Whether to add noise or a noise array (default is True).
        **kwargs
            Additional keyword arguments passed to the sensor measure method.

        Returns
        -------
        dict
            Dictionary mapping SensorNode to set of TrueDetection or Clutter.
        """
        all_detections = dict()

        # Filter out only the ground truths that have already happened at self.current_time
        current_ground_truths = OrderedSet()
        for ground_truth_path in ground_truths:
            available_gtp = GroundTruthPath(ground_truth_path[:self.current_time +
                                                              timedelta(microseconds=1)])
            if len(available_gtp) > 0:
                current_ground_truths.add(available_gtp)

        for sensor_node in self.sensor_nodes:
            all_detections[sensor_node] = set()
            for detection in sensor_node.sensor.measure(current_ground_truths, noise, **kwargs):
                all_detections[sensor_node].add(detection)

            for data in all_detections[sensor_node]:
                # The sensor acquires its own data instantly
                sensor_node.update(data.timestamp, data.timestamp,
                                   DataPiece(sensor_node, sensor_node, data, data.timestamp),
                                   'created')

        return all_detections


class NonPropagatingArchitecture(Architecture):
    """
    A simple Architecture class that does not simulate propagation of any data. Can be used for
    performing network operations on an :class:`~.Edges` object.
    """
    def propagate(self, time_increment: float):
        """
        Does not simulate propagation of any data.

        Parameters
        ----------
        time_increment : float
            Time increment for propagation (unused).
        """
        pass


class InformationArchitecture(Architecture):
    """The architecture for how information is shared through the network. Node A is
    connected to Node B if and only if the information A creates by processing and/or
    sensing is received and opened by B without modification by another node.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if len(self.repeater_nodes) != 0:
            raise TypeError("Information architecture should not contain any repeater "
                            "nodes")

    def propagate(self, time_increment: float, failed_edges: Collection = None):
        """
        Performs the propagation of the measurements through the network.

        Parameters
        ----------
        time_increment : float
            Time increment for propagation.
        failed_edges : Collection, optional
            Collection of failed edges (default is None).
        """
        # Update each edge with messages received/sent
        for edge in self.edges.edges:
            # TODO: Future work - Introduce failed edges functionality

            # Initial update of message categories
            edge.update_messages(self.current_time, use_arrival_time=self.use_arrival_time)
            for data_piece, time_pertaining in edge.unsent_data:
                edge.send_message(data_piece, time_pertaining, data_piece.time_arrived)

            # Need to re-run update messages so that messages aren't left as 'pending'
            edge.update_messages(self.current_time, use_arrival_time=self.use_arrival_time)

        for fuse_node in self.fusion_nodes:
            fuse_node.fuse()

        if self.fully_propagated:
            self.current_time += timedelta(seconds=time_increment)
            return
        else:
            self.propagate(time_increment, failed_edges)


class NetworkArchitecture(Architecture):
    """The architecture for how data is propagated through the network. Node A is connected
    to Node B if and only if A sends its data through B. """
    information_arch: InformationArchitecture = Property(default=None)
    information_architecture_edges: Edges = Property(default=None)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Check whether an InformationArchitecture is provided, if not, see if one can be created
        if self.information_arch is None:

            # If info edges are provided, we can deduce an information architecture, otherwise:
            if self.information_architecture_edges is None:

                # If repeater nodes are present in the Network architecture, we can deduce an
                # Information architecture
                if len(self.repeater_nodes) > 0:
                    self.information_architecture_edges = Edges(inherit_edges(Edges(self.edges)))
                    self.information_arch = InformationArchitecture(
                        edges=self.information_architecture_edges, current_time=self.current_time)
                else:
                    self.information_architecture_edges = self.edges
                    self.information_arch = InformationArchitecture(self.edges, self.current_time)
            else:
                self.information_arch = InformationArchitecture(
                    edges=self.information_architecture_edges, current_time=self.current_time)

        # Need to reset digraph for info-arch
        self.di_graph = nx.to_networkx_graph(self.edges.edge_list, create_using=nx.DiGraph)
        # Set attributes such as label, colour, shape, etc. for each node
        for node in self.di_graph.nodes:
            self.di_graph.nodes[node].update(self._node_kwargs(node))

    def propagate(self, time_increment: float, failed_edges: Collection = None):
        """
        Performs the propagation of the measurements through the network.

        Parameters
        ----------
        time_increment : float
            Time increment for propagation.
        failed_edges : Collection, optional
            Collection of failed edges (default is None).
        """
        # Update each edge with messages received/sent
        for edge in self.edges.edges:
            # TODO: Future work - Introduce failed edges functionality

            to_network_node = edge.recipient not in self.information_arch.all_nodes

            # Initial update of message categories
            edge.update_messages(
                self.current_time,
                to_network_node=to_network_node,
                use_arrival_time=self.use_arrival_time)

            # Send available messages from nodes to the edges
            if edge.sender in self.information_arch.all_nodes:
                for data_piece, time_pertaining in edge.unsent_data:
                    edge.send_message(data_piece, time_pertaining, data_piece.time_arrived)
            else:
                for message in edge.sender.messages_to_pass_on:
                    if edge.recipient not in message.data_piece.sent_to:
                        edge.pass_message(message)

            # Need to re-run update messages so that messages aren't left as 'pending'
            edge.update_messages(
                self.current_time,
                to_network_node=to_network_node,
                use_arrival_time=self.use_arrival_time)

        for fuse_node in self.fusion_nodes:
            fuse_node.fuse()

        if self.fully_propagated:
            self.current_time += timedelta(seconds=time_increment)
            return
        else:
            self.propagate(time_increment, failed_edges)


def inherit_edges(network_architecture):
    """
    Utility function that takes a NetworkArchitecture object and infers what the overlaying
    InformationArchitecture graph would be.

    Parameters
    ----------
    network_architecture : NetworkArchitecture
        A NetworkArchitecture object.

    Returns
    -------
    Edges
        A list of edges for the InformationArchitecture.
    """

    edges = list()
    for edge in network_architecture.edges:
        edges.append(edge)
    temp_arch = NonPropagatingArchitecture(edges=Edges(edges))

    # Iterate through repeater nodes in the Network Architecture to find edges to remove
    for repeaternode in temp_arch.repeater_nodes:
        to_replace = list()
        to_add = list()

        senders = temp_arch.senders(repeaternode)
        recipients = temp_arch.recipients(repeaternode)

        # Find all edges that pass data to the repeater node
        for sender in senders:
            edges = temp_arch.edges.get((sender, repeaternode))
            to_replace += edges

        # Find all edges that pass data from the repeater node
        for recipient in recipients:
            edges = temp_arch.edges.get((repeaternode, recipient))
            to_replace += edges

        # Create a new edge from every sender to every recipient
        for sender in senders:
            for recipient in recipients:

                # Could be possible edges from sender to node, choose path of minimum latency
                poss_edges_to = temp_arch.edges.get((sender, repeaternode))
                latency_to = np.inf
                for edge in poss_edges_to:
                    latency_to = edge.edge_latency if edge.edge_latency <= latency_to else \
                        latency_to

                # Could be possible edges from node to recipient, choose path of minimum latency
                poss_edges_from = temp_arch.edges.get((sender, repeaternode))
                latency_from = np.inf
                for edge in poss_edges_from:
                    latency_from = edge.edge_latency if edge.edge_latency <= latency_from else \
                        latency_from

                latency = latency_to + latency_from + repeaternode.latency
                edge = Edge(nodes=(sender, recipient), edge_latency=latency)
                to_add.append(edge)

        for edge in to_replace:
            temp_arch.edges.remove(edge)
        for edge in to_add:
            temp_arch.edges.add(edge)
    return temp_arch.edges
