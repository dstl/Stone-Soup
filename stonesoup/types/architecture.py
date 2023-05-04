from abc import abstractmethod
from ..base import Property, Base
from .base import Type
from ..sensor.sensor import Sensor
from ..types.groundtruth import GroundTruthPath
from ..types.detection import TrueDetection, Clutter, Detection
from ..types.hypothesis import Hypothesis
from ..types.track import Track
from ..hypothesiser.base import Hypothesiser
from ..predictor.base import Predictor
from ..updater.base import Updater
from ..dataassociator.base import DataAssociator
from ..initiator.base import Initiator
from ..deleter.base import Deleter
from ..tracker.base import Tracker
from ..types.time import TimeRange

from typing import List, Collection, Tuple, Set, Union, Dict
import numpy as np
import networkx as nx
import graphviz
from string import ascii_uppercase as auc
from datetime import datetime, timedelta


class Node(Type):
    """Base node class. Should be abstract"""
    latency: float = Property(
        doc="Contribution to edge latency stemming from this node",
        default=0)
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_held = {"fused": {}, "created": {}, "unfused": {}}
        # Node no longer handles messages. All done by Edge

    def update(self, time_pertaining, time_arrived, data_piece, category, track=None):
        if not isinstance(time_pertaining, datetime) and isinstance(time_arrived, datetime):
            raise TypeError("Times must be datetime objects")
        if not track:
            if not isinstance(data_piece.data, Detection) and not isinstance(data_piece.data, Track):
                raise TypeError(f"Data provided without accompanying Track must be a Detection or a Track, not a "
                                f"{type(data_piece.data).__name__}")
            added, self.data_held[category] = _dict_set(self.data_held[category],
                                                             DataPiece(self, data_piece.originator, data_piece.data,
                                                                       time_arrived),
                                                             time_pertaining)
        else:
            if not isinstance(data_piece.data, Hypothesis):
                raise TypeError("Data provided with Track must be a Hypothesis")
            added, self.data_held[category] = _dict_set(self.data_held[category],
                                                             DataPiece(self, data_piece.originator, data_piece.data,
                                                                       time_arrived, track),
                                                             time_pertaining)

        return added


class SensorNode(Node):
    """A node corresponding to a Sensor. Fresh data is created here"""
    sensor: Sensor = Property(doc="Sensor corresponding to this node")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.colour:
            self.colour = '#1f77b4'
        if not self.shape:
            self.shape = 'square'


class FusionNode(Node):
    """A node that does not measure new data, but does process data it receives"""
    predictor: Predictor = Property(
        doc="The predictor used by this node. ")
    updater: Updater = Property(
        doc="The updater used by this node. ")
    hypothesiser: Hypothesiser = Property(
        doc="The hypothesiser used by this node. ")
    data_associator: DataAssociator = Property(
        doc="The data associator used by this node. ")
    initiator: Initiator = Property(
        doc="The initiator used by this node")
    deleter: Deleter = Property(
        doc="The deleter used by this node")
    tracker: Tracker = Property(
        doc="Tracker used by this node. Only trackers which can handle the types of data fused by "
            "the node will work.")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.colour:
            self.colour = '#006400'
        if not self.shape:
            self.shape = 'hexagon'
        self.tracks = set()  # Set of tracks this Node has recorded

    def process(self):
        # deleting this soon
        unprocessed_times = {time for time in self.unprocessed_data} | \
                {time for track in self.unprocessed_hypotheses
                 for time in self.unprocessed_hypotheses[track]}
        for time in unprocessed_times:
            try:
                detect_hypotheses = self.data_associator.associate(self.tracks,
                                                                   self.unprocessed_data[time],
                                                                   time)
            except KeyError:
                detect_hypotheses = False

            associated_detections = set()
            for track in self.tracks:
                if not detect_hypotheses and track not in self.unprocessed_hypotheses:
                    # If this track has no new data
                    continue
                detect_hypothesis = detect_hypotheses[track] if detect_hypotheses else False
                try:
                    # We deliberately re-use old hypotheses. If we just used the unprocessed ones
                    # then information would be lost
                    track_hypotheses = list(self.hypotheses_held[track][time])
                    if detect_hypothesis:
                        hypothesis = mean_combine([detect_hypothesis] + track_hypotheses, time)
                    else:
                        hypothesis = mean_combine(track_hypotheses, time)
                except (TypeError, KeyError):
                    hypothesis = detect_hypothesis

                _, self.hypotheses_held = _dict_set(self.hypotheses_held, hypothesis, track, time)
                _, self.processed_hypotheses = _dict_set(self.processed_hypotheses,
                                                         hypothesis, track, time)

                if hypothesis.measurement:
                    post = self.updater.update(hypothesis)
                    _update_track(track, post, time)
                    associated_detections.add(hypothesis.measurement)
                else:  # Keep prediction
                    _update_track(track, hypothesis.prediction, time)

                # Create or delete tracks
            self.tracks -= self.deleter.delete_tracks(self.tracks)
            if time in self.unprocessed_data: # If we had any detections
                self.tracks |= self.initiator.initiate(self.unprocessed_data[time] -
                                                       associated_detections, time)

        # Send the unprocessed data that was just processed to processed_data
        for time in self.data_held['unprocessed']:
            for data in self.data_held['unprocessed'][time]:
                _, self.data_held['processed'] = _dict_set(self.data_held['processed'], data, time)

        self.data_held['unprocessed'] = set()
        return


class SensorFusionNode(SensorNode, FusionNode):
    """A node that is both a sensor and also processes data"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.colour in ['#006400', '#1f77b4']:
            self.colour = '#909090' # attr dict in Architecture.__init__ also needs updating
        if self.shape in ['square', 'hexagon']:
            self.shape = 'octagon'


class RepeaterNode(Node):
    """A node which simply passes data along to others, without manipulating the data itself. """
    # Latency property could go here
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.colour:
            self.colour = '#ff7f0e'
        if not self.shape:
            self.shape = 'circle'


class DataPiece(Type):
    """A piece of data for use in an architecture. Sent via Messages, and stored in a Node's data_held"""
    node: Node = Property(
        doc="The Node this data piece belongs to")
    originator: Node = Property(
        doc="The node which first created this data, ie by sensing or fusing information together. "
            "If the data is simply passed along the chain, the originator remains unchanged. ")
    data: Union[Detection, Track, Hypothesis] = Property(
        doc="A Detection, Track, or Hypothesis")
    time_arrived: datetime = Property(
        doc="The time at which this piece of data was received by the Node, either by Message or by sensing.")
    track: Track = Property(
        doc="The Track in the event of data being a Hypothesis",
        default=None)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sent_to = set() # all Nodes the data_piece has been sent to, to avoid duplicates


class Edge(Type):
    """Comprised of two connected Nodes"""
    nodes: Tuple[Node, Node] = Property(doc="A pair of nodes in the form (ancestor, descendant")
    edge_latency: float = Property(doc="The latency stemming from the edge itself, "
                                       "and not either of the nodes",
                                   default=0)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.messages_held = {"pending": {},  # For pending, messages indexed by time sent.
                              "received": {}}  # For received, by time received
        self.time_ranges_failed = [] # List of time ranges during which this edge was failed

    def send_message(self, data_piece, time_pertaining, time_sent):
        if not isinstance(data_piece, DataPiece):
            raise TypeError("Message info must be one of the following types: "
                            "Detection, Hypothesis or Track")
        # Add message to 'pending' dict of edge
        message = Message(self, time_pertaining, time_sent, data_piece)
        _, self.messages_held = _dict_set(self.messages_held, message, 'pending', time_sent)
        # ensure message not re-sent
        data_piece.sent_to.add(self.nodes[1])

    def update_messages(self, current_time):
        # Check info type is what we expect
        to_remove = set()  # Needed as we can't change size of a set during iteration
        for time in self.messages_held['pending']:
            for message in self.messages_held['pending'][time]:
                message.update(current_time)
                if message.status == 'received':
                    # Then the latency has passed and message has been received
                    # Move message from pending to received messages in edge
                    to_remove.add((time, message))
                    _, self.messages_held = _dict_set(self.messages_held, message,
                                                      'received', message.arrival_time)
                    # Update
                    message.recipient_node.update(message.time_pertaining, message.arrival_time,
                                                  message.data_piece, "unfused")

        for time, message in to_remove:
            self.messages_held['pending'][time].remove(message)

    def _failed(self, current_time, duration):
        """Keeps track of when this edge was failed using the time_ranges_failed property. """
        end_time = current_time + timedelta(duration)
        self.time_ranges_failed.append(TimeRange(current_time, end_time))

    @property
    def ancestor(self):
        return self.nodes[0]

    @property
    def descendant(self):
        return self.nodes[1]

    @property
    def ovr_latency(self):
        """Overall latency including the two Nodes and the edge latency."""
        return self.ancestor.latency + self.edge_latency + self.descendant.latency

    @property
    def unsent_data(self):
        """Data held by the ancestor that has not been sent to the descendant."""
        unsent = []
        for status in ["fused", "created"]:
            for time_pertaining in self.ancestor.data_held[status]:
                for data_piece in self.ancestor.data_held[status][time_pertaining]:
                    if self.descendant not in data_piece.sent_to:
                        unsent.append((data_piece, time_pertaining))
        return unsent


class Edges(Type):
    """Container class for Edge"""
    edges: List[Edge] = Property(doc="List of Edge objects", default=None)

    def add(self, edge):
        self.edges.append(edge)

    def get(self, node_pair):
        if not isinstance(node_pair, Tuple) and all(isinstance(node, Node) for node in node_pair):
            raise TypeError("Must supply a tuple of nodes")
        if not len(node_pair) == 2:
            raise ValueError("Incorrect tuple length. Must be of length 2")
        for edge in self.edges:
            if edge.nodes == node_pair:
                # Assume this is the only match?
                return edge
        return None

    @property
    def edge_list(self):
        """Returns a list of tuples in the form (ancestor, descendant)"""
        edge_list = []
        for edge in self.edges:
            edge_list.append(edge.nodes)
        return edge_list

    def __len__(self):
        return len(self.edges)


class Message(Type):
    """A message, containing a piece of information, that gets propagated between two Nodes.
    Messages are opened by nodes that are a descendant of the node that sent the message"""
    edge: Edge = Property(
        doc="The directed edge containing the sender and receiver of the message")
    time_pertaining: datetime = Property(
        doc="The latest time for which the data pertains. For a Detection, this would be the time "
            "of the Detection, or for a Track this is the time of the last State in the Track. "
            "Different from time_sent when data is passed on that was not generated by this "
            "Node's ancestor")
    time_sent: datetime = Property(
        doc="Time at which the message was sent")
    data_piece: DataPiece = Property(
        doc="Info that the sent message contains")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.status = "sending"

    @property
    def generator_node(self):
        return self.edge.ancestor

    @property
    def recipient_node(self):
        return self.edge.descendant

    @property
    def arrival_time(self):
        # TODO: incorporate failed time ranges here. Not essential for a first PR. Could do with merging of PR #664
        return self.time_sent + timedelta(seconds=self.edge.ovr_latency)

    def update(self, current_time):
        progress = (current_time - self.time_sent).total_seconds()
        if progress < self.edge.ancestor.latency:
            self.status = "sending"
        elif progress < self.edge.ancestor.latency + self.edge.edge_latency:
            self.status = "transferring"
        elif progress < self.edge.ovr_latency:
            self.status = "receiving"
        else:
            self.status = "received"


class Architecture(Type):
    edges: Edges = Property(
        doc="An Edges object containing all edges. For A to be connected to B we would have an "
            "Edge with edge_pair=(A, B) in this object.")
    current_time: datetime = Property(
        doc="The time which the instance is at for the purpose of simulation. "
            "This is increased by the propagate method. This should be set to the earliest timestep "
            "from the ground truth")
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
        doc='Height and width of nodes for graph icons, default is (0.5, 0.5)')

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
                    "fontsize": f"{self.font_size}", "width": f"{self.node_dim[0]}",
                    "height": f"{self.node_dim[1]}", "fixedsize": True}
            self.di_graph.nodes[node].update(attr)

    def descendants(self, node: Node):
        """Returns a set of all nodes to which the input node has a direct edge to"""
        if node not in self.all_nodes:
            raise ValueError("Node not in this architecture")
        descendants = set()
        for other in self.all_nodes:
            if (node, other) in self.edges.edge_list:
                descendants.add(other)
        return descendants

    def ancestors(self, node: Node):
        """Returns a set of all nodes to which the input node has a direct edge from"""
        if node not in self.all_nodes:
            raise ValueError("Node not in this architecture")
        ancestors = set()
        for other in self.all_nodes:
            if (other, node) in self.edges.edge_list:
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
        sensor_nodes = set()
        for node in self.all_nodes:
            if isinstance(node, SensorNode):
                sensor_nodes.add(node)
        return sensor_nodes

    @property
    def processing_nodes(self):
        processing = set()
        for node in self.all_nodes:
            if isinstance(node, FusionNode):
                processing.add(node)
        return processing

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
        if plot_title:
            if plot_title is True:
                plot_title = self.name
            elif not isinstance(plot_title, str):
                raise ValueError("Plot title must be a string, or True")
            dot = dot[:-2] + "labelloc=\"t\";\n" + f"label=\"{plot_title}\";" + "}"
        if not filename:
            filename = self.name
        viz_graph = graphviz.Source(dot, filename=filename, directory=dir_path, engine='neato')
        viz_graph.view()

    @property
    def density(self):
        """Returns the density of the graph, ie. the proportion of possible edges between nodes
        that exist in the graph"""
        num_nodes = len(self.all_nodes)
        num_edges = len(self.edges)
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

    @property
    def fully_propagated(self):
        """Checks if all data for each node have been transferred
        to its descendants. With zero latency, this should be the case after running propagate"""
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
                sensor_node.update(data.timestamp, data.timestamp, DataPiece(sensor_node, sensor_node, data,
                                                                             data.timestamp), "created")

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
        self.current_time += timedelta(seconds=time_increment)


class NetworkArchitecture(Architecture):
    """The architecture for how data is propagated through the network. Node A is connected "
            "to Node B if and only if A sends its data through B. """
    def propagate(self, time_increment: float):
        # Still have to deal with latency/bandwidth
        self.current_time += timedelta(seconds=time_increment)
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


def mean_combine(objects: List, current_time: datetime = None):
    """Combine a list of objects of the same type by averaging all numbers"""
    this_type = type(objects[0])
    if any(type(obj) is not this_type for obj in objects):
        raise TypeError("Objects must be of identical type")

    new_values = dict()
    for name in type(objects[0]).properties:
        value = getattr(objects[0], name)
        if isinstance(value, (int, float, complex)) and not isinstance(value, bool):
            # average them all
            new_values[name] = np.mean([getattr(obj, name) for obj in objects])
        elif isinstance(value, datetime):
            # Take the input time, as we likely want the current simulation time
            new_values[name] = current_time
        elif isinstance(value, Base):  # if it's a Stone Soup object
            # recurse
            new_values[name] = mean_combine([getattr(obj, name) for obj in objects])
        else:
            # just take 1st value
            new_values[name] = getattr(objects[0], name)

        return this_type.__init__(**new_values)


def _dict_set(my_dict, value, key1, key2=None):
    """Utility function to add value to my_dict at the specified key(s)
    Returns True iff the set increased in size, ie the value was new to its position"""
    if not my_dict:
        if key2:
            my_dict = {key1: {key2: {value}}}
        else:
            my_dict = {key1: {value}}
    elif key2:
        if key1 in my_dict:
            if key2 in my_dict[key1]:
                old_len = len(my_dict[key1][key2])
                my_dict[key1][key2].add(value)
                return len(my_dict[key1][key2]) == old_len + 1, my_dict
            else:
                my_dict[key1][key2] = {value}
        else:
            my_dict[key1] = {key2: {value}}
    else:
        if key1 in my_dict:
            old_len = len(my_dict[key1])
            my_dict[key1].add(value)
            return len(my_dict[key1]) == old_len + 1, my_dict
        else:
            my_dict[key1] = {value}
    return True, my_dict


def _update_track(track, state, time):
    for state_num in range(len(track)):
        if time == track[state_num].timestamp:
            track[state_num] = state
            return
    track.append(state)

