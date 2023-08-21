from __future__ import annotations

from ..base import Base, Property
from ..types.time import TimeRange
from ..types.track import Track
from ..types.detection import Detection
from ..types.hypothesis import Hypothesis
from node import _dict_set

from typing import Union, Tuple, List, TYPE_CHECKING
from datetime import datetime, timedelta
from queue import Queue

if TYPE_CHECKING:
    from .node import Node


class FusionQueue(Queue):
    """A queue from which fusion nodes draw data they have yet to fuse"""
    def __init__(self):
        super().__init__(maxsize=999999)

    def get_message(self):
        value = self.get()
        return value

    def set_message(self, value):
        self.put(value)


class DataPiece(Base):
    """A piece of data for use in an architecture. Sent via Messages, and stored in a Node's data_held"""
    node: "Node" = Property(
        doc="The Node this data piece belongs to")
    originator: "Node" = Property(
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


class Edge(Base):
    """Comprised of two connected Nodes"""
    nodes: Tuple[Node, Node] = Property(doc="A pair of nodes in the form (child, parent")
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

    def failed(self, current_time, duration):
        """Keeps track of when this edge was failed using the time_ranges_failed property. """
        end_time = current_time + timedelta(duration)
        self.time_ranges_failed.append(TimeRange(current_time, end_time))

    @property
    def child(self):
        return self.nodes[0]

    @property
    def parent(self):
        return self.nodes[1]

    @property
    def ovr_latency(self):
        """Overall latency including the two Nodes and the edge latency."""
        return self.child.latency + self.edge_latency + self.parent.latency

    @property
    def unsent_data(self):
        """Data held by the child that has not been sent to the parent."""
        unsent = []
        for status in ["fused", "created"]:
            for time_pertaining in self.child.data_held[status]:
                for data_piece in self.child.data_held[status][time_pertaining]:
                    if self.parent not in data_piece.sent_to:
                        unsent.append((data_piece, time_pertaining))
        return unsent


class Edges(Base):
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
        """Returns a list of tuples in the form (child, parent)"""
        edge_list = []
        for edge in self.edges:
            edge_list.append(edge.nodes)
        return edge_list

    def __len__(self):
        return len(self.edges)


class Message(Base):
    """A message, containing a piece of information, that gets propagated between two Nodes.
    Messages are opened by nodes that are a parent of the node that sent the message"""
    edge: Edge = Property(
        doc="The directed edge containing the sender and receiver of the message")
    time_pertaining: datetime = Property(
        doc="The latest time for which the data pertains. For a Detection, this would be the time "
            "of the Detection, or for a Track this is the time of the last State in the Track. "
            "Different from time_sent when data is passed on that was not generated by this "
            "Node's child")
    time_sent: datetime = Property(
        doc="Time at which the message was sent")
    data_piece: DataPiece = Property(
        doc="Info that the sent message contains")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.status = "sending"

    @property
    def generator_node(self):
        return self.edge.child

    @property
    def recipient_node(self):
        return self.edge.parent

    @property
    def arrival_time(self):
        # TODO: incorporate failed time ranges here. Not essential for a first PR. Could do with merging of PR #664
        return self.time_sent + timedelta(seconds=self.edge.ovr_latency)

    def update(self, current_time):
        progress = (current_time - self.time_sent).total_seconds()
        if progress < self.edge.child.latency:
            self.status = "sending"
        elif progress < self.edge.child.latency + self.edge.edge_latency:
            self.status = "transferring"
        elif progress < self.edge.ovr_latency:
            self.status = "receiving"
        else:
            self.status = "received"
