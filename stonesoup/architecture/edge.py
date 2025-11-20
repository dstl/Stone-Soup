import copy
from collections.abc import Collection, Sequence
from datetime import datetime, timedelta
from numbers import Number
from queue import Queue
from typing import Union, TYPE_CHECKING

from ..base import Base, Property
from ..types.time import TimeRange, CompoundTimeRange
from ..types.track import Track
from ..types.detection import Detection
from ..types.hypothesis import Hypothesis
from ._functions import _dict_set

if TYPE_CHECKING:
    from .node import Node


class FusionQueue(Queue):
    """A queue from which fusion nodes draw data they have yet to fuse

    Iterable, where it blocks attempting to yield items on the queue
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._to_consume = 0
        self._consuming = False
        self.received = set()

    def _put(self, *args, **kwargs):
        super()._put(*args, **kwargs)
        self._to_consume += 1

    def __iter__(self):
        self._consuming = True
        while True:
            yield super().get()
            self._to_consume -= 1

    @property
    def waiting_for_data(self):
        """
        Returns True if the queue is consuming and waiting for data.

        Returns
        -------
        bool
            Whether the queue is waiting for data.
        """
        return self._consuming and not self._to_consume

    def get(self, *args, **kwargs):
        raise NotImplementedError("Getting items from queue must use iteration")


class DataPiece(Base):
    """A piece of data for use in an architecture. Sent via a :class:`~.Message`,
    and stored in a Node's :attr:`data_held`"""
    node: 'Node' = Property(
        doc="The Node this data piece belongs to")
    originator: 'Node' = Property(
        doc="The node which first created this data, ie by sensing or fusing information together."
            " If the data is simply passed along the chain, the originator remains unchanged. ")
    data: Union[Detection, Track, Hypothesis] = Property(
        doc="A Detection, Track, or Hypothesis")
    time_arrived: datetime = Property(
        doc="The time at which this piece of data was received by the Node, either by Message or "
            "by sensing.")
    track: Track = Property(
        doc="The Track in the event of data being a Hypothesis",
        default=None)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sent_to = set()  # all Nodes the data_piece has been sent to, to avoid duplicates


class Edge(Base):
    """Comprised of two connected :class:`~.Node` instances"""
    nodes: tuple["Node", "Node"] = Property(doc="A pair of nodes in the form (sender, recipient)")
    edge_latency: float = Property(doc="The latency stemming from the edge itself, "
                                       "and not either of the nodes",
                                   default=0.0)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.edge_latency, Number):
            raise TypeError(f"edge_latency should be a float, not a {type(self.edge_latency)}")
        self.messages_held = {"pending": {},  # For pending, messages indexed by time sent.
                              "received": {}}  # For received, by time received
        self.time_range_failed = CompoundTimeRange()  # Times during which this edge was failed
        self.nodes = tuple(self.nodes)

    def send_message(self, data_piece, time_pertaining, time_sent):
        """
        Takes a piece of data retrieved from the edge's sender node, and propagates it
        along the edge.

        Parameters
        ----------
        data_piece : DataPiece
            DataPiece object pulled from the edge's sender.
        time_pertaining : datetime
            The latest time for which the data pertains. For a Detection, this would be the
            time of the Detection, or for a Track this is the time of the last State in the Track.
        time_sent : datetime
            Time at which the message was sent.
        """
        if not isinstance(data_piece, DataPiece):
            raise TypeError(f"data_piece is type {type(data_piece)}. Expected DataPiece")
        message = Message(edge=self, time_pertaining=time_pertaining, time_sent=time_sent,
                          data_piece=data_piece, destinations={self.recipient})
        _, self.messages_held = _dict_set(self.messages_held, message, 'pending', time_sent)
        # ensure message not re-sent
        data_piece.sent_to.add(self)

    def pass_message(self, message):
        """
        Takes a message from a Node's 'messages_to_pass_on' store and propagates it to the
        relevant edges.

        Parameters
        ----------
        message : Message
            Message to propagate.
        """
        message_copy = copy.copy(message)
        message_copy.edge = self
        if message_copy.destinations == {self.sender} or message.destinations is None:
            message_copy.destinations = {self.recipient}
        _, self.messages_held = _dict_set(self.messages_held, message_copy, 'pending',
                                          message_copy.time_sent)
        # Message not opened by repeater node, remove node from 'sent_to'
        message_copy.data_piece.sent_to.add(self)

    def update_messages(self, current_time, to_network_node=False, use_arrival_time=False):
        """
        Updates the category of messages stored in edge.messages_held if latency time has passed.
        Adds messages that have 'arrived' at recipient to the relevant holding area of the node.

        Parameters
        ----------
        current_time : datetime
            Current time in simulation.
        to_network_node : bool, optional
            True if recipient node is not in the information architecture (default is False).
        use_arrival_time : bool, optional
            True if arriving data should use arrival time as its timestamp (default is False).
        """
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

                    # Assign destination as recipient of edge if no destination provided
                    if message.destinations is None:
                        message.destinations = {self.recipient}

                    # Update node according to inclusion in Information Architecture
                    if not to_network_node and message.destinations == {self.recipient}:
                        # Add data to recipient's data_held
                        self.recipient.update(message.time_pertaining,
                                              message.arrival_time,
                                              message.data_piece, "unfused",
                                              use_arrival_time=use_arrival_time)

                    elif not to_network_node and self.recipient in message.destinations:
                        # Add data to recipient's data held, and message to messages_to_pass_on
                        self.recipient.update(message.time_pertaining,
                                              message.arrival_time,
                                              message.data_piece, "unfused",
                                              use_arrival_time=use_arrival_time)
                        message.destinations = None
                        self.recipient.messages_to_pass_on.append(message)

                    elif to_network_node or self.recipient not in message.destinations:
                        # Add message to recipient's messages_to_pass_on
                        message.destinations = None
                        self.recipient.messages_to_pass_on.append(message)

        for time, message in to_remove:
            self.messages_held['pending'][time].remove(message)
            if len(self.messages_held['pending'][time]) == 0:
                del self.messages_held['pending'][time]

    def failed(self, current_time, delta):
        """
        Keeps track of when this edge was failed using the time_ranges_failed property.

        Parameters
        ----------
        current_time : datetime
            The current time.
        delta : timedelta
            The duration for which the edge is failed.
        """
        end_time = current_time + delta
        self.time_range_failed.add(TimeRange(current_time, end_time))

    @property
    def sender(self):
        return self.nodes[0]

    @property
    def recipient(self):
        return self.nodes[1]

    @property
    def ovr_latency(self):
        """Overall latency of this :class:`~.Edge`"""
        return self.sender.latency + self.edge_latency

    @property
    def unpassed_data(self):
        unpassed = []
        for message in self.sender.messages_to_pass_on:
            if self not in message.data_piece.sent_to:
                unpassed.append(message)
        return unpassed

    @property
    def unsent_data(self):
        """Data modified by the sender that has not been sent to the
        recipient."""
        unsent = []
        if isinstance(type(self.sender.data_held), type(None)) or self.sender.data_held is None:
            return unsent
        else:
            for status in ["fused", "created"]:
                for time_pertaining in self.sender.data_held[status]:
                    for data_piece in self.sender.data_held[status][time_pertaining]:
                        # Data will be sent to any nodes it hasn't been sent to before
                        if self not in data_piece.sent_to:
                            unsent.append((data_piece, time_pertaining))
            return unsent

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        return all(getattr(self, name) == getattr(other, name) for name in type(self).properties)

    def __hash__(self):
        return hash(tuple(getattr(self, name) for name in type(self).properties))


class Edges(Base, Collection):
    """Container class for :class:`~.Edge`"""
    edges: list[Edge] = Property(doc="List of Edge objects", default_factory=list)

    def __iter__(self):
        return self.edges.__iter__()

    def __contains__(self, item):
        return item in self.edges

    def add(self, edge):
        self.edges.append(edge)

    def remove(self, edge):
        self.edges.remove(edge)

    def get(self, node_pair):
        from .node import Node
        if not (isinstance(node_pair, Sequence) and
                all(isinstance(node, Node) for node in node_pair)):
            raise TypeError("Must supply a tuple of nodes")
        if not len(node_pair) == 2:
            raise ValueError("Incorrect tuple length. Must be of length 2")
        edges = list()
        for edge in self.edges:
            if edge.nodes == node_pair:
                edges.append(edge)
        return edges

    @property
    def edge_list(self):
        """Returns a list of tuples in the form (sender, recipient)"""
        if not self.edges:
            return []
        return [edge.nodes for edge in self.edges]

    def __len__(self):
        return len(self.edges)


class Message(Base):
    """A message, containing a piece of information, that gets propagated between two Nodes.
    Messages are opened by nodes that are a recipient of the node that sent the message"""
    edge: Edge = Property(
        doc="The directed edge containing the sender and receiver of the message")
    time_pertaining: datetime = Property(
        doc="The latest time for which the data pertains. For a Detection, this would be the time "
            "of the Detection, or for a Track this is the time of the last State in the Track. "
            "Different from time_sent when data is passed on that was not generated by the "
            "sender")
    time_sent: datetime = Property(
        doc="Time at which the message was sent")
    data_piece: DataPiece = Property(
        doc="Info that the sent message contains")
    destinations: set['Node'] = Property(doc="Nodes in the information architecture that the "
                                             "message is being sent to",
                                         default=None)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.status = "sending"

    @property
    def sender_node(self):
        return self.edge.sender

    @property
    def recipient_node(self):
        return self.edge.recipient

    @property
    def arrival_time(self):
        # TODO: incorporate failed time ranges here.
        return self.time_sent + timedelta(seconds=self.edge.ovr_latency)

    def update(self, current_time):
        progress = (current_time - self.time_sent).total_seconds()
        if progress < 0:
            raise ValueError("Current time cannot be before the Message was sent")
        if progress < self.edge.sender.latency:
            self.status = "sending"
        elif progress < self.edge.ovr_latency:
            self.status = "transferring"
        else:
            self.status = "received"

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False

        return all(getattr(self, name) == getattr(other, name)
                   for name in type(self).properties
                   if name not in ['destinations', 'edge'])

    def __hash__(self):
        return hash(tuple(getattr(self, name)
                          for name in type(self).properties
                          if name not in ['destinations', 'edge']))
