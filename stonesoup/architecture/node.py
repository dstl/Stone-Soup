from ..base import Property, Base
from ..sensor.sensor import Sensor
from ..types.detection import Detection
from ..types.hypothesis import Hypothesis
from ..types.track import Track
from ..tracker.base import Tracker
from .edge import DataPiece, FusionQueue
from ..tracker.fusion import SimpleFusionTracker
from datetime import datetime
from typing import Tuple


class Node(Base):
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
    font_size: int = Property(
        default=None,
        doc='Font size for node labels')
    node_dim: tuple = Property(
        default=None,
        doc='Width and height of nodes for graph icons, default is (0.5, 0.5)')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_held = {"fused": {}, "created": {}, "unfused": {}}

    def update(self, time_pertaining, time_arrived, data_piece, category, track=None):
        if not isinstance(time_pertaining, datetime) and isinstance(time_arrived, datetime):
            raise TypeError("Times must be datetime objects")
        if not track:
            if not isinstance(data_piece.data, Detection) and not isinstance(data_piece.data, Track):
                raise TypeError(f"Data provided without accompanying Track must be a Detection or a Track, not a "
                                f"{type(data_piece.data).__name__}")
            new_data_piece = DataPiece(self, data_piece.originator, data_piece.data, time_arrived)
        else:
            if not isinstance(data_piece.data, Hypothesis):
                raise TypeError("Data provided with Track must be a Hypothesis")
            new_data_piece = DataPiece(self, data_piece.originator, data_piece.data, time_arrived, track)

        added, self.data_held[category] = _dict_set(self.data_held[category], new_data_piece, time_pertaining)
        if isinstance(self, FusionNode):
            self.fusion_queue.set_message(new_data_piece)

        return added


class SensorNode(Node):
    """A node corresponding to a Sensor. Fresh data is created here"""
    sensor: Sensor = Property(doc="Sensor corresponding to this node")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.colour:
            self.colour = '#1f77b4'
        if not self.shape:
            self.shape = 'oval'
        if not self.font_size:
            self.font_size = 5
        if not self.node_dim:
            self.node_dim = (0.5, 0.3)


class FusionNode(Node):
    """A node that does not measure new data, but does process data it receives"""
    # feeder probably as well
    tracker: Tracker = Property(
        doc="Tracker used by this Node to fuse together Tracks and Detections")
    fusion_queue: FusionQueue = Property(
        default=FusionQueue(),
        doc="The queue from which this node draws data to be fused")
    track_fusion_tracker: Tracker = Property(
        default=None, #SimpleFusionTracker(),
        doc="Tracker for associating tracks at the node")
    tracks: set = Property(default=None,
                           doc="Set of tracks tracked by the fusion node")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.colour:
            self.colour = '#006400'
        if not self.shape:
            self.shape = 'hexagon'
        if not self.font_size:
            self.font_size = 5
        if not self.node_dim:
            self.node_dim = (0.6, 0.3)
        self.tracks = set()  # Set of tracks this Node has recorded

    def fuse(self):
        print("A node be fusin")
        # we have a queue.
        data = self.fusion_queue.get_message()

        # Sort detections and tracks and group by time

        if data:
            # track it
            print("there's data")
            for time, track in self.tracker:
                self.tracks.update(track)
        else:
            print("no data")
            return


class SensorFusionNode(SensorNode, FusionNode):
    """A node that is both a sensor and also processes data"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.colour in ['#006400', '#1f77b4']:
            self.colour = '#909090'  # attr dict in Architecture.__init__ also needs updating
        if self.shape in ['oval', 'hexagon']:
            self.shape = 'rectangle'
        if not self.font_size:
            self.font_size = 5
        if not self.node_dim:
            self.node_dim = (0.1, 0.3)


class RepeaterNode(Node):
    """A node which simply passes data along to others, without manipulating the data itself. """
    # Latency property could go here
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.colour:
            self.colour = '#ff7f0e'
        if not self.shape:
            self.shape = 'circle'
        if not self.font_size:
            self.font_size = 5
        if not self.node_dim:
            self.node_dim = (0.5, 0.3)


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
