import copy
import threading
from datetime import datetime
from queue import Queue, Empty
from typing import Tuple

from ..base import Property, Base
from ..sensor.sensor import Sensor
from ..types.detection import Detection
from ..types.hypothesis import Hypothesis
from ..types.track import Track
from .edge import DataPiece, FusionQueue
from ..tracker.base import Tracker
from ._functions import _dict_set


class Node(Base):
    """Base node class. Should be abstract"""
    latency: float = Property(
        doc="Contribution to edge latency stemming from this node",
        default=0.0)
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
        default=5,
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
            if not isinstance(data_piece.data, Detection) and \
                    not isinstance(data_piece.data, Track):
                raise TypeError(f"Data provided without accompanying Track must be a Detection or "
                                f"a Track, not a "
                                f"{type(data_piece.data).__name__}")
            new_data_piece = DataPiece(self, data_piece.originator, data_piece.data, time_arrived)
        else:
            if not isinstance(data_piece.data, Hypothesis):
                raise TypeError("Data provided with Track must be a Hypothesis")
            new_data_piece = DataPiece(self, data_piece.originator, data_piece.data,
                                       time_arrived, track)

        added, self.data_held[category] = _dict_set(self.data_held[category],
                                                    new_data_piece, time_pertaining)
        if isinstance(self, FusionNode) and category in ("created", "unfused"):
            self.fusion_queue.put((time_pertaining, {data_piece.data}))

        return added


class SensorNode(Node):
    """A node corresponding to a Sensor. Fresh data is created here"""
    sensor: Sensor = Property(doc="Sensor corresponding to this node")
    colour: str = Property(
        default='#1f77b4',
        doc='Colour to be displayed on graph. Default is the hex colour code #1f77b4')
    shape: str = Property(
        default='oval',
        doc='Shape used to display nodes. Default is an oval')
    node_dim: tuple = Property(
        default=(0.5, 0.3),
        doc='Width and height of nodes for graph icons. Default is (0.5, 0.3)')


class FusionNode(Node):
    """A node that does not measure new data, but does process data it receives"""
    # feeder probably as well
    tracker: Tracker = Property(
        doc="Tracker used by this Node to fuse together Tracks and Detections")
    fusion_queue: FusionQueue = Property(
        default=FusionQueue(),
        doc="The queue from which this node draws data to be fused")
    tracks: set = Property(default=None,
                           doc="Set of tracks tracked by the fusion node")
    colour: str = Property(
        default='#006400',
        doc='Colour to be displayed on graph. Default is the hex colour code #006400')
    shape: str = Property(
        default='hexagon',
        doc='Shape used to display nodes. Default is a hexagon')
    node_dim: tuple = Property(
        default=(0.6, 0.3),
        doc='Width and height of nodes for graph icons. Default is (0.6, 0.3)')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tracks = set()  # Set of tracks this Node has recorded

        self._track_queue = Queue()
        self._tracking_thread = threading.Thread(
            target=self._track_thread,
            args=(self.tracker, self.fusion_queue, self._track_queue),
            daemon=True)

    def fuse(self):
        if not self._tracking_thread.is_alive():
            try:
                self._tracking_thread.start()
            except RuntimeError:  # Previously started
                raise RuntimeError(f"Tracking thread in {self.label!r} unexpectedly ended")

        added = False
        updated_tracks = set()
        while True:
            waiting_for_data = self.fusion_queue.waiting_for_data
            try:
                data = self._track_queue.get(timeout=1e-6)
            except Empty:
                if not self._tracking_thread.is_alive() or waiting_for_data:
                    break
            else:
                time, tracks = data
                self.tracks.update(tracks)
                updated_tracks |= tracks

        for track in updated_tracks:
            data_piece = DataPiece(self, self, copy.copy(track), track.timestamp, True)
            added, self.data_held['fused'] = _dict_set(
                self.data_held['fused'], data_piece, track.timestamp)
        return added

    @staticmethod
    def _track_thread(tracker, input_queue, output_queue):
        for time, tracks in tracker:
            output_queue.put((time, tracks))
            input_queue.task_done()


class SensorFusionNode(SensorNode, FusionNode):
    """A node that is both a sensor and also processes data"""
    colour: str = Property(
        default='#909090',
        doc='Colour to be displayed on graph. Default is the hex colour code #909090')
    shape: str = Property(
        default='rectangle',
        doc='Shape used to display nodes. Default is a rectangle')
    node_dim: tuple = Property(
        default=(0.1, 0.3),
        doc='Width and height of nodes for graph icons. Default is (0.1, 0.3)')


class RepeaterNode(Node):
    """A node which simply passes data along to others, without manipulating the data itself. """
    colour: str = Property(
        default='#ff7f0e',
        doc='Colour to be displayed on graph. Default is the hex colour code #ff7f0e')
    shape: str = Property(
        default='circle',
        doc='Shape used to display nodes. Default is a circle')
    node_dim: tuple = Property(
        default=(0.5, 0.3),
        doc='Width and height of nodes for graph icons. Default is (0.5, 0.3)')
