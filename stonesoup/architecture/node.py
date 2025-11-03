import copy
import threading
from datetime import datetime
from queue import Queue, Empty

from ..base import Property, Base
from ..sensor.sensor import Sensor
from ..types.detection import Detection
from ..types.hypothesis import Hypothesis
from ..types.track import Track
from .edge import DataPiece, FusionQueue
from ..tracker.base import Tracker
from ._functions import _dict_set


class Node(Base):
    """Base Node class. Generally a subclass should be used. Note that most user-defined
    properties are for graphical use only, all with default values. """
    latency: float = Property(
        doc="Contribution to edge latency stemming from this node. Default is 0.0",
        default=0.0)
    label: str = Property(
        doc="Label to be displayed on graph. Default is to label by class and then "
            "differentiate via alphabetical labels",
        default=None)
    position: tuple[float, float] = Property(
        default=None,
        doc="Cartesian coordinates for node. Determined automatically by default")
    colour: str = Property(
        default='#909090',
        doc='Colour to be displayed on graph. Default is grey')
    shape: str = Property(
        default='rectangle',
        doc='Shape used to display nodes. Default is a rectangle')
    font_size: int = Property(
        default=None,
        doc='Font size for node labels. Default is None')
    node_dim: tuple[float, float] = Property(
        default=None,
        doc='Width and height of nodes for graph icons. '
            'Default is None, which will size to label automatically.')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_held = {"fused": {}, "created": {}, "unfused": {}}
        self.messages_to_pass_on = []

    def update(self, time_pertaining, time_arrived, data_piece, category, track=None,
               use_arrival_time=False):
        """
        Updates this Node's data_held using a new data piece.

        Parameters
        ----------
        time_pertaining : datetime
            Time the data is relevant to, when it was created.
        time_arrived : datetime
            Time the data arrived at this node.
        data_piece : DataPiece
            The specific data piece, containing for example a Detection.
        category : str
            A string matching either "fused", "created", or "unfused".
        track : Track, optional
            Track to which the data piece is assigned, if it contains a Hypothesis
            (default is None).
        use_arrival_time : bool, optional
            If True, make `data.timestamp` equal to `time_arrived` (default is False).

        Returns
        -------
        bool
            True if new data has been added.
        """
        if not (isinstance(time_pertaining, datetime) and isinstance(time_arrived, datetime)):
            raise TypeError("Times must be datetime objects")
        if not isinstance(data_piece, DataPiece):
            raise TypeError(f"data_piece must be a DataPiece. Provided type {type(data_piece)}")
        if category not in self.data_held.keys():
            raise ValueError(f"category must be one of {self.data_held.keys()}")
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

        if use_arrival_time and isinstance(self, FusionNode) and \
                category in ("created", "unfused"):
            data = copy.copy(data_piece.data)
            data.timestamp = time_arrived
            if data not in self.fusion_queue.received:
                self.fusion_queue.received.add(data)
                self.fusion_queue.put((time_pertaining, {data}))

        elif isinstance(self, FusionNode) and \
                category in ("created", "unfused") and \
                data_piece.data not in self.fusion_queue.received:
            self.fusion_queue.received.add(data_piece.data)
            self.fusion_queue.put((time_pertaining, {data_piece.data}))

        return added


class SensorNode(Node):
    """A :class:`~.Node` corresponding to a :class:`~.Sensor`. Fresh data is created here"""
    sensor: Sensor = Property(doc="Sensor corresponding to this node")
    colour: str = Property(
        default='#006eff',
        doc='Colour to be displayed on graph. Default is the hex colour code #006eff')
    shape: str = Property(
        default='oval',
        doc='Shape used to display nodes. Default is an oval')


class FusionNode(Node):
    """A :class:`~.Node` that does not measure new data, but does process data it receives"""
    tracker: Tracker = Property(
        doc="Tracker used by this Node to fuse together Tracks and Detections")
    fusion_queue: FusionQueue = Property(
        default=None,
        doc="The queue from which this node draws data to be fused. Default is a standard "
            "FusionQueue")
    tracks: set = Property(default_factory=set,
                           doc="Set of tracks tracked by the fusion node")
    colour: str = Property(
        default='#00b53d',
        doc='Colour to be displayed on graph. Default is the hex colour code #00b53d')
    shape: str = Property(
        default='hexagon',
        doc='Shape used to display nodes. Default is a hexagon')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.fusion_queue:
            if self.tracker.detector:
                self.fusion_queue = self.tracker.detector
            else:
                self.fusion_queue = FusionQueue()
                self.tracker.detector = self.fusion_queue

        self._track_queue = Queue()
        self._tracking_thread = threading.Thread(
            target=self._track_thread,
            args=(self.tracker, self.fusion_queue, self._track_queue),
            daemon=True)

    def fuse(self):
        """
        Fuse data in the fusion queue and update the node's tracks and data_held.

        Returns
        -------
        bool
            True if new data has been added.
        """
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
                _, tracks = data
                self.tracks.update(tracks)
                updated_tracks = updated_tracks.union(tracks)

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
    """A :class:`~.Node` that is both a :class:`~.Sensor` and also processes data"""
    colour: str = Property(
        default='#fc9000',
        doc='Colour to be displayed on graph. Default is the hex colour code #fc9000')
    shape: str = Property(
        default='diamond',
        doc='Shape used to display nodes. Default is a diamond')


class RepeaterNode(Node):
    """A :class:`~.Node` which simply passes data along to others, without manipulating the
    data itself. Consequently, :class:`~.RepeaterNode`s are only used within a
    :class:`~.NetworkArchitecture`"""
    colour: str = Property(
        default='#909090',
        doc='Colour to be displayed on graph. Default is the hex colour code #909090')
    shape: str = Property(
        default='rectangle',
        doc='Shape used to display nodes. Default is a rectangle')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_held = None
