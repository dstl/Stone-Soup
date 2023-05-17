import datetime
import uuid
from typing import Set, List
import collections.abc

from ...base import Property
from ...types.base import Type
from ...types.track import Track
from ...types.detection import Detection
from ...models.transition import TransitionModel


class Tracklet(Track):
    pass


class SensorTracks(Type, collections.abc.Set):
    """ A container object for tracks relating to a particular sensor """
    tracks: Set[Track] = Property(doc='A list of tracks', default=None)
    sensor_id: str = Property(doc='The id of the sensor', default=None)
    transition_model: TransitionModel = Property(doc='Transition model used by the tracker',
                                                 default=None)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.tracks is None:
            self.tracks = set()

    def __iter__(self):
        return (t for t in self.tracks)

    def __len__(self):
        return self.tracks.__len__()

    def __contains__(self, item):
        return self.tracks.__contains__(item)


class SensorTracklets(SensorTracks):
    """ A container object for tracklets relating to a particular sensor """
    pass


class SensorScan(Type):
    """ A wrapper around a set of detections produced by a particular sensor """
    sensor_id: str = Property(doc='The id of the sensor')
    detections: Set[Detection] = Property(doc='The detections contained in the scan')
    id: str = Property(default=None, doc="The unique scan ID")
    timestamp: datetime.datetime = Property(default=None, doc='The scan timestamp')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.id is None:
            self.id = str(uuid.uuid4())


class Scan(Type):
    """ A wrapper around a set of sensor scans within a given time interval  """
    start_time: datetime.datetime = Property(doc='The scan start time')
    end_time: datetime.datetime = Property(doc='The scan end time')
    sensor_scans: List[SensorScan] = Property(doc='The sensor scans')
    id: str = Property(default=None, doc="The unique scan ID")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.id is None:
            self.id = str(uuid.uuid4())


