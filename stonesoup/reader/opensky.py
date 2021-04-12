# -*- coding: utf-8 -*-
import datetime
from time import sleep
from typing import Tuple

try:
    import requests
    from requests.compat import urljoin
except ImportError as error:
    raise ImportError(
        "Usage of opensky requires the dependency 'requests' is installed. ") from error


from .base import Reader, DetectionReader, GroundTruthReader
from ..base import Property
from ..buffered_generator import BufferedGenerator
from ..types.detection import Detection
from ..types.groundtruth import GroundTruthPath, GroundTruthState
from ..types.state import State


class _OpenSkyNetworkReader(Reader):
    """OpenSky Network reader

    This reader uses the `OpenSky Network <https://opensky-network.org/>`_ REST
    API to fetch air traffic control data.

    The state vector consists of longitude, latitude
    (in decimal degrees) and altitude (in meters).

    .. note::

        By using this reader, you are agreeing to `OpenSky Network's terms of
        use <https://opensky-network.org/about/terms-of-use>`_.

    """

    url = "https://opensky-network.org/"
    sources = {
        0: "ADS-B",
        1: "ASTERIX",
        2: "MLAT",
        3: "FLARM",
    }

    bbox: Tuple[float, float, float, float] = Property(
        default=None,
        doc="Bounding box to filter data to (left, bottom, right, top). "
            "Default `None` which will include global data.")
    timestep: datetime.timedelta = Property(
        default=datetime.timedelta(seconds=15),
        doc="Time of each poll after reported time from OpenSky. "
            "Must be greater than 10 seconds. Default 15 seconds.")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.timestep < datetime.timedelta(seconds=10):
            raise ValueError("'timestep' must be >= 10 seconds.")

    def data_gen(self):
        if self.bbox:
            params = {
             'lomin': self.bbox[0],
             'lamin': self.bbox[1],
             'lomax': self.bbox[2],
             'lamax': self.bbox[3],
             }
        else:
            params = {}  # Global

        url = urljoin(self.url, "api/states/all")
        time = None
        with requests.Session() as session:
            while True:
                response = session.get(url, params=params)
                response.raise_for_status()
                data = response.json()

                states_and_metadata = []
                for state in data['states']:
                    if state[8]:  # On ground
                        continue
                    # Must have position (lon, lat, geo-alt)
                    if not all(state[index] for index in (5, 6, 13)):
                        continue
                    timestamp = datetime.datetime.utcfromtimestamp(state[3])
                    # Skip old detections
                    if time is not None and timestamp <= time:
                        continue

                    states_and_metadata.append((
                        State([[state[5]], [state[6]], [state[13]]], timestamp=timestamp),
                        {
                            'icao24': state[0],
                            'callsign': state[1],
                            'orign_country': state[2],
                            'sensors': state[12],
                            'squawk': state[14],
                            'spi': state[15],
                            'source': self.sources[state[16]],
                        }
                    ))
                time = datetime.datetime.utcfromtimestamp(data['time'])
                yield time, states_and_metadata

                while time + self.timestep > datetime.datetime.utcnow():
                    sleep(0.1)


class OpenSkyNetworkDetectionReader(_OpenSkyNetworkReader, DetectionReader):
    """OpenSky Network detection reader

    This reader uses the `OpenSky Network <https://opensky-network.org/>`_ REST
    API to fetch air traffic control data.

    The detection state vector consists of longitude, latitude
    (in decimal degrees) and altitude (in meters).

    .. note::

        By using this reader, you are agreeing to `OpenSky Network's terms of
        use <https://opensky-network.org/about/terms-of-use>`_.

    """

    @BufferedGenerator.generator_method
    def detections_gen(self):
        for time, states_and_metadata in self.data_gen():
            yield time, {Detection(state.state_vector, state.timestamp, metadata)
                         for state, metadata in states_and_metadata}


class OpenSkyNetworkGroundTruthReader(_OpenSkyNetworkReader, GroundTruthReader):
    """OpenSky Network groundtruth reader

    This reader uses the `OpenSky Network <https://opensky-network.org/>`_ REST
    API to fetch air traffic control data.

    The groundtruth state vector consists of longitude, latitude
    (in decimal degrees) and altitude (in meters).

    Paths that are yielded are grouped based on the International Civil Aviation
    Organisation (ICAO) 24-bit address.

    .. note::

        By using this reader, you are agreeing to `OpenSky Network's terms of
        use <https://opensky-network.org/about/terms-of-use>`_.

    """

    @BufferedGenerator.generator_method
    def groundtruth_paths_gen(self):
        groundtruth_dict = {}

        for time, states_and_metadata in self.data_gen():
            updated_paths = set()
            for state, metadata in states_and_metadata:
                path_id = metadata.get('icao24')
                if path_id is None:
                    path = GroundTruthPath()
                else:
                    if path_id not in groundtruth_dict:
                        groundtruth_dict[path_id] = GroundTruthPath([], id=path_id)
                    path = groundtruth_dict[path_id]

                path.append(GroundTruthState(state.state_vector, state.timestamp, metadata))
                updated_paths.add(groundtruth_dict[path_id])

            yield time, updated_paths
