# -*- coding: utf-8 -*-
import datetime
from time import sleep

import requests
from requests.compat import urljoin

from .base import DetectionReader
from ..base import Property
from ..types.detection import Detection
from ..types.state import StateVector


class OpenSkyNetworkReader(DetectionReader):
    """OpenSky Network reader

    This reader uses the `OpenSky Network <https://opensky-network.org/>`_ REST
    API to fetch air traffic control data.

    The detection state vector consists of longitude, latitude
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

    bbox = Property(
        (float, float, float, float),
        default=None,
        doc="Bounding box to filter data to (left, bottom, right, top). "
            "Default `None` which will include global data.")
    timestep = Property(
        datetime.timedelta,
        default=datetime.timedelta(seconds=15),
        doc="Time between each poll. Must be greater than 10 seconds."
            "Default 15 seconds.")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._detections = set()

        if self.timestep < datetime.timedelta(seconds=10):
            raise ValueError("'timestep' must be >= 10 seconds.")

    @property
    def detections(self):
        return self._detections.copy()

    def detections_gen(self):
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

                self._detections = set()
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

                    self._detections.add(Detection(
                        StateVector([[state[5]], [state[6]], [state[13]]]),
                        timestamp=timestamp,
                        metadata={
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
                yield time, self.detections

                while time + self.timestep > datetime.datetime.utcnow():
                    sleep(0.5)
