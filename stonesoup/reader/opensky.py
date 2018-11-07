# -*- coding: utf-8 -*-
import datetime
from time import sleep

import requests
from requests.compat import urljoin

from .base import DetectionReader
from ..base import Property
from ..types.detection import Detection
from ..types.state import StateVector


class OpenSkyReader(DetectionReader):
    url = "https://opensky-network.org/"
    bbox = Property(
        (float, float, float, float),
        default=None,
        doc="left, bottom, right, top")
    timestep = Property(
        datetime.timedelta,
        default=datetime.timedelta(seconds=15),
        doc="Time between each poll. Should be greater than 10 seconds."
            "Default 15 seconds.")
    number_steps = Property(
        int, default=10, doc="Number of time steps to run for")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._detections = set()

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
            for _ in range(self.number_steps):
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
                    if state[13] < 0:  # Negative altitude??
                        continue
                    timestamp = datetime.datetime.utcfromtimestamp(state[3])
                    # Skip old detections
                    if time is not None and timestamp <= time:
                        continue

                    self._detections.add(Detection(
                        StateVector([[state[5]], [state[6]], [state[13]]]),
                        timestamp=timestamp)
                    )
                time = datetime.datetime.utcfromtimestamp(data['time'])
                yield time, self.detections

                while time + self.timestep > datetime.datetime.utcnow():
                    sleep(0.5)
