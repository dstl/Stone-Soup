import datetime

import numpy as np
import pytest

pytest.importorskip('requests')

from .. import opensky  # noqa: E402
from ..opensky import OpenSkyNetworkDetectionReader  # noqa: E402


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


class _Session:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def get(self, *args, **kwargs):
        return _Response(self.payload)


def _state_vector(longitude, latitude, geo_altitude):
    return [
        'abc123', 'TEST123', 'United Kingdom', 1704067199, 1704067199,
        longitude, latitude, 0.0, False, 120.0, 90.0, 0.0, None,
        geo_altitude, '7000', False, 0,
    ]


def test_opensky_accepts_zero_position_values_and_rejects_missing_values(monkeypatch):
    payload = {
        'time': 1704067200,
        'states': [
            _state_vector(0.0, 0.0, 0.0),
            _state_vector(None, 1.0, 100.0),
        ],
    }
    monkeypatch.setattr(opensky.requests, 'Session', lambda: _Session(payload))

    reader = OpenSkyNetworkDetectionReader()
    time, states_and_metadata = next(reader.data_gen())

    assert time == datetime.datetime(2024, 1, 1, 0, 0)
    assert len(states_and_metadata) == 1

    state, metadata = states_and_metadata[0]
    assert np.array_equal(state.state_vector, [[0.0], [0.0], [0.0]])
    assert metadata['icao24'] == 'abc123'
