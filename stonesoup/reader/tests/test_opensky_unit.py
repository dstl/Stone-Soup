import datetime

import pytest

pytest.importorskip('requests')

from .. import opensky  # noqa: E402


class _Response:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self._data


class _Session:
    def __init__(self, data):
        self._data = data

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def get(self, *args, **kwargs):
        return _Response(self._data)


def _state(longitude, latitude, altitude):
    return [
        'abc123', 'TEST', 'GB', 100, 100,
        longitude, latitude, altitude, False,
        100, 90, 0, None, altitude, None, False, 0,
    ]


def test_opensky_zero_position_values_are_valid(monkeypatch):
    data = {
        'time': 100,
        'states': [
            _state(0.0, 0.0, 0.0),
            _state(None, 1.0, 100.0),
        ],
    }
    monkeypatch.setattr(opensky.requests, 'Session', lambda: _Session(data))

    reader = opensky.OpenSkyNetworkDetectionReader(
        timestep=datetime.timedelta(seconds=10))
    _, states_and_metadata = next(reader.data_gen())

    assert len(states_and_metadata) == 1
    state, _ = states_and_metadata[0]
    assert state.state_vector.tolist() == [[0.0], [0.0], [0.0]]
