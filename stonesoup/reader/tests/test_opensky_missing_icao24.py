import datetime

import pytest

pytest.importorskip('requests')

from ..opensky import OpenSkyNetworkGroundTruthReader  # noqa: E402
from ...types.state import State  # noqa: E402


def test_opensky_groundtruth_without_icao24_is_yielded(monkeypatch):
    timestamp = datetime.datetime(2026, 1, 1)
    state = State([[1.0], [2.0], [3.0]], timestamp=timestamp)
    metadata = {'icao24': None, 'callsign': 'TEST'}

    reader = OpenSkyNetworkGroundTruthReader()
    monkeypatch.setattr(
        reader,
        'data_gen',
        lambda: iter([(timestamp, [(state, metadata)])]),
    )

    time, paths = next(reader.groundtruth_paths_gen())

    assert time == timestamp
    assert len(paths) == 1
    path = next(iter(paths))
    assert len(path) == 1
    assert path[0].state_vector.tolist() == [[1.0], [2.0], [3.0]]
    assert path[0].metadata['icao24'] is None
