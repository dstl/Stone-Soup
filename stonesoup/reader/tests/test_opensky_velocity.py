import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytest.importorskip('requests')

from ..opensky import (  # noqa: E402
    OpenSkyNetworkDetectionReader, OpenSkyNetworkGroundTruthReader,
    _OpenSkyNetworkReader)


def _opensky_state(
        icao24='abcdef',
        callsign='TEST123 ',
        lon=1.0,
        lat=51.0,
        geo_alt=1000.0,
        on_ground=False,
        velocity=100.0,
        true_track=90.0,
        vertical_rate=5.0,
        time_position=1_700_000_000):
    """Build a single OpenSky REST state vector for tests."""
    return [
        icao24,            # 0 icao24
        callsign,          # 1 callsign
        'United Kingdom',  # 2 origin_country
        time_position,     # 3 time_position
        time_position,     # 4 last_contact
        lon,               # 5 longitude
        lat,               # 6 latitude
        geo_alt,           # 7 baro_altitude
        on_ground,         # 8 on_ground
        velocity,          # 9 velocity
        true_track,        # 10 true_track
        vertical_rate,     # 11 vertical_rate
        None,              # 12 sensors
        geo_alt,           # 13 geo_altitude
        '7000',            # 14 squawk
        False,             # 15 spi
        0,                 # 16 position_source (ADS-B)
    ]


@pytest.mark.parametrize(
    'true_track,velocity,vertical_rate,expected',
    [
        # Due East: East = |v|, North ≈ 0
        (90.0, 100.0, 5.0, (100.0, 0.0, 5.0)),
        # Due North: East ≈ 0, North = |v|
        (0.0, 50.0, -2.0, (0.0, 50.0, -2.0)),
        # North-East (45°): equal East and North components
        (45.0, np.sqrt(2) * 10.0, 0.0, (10.0, 10.0, 0.0)),
    ],
    ids=['east', 'north', 'north-east'])
def test_velocity_components(true_track, velocity, vertical_rate, expected):
    state = _opensky_state(
        velocity=velocity, true_track=true_track, vertical_rate=vertical_rate)
    east, north, up = _OpenSkyNetworkReader._velocity_components(state)
    assert east == pytest.approx(expected[0], abs=1e-9)
    assert north == pytest.approx(expected[1], abs=1e-9)
    assert up == pytest.approx(expected[2], abs=1e-9)


@pytest.mark.parametrize(
    'velocity,true_track,vertical_rate',
    [
        (None, 90.0, 1.0),
        (10.0, None, 1.0),
        (10.0, 90.0, None),
        (None, None, None),
    ],
    ids=['no-velocity', 'no-track', 'no-rate', 'all-missing'])
def test_velocity_components_missing(velocity, true_track, vertical_rate):
    state = _opensky_state(
        velocity=velocity, true_track=true_track, vertical_rate=vertical_rate)
    assert _OpenSkyNetworkReader._velocity_components(state) is None


def _mock_response(states, time=1_700_000_010):
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json.return_value = {'time': time, 'states': states}
    return response


def test_include_velocity_false_keeps_position_only():
    """Default behaviour: state vector is lon, lat, alt only."""
    states = [
        _opensky_state(icao24='aaaaaa', velocity=100.0, true_track=90.0),
        _opensky_state(
            icao24='bbbbbb', lon=2.0, lat=52.0, geo_alt=2000.0,
            velocity=None, true_track=None, vertical_rate=None,
            time_position=1_700_000_001),
    ]
    with patch('stonesoup.reader.opensky.requests.Session') as Session:
        session = Session.return_value.__enter__.return_value
        session.get.return_value = _mock_response(states)

        reader = OpenSkyNetworkDetectionReader(include_velocity=False)
        time, detections = next(iter(reader))

    assert isinstance(time, datetime.datetime)
    assert len(detections) == 2
    for detection in detections:
        assert detection.state_vector.shape == (3, 1)


def test_include_velocity_true_appends_enu_and_skips_incomplete():
    """With include_velocity, incomplete velocity reports are skipped."""
    complete = _opensky_state(
        icao24='aaaaaa', velocity=100.0, true_track=90.0, vertical_rate=5.0)
    incomplete = _opensky_state(
        icao24='bbbbbb', lon=2.0, lat=52.0, geo_alt=2000.0,
        velocity=None, true_track=90.0, vertical_rate=5.0,
        time_position=1_700_000_001)
    on_ground = _opensky_state(
        icao24='cccccc', on_ground=True, time_position=1_700_000_002)

    with patch('stonesoup.reader.opensky.requests.Session') as Session:
        session = Session.return_value.__enter__.return_value
        session.get.return_value = _mock_response(
            [complete, incomplete, on_ground])

        reader = OpenSkyNetworkDetectionReader(include_velocity=True)
        time, detections = next(iter(reader))

    assert len(detections) == 1
    detection = detections.pop()
    assert detection.state_vector.shape == (6, 1)
    assert detection.state_vector[0, 0] == pytest.approx(1.0)   # lon
    assert detection.state_vector[1, 0] == pytest.approx(51.0)  # lat
    assert detection.state_vector[2, 0] == pytest.approx(1000.0)  # alt
    assert detection.state_vector[3, 0] == pytest.approx(100.0)  # East
    assert detection.state_vector[4, 0] == pytest.approx(0.0, abs=1e-9)  # North
    assert detection.state_vector[5, 0] == pytest.approx(5.0)   # Up
    assert detection.metadata['icao24'] == 'aaaaaa'
    assert detection.metadata['source'] == 'ADS-B'


def test_include_velocity_groundtruth_paths():
    states = [
        _opensky_state(
            icao24='abcdef', velocity=50.0, true_track=0.0, vertical_rate=-1.0),
    ]
    with patch('stonesoup.reader.opensky.requests.Session') as Session:
        session = Session.return_value.__enter__.return_value
        session.get.return_value = _mock_response(states)

        reader = OpenSkyNetworkGroundTruthReader(include_velocity=True)
        time, paths = next(iter(reader))

    assert len(paths) == 1
    path = paths.pop()
    assert path.id == 'abcdef'
    assert len(path) == 1
    assert path[-1].state_vector.shape == (6, 1)
    assert path[-1].state_vector[3, 0] == pytest.approx(0.0, abs=1e-9)  # East
    assert path[-1].state_vector[4, 0] == pytest.approx(50.0)  # North
    assert path[-1].state_vector[5, 0] == pytest.approx(-1.0)  # Up


def test_include_velocity_rejects_short_timestep():
    with pytest.raises(ValueError, match='timestep'):
        OpenSkyNetworkDetectionReader(
            timestep=datetime.timedelta(seconds=5), include_velocity=True)
