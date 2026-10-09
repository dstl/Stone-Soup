import datetime

import numpy as np
import pytest

pytest.importorskip("cbor2")

from ..cbor import CBORWriter  # noqa: E402
from ...reader.cbor import (  # noqa: E402
    CBORDetectionReader, CBORGroundTruthReader, CBORTrackReader)


def test_detections_cbor(detection_reader, tmpdir):
    filename = tmpdir.join("detections.cbor")

    with CBORWriter(filename.strpath, detections_source=detection_reader) as writer:
        writer.write()

    reader = CBORDetectionReader(filename.strpath)
    for n, (time, detections) in enumerate(reader):
        assert time == datetime.datetime(2018, 1, 1, 14, n)
        assert len(detections) == n
        for detection in detections:
            assert np.array_equal(detection.state_vector, [[n]])
            assert detection.timestamp == time
    assert n == 2


def test_groundtruth_paths_cbor(groundtruth_reader, tmpdir):
    filename = tmpdir.join("groundtruth_paths.cbor")

    with CBORWriter(filename.strpath, groundtruth_source=groundtruth_reader) as writer:
        writer.write()

    reader = CBORGroundTruthReader(filename.strpath)
    for n, (time, paths) in enumerate(reader):
        assert time == datetime.datetime(2018, 1, 1, 14, n)
        assert len(paths) == n
        for path in paths:
            assert path.id == '0'
            assert len(path) == n
    assert n == 1


def test_tracks_cbor(tracker, tmpdir):
    filename = tmpdir.join("tracks.cbor")

    with CBORWriter(filename.strpath, tracks_source=tracker) as writer:
        writer.write()

    reader = CBORTrackReader(filename.strpath)
    for n, (time, tracks) in enumerate(reader):
        assert time == datetime.datetime(2018, 1, 1, 14, n)
        assert len(tracks) == n
        for track in tracks:
            assert track.id == '0'
            assert len(track) == n
            assert track.state.timestamp == time
            assert np.array_equal(track.state_vector, [[1]])
        assert reader.tracks.keys() == {track.id for track in tracks}
    assert n == 1


@pytest.mark.parametrize('timezone', [None, datetime.timezone.utc], ids=['naive', 'utc'])
def test_timezone_cbor(detection_reader, tmpdir, timezone):
    filename = tmpdir.join("detections.cbor")

    with CBORWriter(filename.strpath, detections_source=detection_reader,
                    timezone=timezone) as writer:
        writer.write()

    reader = CBORDetectionReader(filename.strpath)
    for n, (time, detections) in enumerate(reader):
        assert time.tzinfo is timezone
        assert time.replace(tzinfo=None) == datetime.datetime(2018, 1, 1, 14, n)
        for detection in detections:
            assert detection.timestamp == time
    assert n == 2


def test_cbor_bad_init(tmpdir):
    filename = tmpdir.join("bad_init.cbor")
    with pytest.raises(ValueError, match="At least one source required"):
        CBORWriter(filename.strpath)
