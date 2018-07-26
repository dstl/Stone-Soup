# -*- coding: utf-8 -*-
import datetime
from textwrap import dedent

import numpy as np

from ..yaml import YAMLReader


def test_detections_yaml(tmpdir):
    filename = tmpdir.join("detections.yaml")
    with filename.open('w') as file:
        file.write(dedent("""\
            ---
            time: &id001 2018-01-01 14:00:00
            detections: !!set {}
            ...
            ---
            time: &id001 2018-01-01 14:01:00
            detections: !!set
              ? !stonesoup.types.detection.Detection
              - state_vector: !numpy.ndarray
                - [11]
                - [21]
              - timestamp: *id001
              :
            ...
            ---
            time: &id001 2018-01-01 14:02:00
            detections: !!set
              ? !stonesoup.types.detection.Detection
              - state_vector: !numpy.ndarray
                - [12]
                - [22]
              - timestamp: *id001
              :
              ? !stonesoup.types.detection.Clutter
              - state_vector: !numpy.ndarray
                - [12]
                - [22]
              - timestamp: *id001
              :
            """))

    reader = YAMLReader(filename.strpath)

    for n, (time, detections) in enumerate(reader.detections_gen()):
        assert len(detections) == n
        for detection in detections:
            assert np.array_equal(
                detection.state_vector, np.array([[10 + n], [20 + n]]))
            assert detection.timestamp.hour == 14
            assert detection.timestamp.minute == n
            assert detection.timestamp.date() == datetime.date(2018, 1, 1)
            assert detection.timestamp == time

        assert not reader.groundtruth_paths
        assert not reader.sensor_data
        assert not reader.tracks


def test_groundtruth_paths_yaml(tmpdir):
    filename = tmpdir.join("groundtruth_paths.yaml")
    with filename.open('w') as file:
        file.write(dedent("""\
            ---
            time: &id001 2018-01-01 14:00:00
            groundtruth_paths: !!set {}
            ...
            ---
            time: &id001 2018-01-01 14:01:00
            groundtruth_paths: !!set
              ? !stonesoup.types.groundtruth.GroundTruthPath
              - states:
                - !stonesoup.types.groundtruth.GroundTruthState
                  - state_vector: !numpy.ndarray
                    - [11]
                    - [12]
                  - timestamp: *id001
              :
            ...
            ---
            time: &id001 2018-01-01 14:02:00
            groundtruth_paths: !!set
              ? !stonesoup.types.groundtruth.GroundTruthPath
              - states:
                - !stonesoup.types.groundtruth.GroundTruthState
                  - state_vector: !numpy.ndarray
                    - [11]
                    - [12]
                  - timestamp: 2018-01-01 14:01:00
                - !stonesoup.types.groundtruth.GroundTruthState
                  - state_vector: !numpy.ndarray
                    - [21]
                    - [22]
                  - timestamp: *id001
              :
              ? !stonesoup.types.groundtruth.GroundTruthPath
              - states:
                - !stonesoup.types.groundtruth.GroundTruthState
                  - state_vector: !numpy.ndarray
                    - [31]
                    - [32]
                  - timestamp: *id001
              :
            ...
            """))

    reader = YAMLReader(filename.strpath)

    ptime = None
    for n, (time, paths) in enumerate(reader.groundtruth_paths_gen()):
        assert len(paths) == n
        for path in paths:
            assert time.hour == 14
            assert time.minute == n
            assert time.date() == datetime.date(2018, 1, 1)
            assert path[-1].timestamp == time
            if len(path) > 1:
                assert path[-2].timestamp == ptime
        assert not reader.sensor_data
        assert not reader.detections
        assert not reader.tracks
        ptime = time


def test_tracks_yaml(tmpdir):
    filename = tmpdir.join("tracks.yaml")
    with filename.open('w') as file:
        file.write(dedent("""\
            ---
            time: &id001 2018-01-01 14:00:00
            tracks: !!set {}
            ...
            ---
            time: &id001 2018-01-01 14:01:00
            tracks: !!set
              ? !stonesoup.types.track.Track
              - states:
                - !stonesoup.types.track.State
                  - state_vector: !numpy.ndarray
                    - [11]
                    - [12]
                  - timestamp: *id001
              :
            ...
            ---
            time: &id001 2018-01-01 14:02:00
            tracks: !!set
              ? !stonesoup.types.track.Track
              - states:
                - !stonesoup.types.track.State
                  - state_vector: !numpy.ndarray
                    - [11]
                    - [12]
                  - timestamp: 2018-01-01 14:01:00
                - !stonesoup.types.track.State
                  - state_vector: !numpy.ndarray
                    - [21]
                    - [22]
                  - timestamp: *id001
              :
              ? !stonesoup.types.track.Track
              - states:
                - !stonesoup.types.track.State
                  - state_vector: !numpy.ndarray
                    - [31]
                    - [32]
                  - timestamp: *id001
              :
            ...
            """))

    reader = YAMLReader(filename.strpath)

    ptime = None
    for n, (time, tracks) in enumerate(reader.tracks_gen()):
        assert len(tracks) == n
        for track in tracks:
            assert time.hour == 14
            assert time.minute == n
            assert time.date() == datetime.date(2018, 1, 1)
            assert track.timestamp == time
            if len(track.states) > 1:
                assert track.states[-2].timestamp == ptime
        assert not reader.groundtruth_paths
        assert not reader.sensor_data
        assert not reader.detections
        ptime = time
