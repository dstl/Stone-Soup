# -*- coding: utf-8 -*-
import sys
from textwrap import dedent

import pytest

from ..yaml import YAMLWriter


dict_order_skip = pytest.mark.skipif(
    sys.version_info < (3, 6), reason="requires Python >= 3.6 for dict order")


@dict_order_skip
def test_detections_yaml(detection_reader, tmpdir):
    filename = tmpdir.join("detections.yaml")

    with YAMLWriter(filename.strpath,
                    detections_source=detection_reader) as writer:
        writer.write()

    with filename.open('r') as yaml_file:
        generated_yaml = yaml_file.read()

    expected_yaml = dedent("""\
         ---
         time: 2018-01-01 14:00:00
         detections: !!set {}
         ...
         ---
         time: &id001 2018-01-01 14:01:00
         detections: !!set
           ? !stonesoup.types.detection.Detection
           - state_vector: !numpy.ndarray
             - [1]
           - timestamp: *id001
           :
         ...
         ---
         time: &id001 2018-01-01 14:02:00
         detections: !!set
           ? !stonesoup.types.detection.Detection
           - state_vector: !numpy.ndarray
             - [2]
           - timestamp: *id001
           :
           ? !stonesoup.types.detection.Detection
           - state_vector: !numpy.ndarray
             - [2]
           - timestamp: *id001
           :
         ...
         """)

    assert generated_yaml == expected_yaml


@dict_order_skip
def test_groundtruth_paths_yaml(groundtruth_reader, tmpdir):
    filename = tmpdir.join("groundtruth_paths.yaml")

    with YAMLWriter(filename.strpath,
                    groundtruth_source=groundtruth_reader) as writer:
        writer.write()

    with filename.open('r') as yaml_file:
        generated_yaml = yaml_file.read()

    expected_yaml = dedent("""\
        ---
        time: 2018-01-01 14:00:00
        groundtruth_paths: !!set {}
        ...
        ---
        time: &id001 2018-01-01 14:01:00
        groundtruth_paths: !!set
          ? !stonesoup.types.groundtruth.GroundTruthPath
          - states:
            - !stonesoup.types.groundtruth.GroundTruthState
              - state_vector: !numpy.ndarray
                - [1]
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
                - [2]
              - timestamp: *id001
            - !stonesoup.types.groundtruth.GroundTruthState
              - state_vector: !numpy.ndarray
                - [12]
              - timestamp: *id001
          :
          ? !stonesoup.types.groundtruth.GroundTruthPath
          - states:
            - !stonesoup.types.groundtruth.GroundTruthState
              - state_vector: !numpy.ndarray
                - [2]
              - timestamp: *id001
            - !stonesoup.types.groundtruth.GroundTruthState
              - state_vector: !numpy.ndarray
                - [12]
              - timestamp: *id001
          :
        ...
        """)

    assert generated_yaml == expected_yaml


@dict_order_skip
def test_tracks_yaml(tracker, tmpdir):
    filename = tmpdir.join("tracks.yaml")

    with YAMLWriter(filename.strpath, tracks_source=tracker) as writer:
        writer.write()

    with filename.open('r') as yaml_file:
        generated_yaml = yaml_file.read()

    expected_yaml = dedent("""\
        ---
        time: 2018-01-01 14:00:00
        tracks: !!set {}
        ...
        ---
        time: &id001 2018-01-01 14:01:00
        tracks: !!set
          ? !stonesoup.types.track.Track
          - states:
            - !stonesoup.types.state.State
              - state_vector: !numpy.ndarray
                - [1]
              - timestamp: *id001
          :
        ...
        ---
        time: &id001 2018-01-01 14:02:00
        tracks: !!set
          ? !stonesoup.types.track.Track
          - states:
            - !stonesoup.types.state.State
              - state_vector: !numpy.ndarray
                - [2]
              - timestamp: *id001
            - !stonesoup.types.state.State
              - state_vector: !numpy.ndarray
                - [12]
              - timestamp: *id001
          :
          ? !stonesoup.types.track.Track
          - states:
            - !stonesoup.types.state.State
              - state_vector: !numpy.ndarray
                - [2]
              - timestamp: *id001
            - !stonesoup.types.state.State
              - state_vector: !numpy.ndarray
                - [12]
              - timestamp: *id001
          :
        ...
        """)

    assert generated_yaml == expected_yaml


def test_yaml_bad_init(tmpdir):
    filename = tmpdir.join("bad_init.yaml")
    with pytest.raises(ValueError, match="At least one source required"):
        YAMLWriter(filename.strpath)
