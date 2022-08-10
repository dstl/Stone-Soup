from textwrap import dedent

import pytest

from ..yaml import YAMLWriter


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
           - state_vector: !stonesoup.types.array.StateVector
             - [1]
           - timestamp: *id001
           - metadata: {}
           :
         ...
         ---
         time: &id001 2018-01-01 14:02:00
         detections: !!set
           ? !stonesoup.types.detection.Detection
           - state_vector: !stonesoup.types.array.StateVector
             - [2]
           - timestamp: *id001
           - metadata: {}
           :
           ? !stonesoup.types.detection.Detection
           - state_vector: !stonesoup.types.array.StateVector
             - [2]
           - timestamp: *id001
           - metadata: {}
           :
         ...
         """)

    assert generated_yaml == expected_yaml


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
              - state_vector: !stonesoup.types.array.StateVector
                - [1]
              - timestamp: *id001
              - metadata: {}
          - id: '0'
          :
        ...
        """)

    assert generated_yaml == expected_yaml


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
              - state_vector: !stonesoup.types.array.StateVector
                - [1]
              - timestamp: *id001
          - id: '0'
          :
        ...
        """)

    assert generated_yaml == expected_yaml


def test_yaml_bad_init(tmpdir):
    filename = tmpdir.join("bad_init.yaml")
    with pytest.raises(ValueError, match="At least one source required"):
        YAMLWriter(filename.strpath)
