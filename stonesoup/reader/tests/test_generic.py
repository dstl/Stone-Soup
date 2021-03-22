# -*- coding: utf-8 -*-
import datetime
from operator import attrgetter
from textwrap import dedent

import numpy as np
import pytest

from ..generic import CSVDetectionReader, CSVGroundTruthReader


@pytest.fixture()
def csv_gt_filename(tmpdir):
    csv_filename = tmpdir.join("test.csv")
    with csv_filename.open('w') as csv_file:
        csv_file.write(dedent("""\
                x,y,z,identifier,t
                10,20,30,22018332,2018-01-01T14:00:00Z
                11,21,31,22018332,2018-01-01T14:01:00Z
                12,22,32,22018332,2018-01-01T14:02:00Z
                13,23,33,32018332,2018-01-01T14:03:00Z
                14,24,34,32018332,2018-01-01T14:04:00Z
                """))
    return csv_filename


def test_csv_gt_2d(csv_gt_filename):
    # run test with:
    #   - 2d co-ordinates
    #   - default time field format
    #   - no special csv options
    csv_reader = CSVGroundTruthReader(csv_gt_filename.strpath,
                                      state_vector_fields=["x", "y"],
                                      time_field="t",
                                      path_id_field="identifier")

    final_gt_paths = set()
    for _, gt_paths_at_timestep in csv_reader:
        final_gt_paths.update(gt_paths_at_timestep)
    assert len(final_gt_paths) == 2

    ground_truth_states = [
        gt_state
        for gt_path in final_gt_paths
        for gt_state in gt_path]

    for n, gt_state in enumerate(
            sorted(ground_truth_states, key=attrgetter('timestamp'))):
        assert np.array_equal(
            gt_state.state_vector, np.array([[10 + n], [20 + n]]))
        assert gt_state.timestamp.hour == 14
        assert gt_state.timestamp.minute == n
        assert gt_state.timestamp.date() == datetime.date(2018, 1, 1)


def test_csv_gt_3d_time(csv_gt_filename):
    # run test with:
    #   - 3d co-ordinates
    #   - time field format specified
    csv_reader = CSVGroundTruthReader(csv_gt_filename.strpath,
                                      state_vector_fields=["x", "y", "z"],
                                      time_field="t",
                                      time_field_format="%Y-%m-%dT%H:%M:%SZ",
                                      path_id_field="identifier")

    final_gt_paths = set()
    for _, gt_paths_at_timestep in csv_reader:
        final_gt_paths.update(gt_paths_at_timestep)
    assert len(final_gt_paths) == 2

    ground_truth_states = [
        gt_state
        for gt_path in final_gt_paths
        for gt_state in gt_path]

    for n, gt_state in enumerate(
            sorted(ground_truth_states, key=attrgetter('timestamp'))):
        assert np.array_equal(
            gt_state.state_vector, np.array([[10 + n], [20 + n], [30 + n]]))
        assert gt_state.timestamp.hour == 14
        assert gt_state.timestamp.minute == n
        assert gt_state.timestamp.date() == datetime.date(2018, 1, 1)


def test_csv_gt_3d_timestamp_csv_opt(tmpdir):
    # run test with:
    #   - time field represented as a Unix epoch timestamp
    #   - csv options specified
    csv_gt_filename = tmpdir.join("test.csv")
    with csv_gt_filename.open('w') as csv_file:
        csv_file.write(dedent("""\
                10,20,30,22018332,1514815200
                11,21,31,22018332,1514815260
                12,22,32,22018332,1514815320
                13,23,33,32018332,1514815380
                14,24,34,32018332,1514815440
                """))

    csv_reader = CSVGroundTruthReader(csv_gt_filename.strpath,
                                      state_vector_fields=["x", "y", "z"],
                                      time_field="t",
                                      timestamp=True,
                                      path_id_field="identifier",
                                      csv_options={'fieldnames':
                                                    ['x', 'y', 'z',
                                                        'identifier', 't']})

    final_gt_paths = set()
    for _, gt_paths_at_timestep in csv_reader:
        final_gt_paths.update(gt_paths_at_timestep)
    assert len(final_gt_paths) == 2

    ground_truth_states = [
        gt_state
        for gt_path in final_gt_paths
        for gt_state in gt_path]

    for n, gt_state in enumerate(
            sorted(ground_truth_states, key=attrgetter('timestamp'))):
        assert np.array_equal(
            gt_state.state_vector, np.array([[10 + n], [20 + n], [30 + n]]))
        assert gt_state.timestamp.hour == 14
        assert gt_state.timestamp.minute == n
        assert gt_state.timestamp.date() == datetime.date(2018, 1, 1)


def test_csv_gt_multi_per_timestep(tmpdir):
    csv_gt_filename = tmpdir.join("test.csv")
    with csv_gt_filename.open('w') as csv_file:
        csv_file.write(dedent("""\
                x,y,z,identifier,t
                10,20,30,22018332,2018-01-01T14:00:00Z
                11,21,31,22018332,2018-01-01T14:01:00Z
                12,22,32,22018332,2018-01-01T14:02:00Z
                13,23,33,32018332,2018-01-01T14:02:00Z
                14,24,34,32018332,2018-01-01T14:03:00Z
                """))

    csv_reader = CSVGroundTruthReader(csv_gt_filename.strpath,
                                      state_vector_fields=["x", "y"],
                                      time_field="t",
                                      path_id_field="identifier")

    for time, ground_truth_paths in csv_reader:
        if time == datetime.datetime(2018, 1, 1, 14, 2):
            assert len(ground_truth_paths) == 2
        else:
            assert len(ground_truth_paths) == 1


@pytest.fixture()
def csv_det_filename(tmpdir):
    csv_filename = tmpdir.join("test.csv")
    with csv_filename.open('w') as csv_file:
        csv_file.write(dedent("""\
            x,y,z,identifier,t
            10,20,30,22018332,2018-01-01T14:00:00Z
            11,21,31,22018332,2018-01-01T14:01:00Z
            12,22,32,22018332,2018-01-01T14:02:00Z
            """))
    return csv_filename


def test_csv_default(csv_det_filename):
    # run test with:
    #   - 'metadata_fields' for 'CSVDetectionReader' == default
    #   - copy all metadata items
    csv_reader = CSVDetectionReader(csv_det_filename.strpath, ["x", "y"], "t")
    detections = [
        detection
        for _, detections in csv_reader
        for detection in detections]

    for n, detection in enumerate(detections):
        assert np.array_equal(
            detection.state_vector, np.array([[10 + n], [20 + n]]))
        assert detection.timestamp.hour == 14
        assert detection.timestamp.minute == n
        assert detection.timestamp.date() == datetime.date(2018, 1, 1)
        assert isinstance(detection.metadata, dict)
        assert len(detection.metadata) == 2
        assert 'z' in detection.metadata.keys()
        assert 'identifier' in detection.metadata.keys()
        assert int(detection.metadata['z']) == 30 + n
        assert detection.metadata['identifier'] == '22018332'


def test_csv_metadata_time(csv_det_filename):
    # run test with:
    #   - 'metadata_fields' for 'CSVDetectionReader' contains
    #       'z' but not 'identifier'
    #   - 'time_field_format' is specified
    csv_reader = CSVDetectionReader(csv_det_filename.strpath, ["x", "y"], "t",
                                    time_field_format="%Y-%m-%dT%H:%M:%SZ",
                                    metadata_fields=["z"])
    detections = [
        detection
        for _, detections in csv_reader
        for detection in detections]

    for n, detection in enumerate(detections):
        assert np.array_equal(
            detection.state_vector, np.array([[10 + n], [20 + n]]))
        assert detection.timestamp.hour == 14
        assert detection.timestamp.minute == n
        assert detection.timestamp.date() == datetime.date(2018, 1, 1)
        assert isinstance(detection.metadata, dict)
        assert len(detection.metadata) == 1
        assert 'z' in detection.metadata.keys()
        assert int(detection.metadata['z']) == 30 + n


def test_csv_missing_metadata_timestamp(tmpdir):
    # run test with:
    #   - 'metadata_fields' for 'CSVDetectionReader' contains
    #       column names that do not exist in CSV file
    #   - 'time' field represented as a Unix epoch timestamp
    csv_det_filename = tmpdir.join("test.csv")
    with csv_det_filename.open('w') as csv_file:
        csv_file.write(dedent("""\
            x,y,z,identifier,t
            10,20,30,22018332,1514815200
            11,21,31,22018332,1514815260
            12,22,32,22018332,1514815320
            """))

    csv_reader = CSVDetectionReader(csv_det_filename.strpath, ["x", "y"], "t",
                                    metadata_fields=["heading"],
                                    timestamp=True)
    detections = [
        detection
        for _, detections in csv_reader
        for detection in detections]

    for n, detection in enumerate(detections):
        assert np.array_equal(
            detection.state_vector, np.array([[10 + n], [20 + n]]))
        assert detection.timestamp.hour == 14
        assert detection.timestamp.minute == n
        assert detection.timestamp.date() == datetime.date(2018, 1, 1)
        assert isinstance(detection.metadata, dict)
        assert len(detection.metadata) == 0


def test_csv_multi_per_timestep(tmpdir):
    csv_det_filename = tmpdir.join("test.csv")
    with csv_det_filename.open('w') as csv_file:
        csv_file.write(dedent("""\
                x,y,z,identifier,t
                10,20,30,22018332,2018-01-01T14:00:00Z
                11,21,31,22018332,2018-01-01T14:01:00Z
                12,22,32,22018332,2018-01-01T14:02:00Z
                13,23,33,32018332,2018-01-01T14:02:00Z
                14,24,34,32018332,2018-01-01T14:03:00Z
                """))

    csv_reader = CSVDetectionReader(csv_det_filename.strpath, ["x", "y"], "t")

    for time, detections in csv_reader:
        if time == datetime.datetime(2018, 1, 1, 14, 2):
            assert len(detections) == 2
        else:
            assert len(detections) == 1


def test_tsv(tmpdir):
    csv_filename = tmpdir.join("test.csv")
    with csv_filename.open('w') as csv_file:
        csv_file.write(dedent("""\
            x\ty\tz\tidentifier\tt
            10\t20\t30\t22018332\t2018-01-01T14:00:00Z
            11\t21\t31\t22018332\t2018-01-01T14:01:00Z
            12\t22\t32\t22018332\t2018-01-01T14:02:00Z
            """))

    # run test with:
    #   - 'metadata_fields' for 'CSVDetectionReader' == default
    #   - copy all metadata items
    csv_reader = CSVDetectionReader(csv_filename.strpath,
                                    ["x", "y"], "t",
                                    csv_options={'dialect': 'excel-tab'})
    detections = [
        detection
        for _, detections in csv_reader
        for detection in detections]

    for n, detection in enumerate(detections):
        assert np.array_equal(
            detection.state_vector, np.array([[10 + n], [20 + n]]))
        assert detection.timestamp.hour == 14
        assert detection.timestamp.minute == n
        assert detection.timestamp.date() == datetime.date(2018, 1, 1)
        assert isinstance(detection.metadata, dict)
        assert len(detection.metadata) == 2
        assert 'z' in detection.metadata.keys()
        assert 'identifier' in detection.metadata.keys()
        assert int(detection.metadata['z']) == 30 + n
        assert detection.metadata['identifier'] == '22018332'
