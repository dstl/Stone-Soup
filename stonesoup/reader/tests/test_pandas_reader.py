import datetime
import numpy as np

from io import StringIO
from operator import attrgetter

from ..pandas_reader import DataFrameGroundTruthReader, DataFrameDetectionReader

try:
    import pandas as pd
except ImportError as error:
    raise ImportError(
        "Pandas Readers require dependency 'pandas' being installed. ") from error


# generate dummy pandas dataframe for ground-truth testing purposes
gt_test_df = pd.read_table(
                    StringIO("""x,y,z,identifier,t
                    10,20,30,22018332,2018-01-01T14:00:00Z
                    11,21,31,22018332,2018-01-01T14:01:00Z
                    12,22,32,22018332,2018-01-01T14:02:00Z
                    13,23,33,32018332,2018-01-01T14:03:00Z
                    14,24,34,32018332,2018-01-01T14:04:00Z
                    """),
                    sep=',', parse_dates=['t'])


def test_gt_df_2d():
    # run test with:
    #   - 2d co-ordinates
    #   - default time field format
    df_reader = DataFrameGroundTruthReader(dataframe=gt_test_df,
                                           state_vector_fields=["x", "y"],
                                           time_field="t",
                                           path_id_field="identifier")

    final_gt_paths = set()
    for _, gt_paths_at_timestep in df_reader:
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


def test_gt_df_3d():
    # run test with:
    #   - 3d co-ordinates
    #   - time field format specified
    df_reader = DataFrameGroundTruthReader(
                    dataframe=gt_test_df,
                    state_vector_fields=["x", "y", "z"],
                    time_field="t",
                    path_id_field="identifier")

    final_gt_paths = set()
    for _, gt_paths_at_timestep in df_reader:
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


def test_gt_df_3d_time():
    # run test with:
    #   - 3d co-ordinates
    #   - time field format specified
    # use dummy dataframe without parsing of input datetime
    test_df = pd.read_table(StringIO("""x,y,z,identifier,t
                    10,20,30,22018332,2018-01-01T14:00:00Z
                    11,21,31,22018332,2018-01-01T14:01:00Z
                    12,22,32,22018332,2018-01-01T14:02:00Z
                    13,23,33,32018332,2018-01-01T14:03:00Z
                    14,24,34,32018332,2018-01-01T14:04:00Z
                    """), sep=',')

    df_reader = DataFrameGroundTruthReader(
                                dataframe=test_df,
                                state_vector_fields=["x", "y", "z"],
                                time_field="t",
                                time_field_format="%Y-%m-%dT%H:%M:%SZ",
                                path_id_field="identifier")

    final_gt_paths = set()
    for _, gt_paths_at_timestep in df_reader:
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


def test_gt_df_3d_timestamp():
    # run test with:
    #   - time field represented as a Unix epoch timestamp
    # use dummy dataframe without parsing of input datetime
    test_df = pd.read_table(StringIO(
                    """x,y,z,identifier,t
                    10,20,30,22018332,1514815200
                    11,21,31,22018332,1514815260
                    12,22,32,22018332,1514815320
                    13,23,33,32018332,1514815380
                    14,24,34,32018332,1514815440
                    """), sep=',')

    df_reader = DataFrameGroundTruthReader(
                                dataframe=test_df,
                                state_vector_fields=["x", "y", "z"],
                                time_field="t",
                                timestamp=True,
                                path_id_field="identifier")

    final_gt_paths = set()
    for _, gt_paths_at_timestep in df_reader:
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


def test_gt_df_multi_per_timestep():
    # Run test with:
    #    - Multiple entries per timestep
    #    - Datetime column formatted as str (pandas object)
    test_df = pd.read_table(
                StringIO("""x,y,z,identifier,t
                10,20,30,22018332,2018-01-01T14:00:00Z
                11,21,31,22018332,2018-01-01T14:01:00Z
                12,22,32,22018332,2018-01-01T14:02:00Z
                13,23,33,32018332,2018-01-01T14:02:00Z
                14,24,34,32018332,2018-01-01T14:03:00Z
                """), sep=',')

    df_reader = DataFrameGroundTruthReader(
                            dataframe=test_df,
                            state_vector_fields=["x", "y"],
                            time_field="t",
                            path_id_field="identifier")

    for time, ground_truth_paths in df_reader:
        if time == datetime.datetime(2018, 1, 1, 14, 2):
            assert len(ground_truth_paths) == 2
        else:
            assert len(ground_truth_paths) == 1


def test_det_df():
    # generate dummy pandas dataframe for detection testing purposes
    det_test_df = pd.read_table(
                StringIO("""x,y,z,identifier,t
                10,20,30,22018332,2018-01-01T14:00:00Z
                11,21,31,22018332,2018-01-01T14:01:00Z
                12,22,32,22018332,2018-01-01T14:02:00Z
                """),
                sep=',', parse_dates=['t'])

    df_reader = DataFrameDetectionReader(
        dataframe=det_test_df,
        state_vector_fields=["x", "y"],
        time_field="t")

    detections = [detection for _, detections in df_reader
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
        assert str(detection.metadata['identifier']) == '22018332'


def test_det_df_metadata_time():
    # run test with:
    #   - 'time_field_format' is specified
    # get dummy dataframe - don't parse date column
    det_test_df = pd.read_table(
                StringIO("""x,y,z,identifier,t
                10,20,30,22018332,2018-01-01T14:00:00Z
                11,21,31,22018332,2018-01-01T14:01:00Z
                12,22,32,22018332,2018-01-01T14:02:00Z
                """), sep=',')

    df_reader = DataFrameDetectionReader(
            dataframe=det_test_df,
            state_vector_fields=["x", "y"],
            time_field="t",
            time_field_format="%Y-%m-%dT%H:%M:%SZ",
            metadata_fields=["z"])

    detections = [
        detection
        for _, detections in df_reader
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


def test_det_df_missing_metadata_timestamp():
    # run test with:
    #   - 'metadata_fields' for 'CSVDetectionReader' contains
    #       column names that do not exist in CSV file
    #   - 'time' field represented as a Unix epoch timestamp
    det_test_df = pd.read_table(
                StringIO("""x,y,z,identifier,t
                10,20,30,22018332,1514815200
                11,21,31,22018332,1514815260
                12,22,32,22018332,1514815320
                """), sep=',')

    df_reader = DataFrameDetectionReader(dataframe=det_test_df,
                                         state_vector_fields=["x", "y"],
                                         time_field="t",
                                         metadata_fields=["heading"],
                                         timestamp=True)
    detections = [
        detection
        for _, detections in df_reader
        for detection in detections]

    for n, detection in enumerate(detections):
        assert np.array_equal(
            detection.state_vector, np.array([[10 + n], [20 + n]]))
        assert detection.timestamp.hour == 14
        assert detection.timestamp.minute == n
        assert detection.timestamp.date() == datetime.date(2018, 1, 1)
        assert isinstance(detection.metadata, dict)
        assert len(detection.metadata) == 0


def test_detections_df_multi_per_timestep():
    # create test df example with multi-detections per timestep
    det_test_df = pd.read_table(
                    StringIO("""x,y,z,identifier,t
                    10,20,30,22018332,2018-01-01T14:00:00Z
                    11,21,31,22018332,2018-01-01T14:01:00Z
                    12,22,32,22018332,2018-01-01T14:02:00Z
                    13,23,33,32018332,2018-01-01T14:02:00Z
                    14,24,34,32018332,2018-01-01T14:03:00Z
                    """),
                    sep=',', parse_dates=['t'])

    # test detections with multiple entries per timestep
    df_reader = DataFrameDetectionReader(
        dataframe=det_test_df,
        state_vector_fields=["x", "y"],
        time_field="t")

    for time, detections in df_reader:
        # remove timestamp from pd timestamp before comparing
        if time.tz_convert(None) == datetime.datetime(2018, 1, 1, 14, 2):
            assert len(detections) == 2
        else:
            assert len(detections) == 1
