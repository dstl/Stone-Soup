import datetime
import numpy as np
import pytest

from io import StringIO
from operator import attrgetter

from ..pandas_reader import DataFrameGroundTruthReader, DataFrameDetectionReader

try:
    import pandas as pd
except ImportError as error:
    raise ImportError(
        "Usage of Pandas Readers readers requires the dependency 'pandas' being installed. ") from error


# generate dummy pandas dataframe for testing purposes
test_dataframe =  pd.read_table(
                    StringIO("""x,y,z,identifier,t
                    10,20,30,22018332,2018-01-01T14:00:00Z
                    11,21,31,22018332,2018-01-01T14:01:00Z
                    12,22,32,22018332,2018-01-01T14:02:00Z
                    13,23,33,32018332,2018-01-01T14:03:00Z
                    14,24,34,32018332,2018-01-01T14:04:00Z
                    """), 
                    sep=',', parse_dates=['t'])


@pytest.mark.parametrize(
    'reader_type',
    (DataFrameGroundTruthReader, DataFrameDetectionReader))
def test_csv_gt_2d(reader_type):
    # run test with:
    #   - 2d co-ordinates
    #   - default time field format
    df_reader = reader_type(dataframe=test_dataframe,
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


@pytest.mark.parametrize(
    'reader_type',
    (DataFrameGroundTruthReader, DataFrameDetectionReader))
def test_csv_gt_3d_time(reader_type):
    # run test with:
    #   - 3d co-ordinates
    df_reader = reader_type(
                    dataframe=test_dataframe,
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


@pytest.mark.parametrize(
    'reader_type',
    (DataFrameGroundTruthReader, DataFrameDetectionReader))
def test_csv_gt_multi_per_timestep(reader_type):
    # test case with multiple entries per timestep
    test_df =  pd.read_table(
                StringIO("""x,y,z,identifier,t
                10,20,30,22018332,2018-01-01T14:00:00Z
                11,21,31,22018332,2018-01-01T14:01:00Z
                12,22,32,22018332,2018-01-01T14:02:00Z
                13,23,33,32018332,2018-01-01T14:02:00Z
                14,24,34,32018332,2018-01-01T14:03:00Z
                """), 
                    sep=',', parse_dates=['t'])

    df_reader = reader_type(
                        dataframe=test_df,
                        state_vector_fields=["x", "y"],
                        time_field="t",
                        path_id_field="identifier")

    for time, ground_truth_paths in df_reader:
        # remove timestamp from pd timestamp before comparing
        if time.tz_convert(None) == datetime.datetime(2018, 1, 1, 14, 2):
            assert len(ground_truth_paths) == 2
        else:
            assert len(ground_truth_paths) == 1
