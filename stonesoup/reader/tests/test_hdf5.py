import datetime
from operator import attrgetter

import h5py
import numpy as np
import pytest

from ..hdf5 import HDF5DetectionReader, HDF5GroundTruthReader


@pytest.fixture()
def hdf5_gt_filename(tmpdir):
    hdf5_filename = tmpdir.join("test.hdf5")
    with h5py.File(hdf5_filename, "w") as hdf5_file:
        hdf5_file.create_dataset("state/x", data=[10, 11, 12, 13, 14])
        hdf5_file.create_dataset("state/y", data=[20, 21, 22, 23, 24])
        hdf5_file.create_dataset("state/z", data=[30, 31, 32, 33, 34])
        hdf5_file.create_dataset(
            "identifier",
            data=["22018332", "22018332", "22018332", "32018332", "32018332"],
            compression="gzip",
            compression_opts=1,
        )
        hdf5_file.create_dataset("meta/valid", data=[40, 41, 42, 43, 44])
        hdf5_file.create_dataset("meta/invalid", data=[1, 2, 3, 4])
        hdf5_file.create_dataset(
            "t",
            data=[
                "2018-01-01T14:00:00Z",
                "2018-01-01T14:01:00Z",
                "2018-01-01T14:02:00Z",
                "2018-01-01T14:03:00Z",
                "2018-01-01T14:04:00Z",
            ],
        )
    return hdf5_filename


def test_hdf5_gt_2d(hdf5_gt_filename):
    # run test with:
    #   - 2d co-ordinates
    #   - default time field format
    hdf5_reader = HDF5GroundTruthReader(
        hdf5_gt_filename.strpath,
        state_vector_fields=["state/x", "state/y"],
        time_field="t",
        path_id_field="identifier",
    )

    final_gt_paths = set()
    for _, gt_paths_at_timestep in hdf5_reader:
        final_gt_paths.update(gt_paths_at_timestep)
    assert len(final_gt_paths) == 2

    ground_truth_states = [
        gt_state for gt_path in final_gt_paths for gt_state in gt_path
    ]

    for n, gt_state in enumerate(
        sorted(ground_truth_states, key=attrgetter("timestamp"))
    ):
        assert np.array_equal(gt_state.state_vector, np.array([[10 + n], [20 + n]]))
        assert gt_state.timestamp.hour == 14
        assert gt_state.timestamp.minute == n
        assert gt_state.timestamp.date() == datetime.date(2018, 1, 1)

        assert isinstance(gt_state.metadata, dict)
        assert len(gt_state.metadata) == 3
        assert "identifier" in gt_state.metadata
        assert "state/z" in gt_state.metadata
        assert "meta/valid" in gt_state.metadata
        assert "meta/invalid" not in gt_state.metadata
        assert int(gt_state.metadata["meta/valid"]) == 40 + n


def test_hdf5_gt_3d_time(hdf5_gt_filename):
    # run test with:
    #   - 3d co-ordinates
    #   - time field format specified
    hdf5_reader = HDF5GroundTruthReader(
        hdf5_gt_filename.strpath,
        state_vector_fields=["state/x", "state/y", "state/z"],
        time_field="t",
        time_field_format="%Y-%m-%dT%H:%M:%SZ",
        path_id_field="identifier",
    )

    final_gt_paths = set()
    for _, gt_paths_at_timestep in hdf5_reader:
        final_gt_paths.update(gt_paths_at_timestep)
    assert len(final_gt_paths) == 2

    ground_truth_states = [
        gt_state for gt_path in final_gt_paths for gt_state in gt_path
    ]

    for n, gt_state in enumerate(
        sorted(ground_truth_states, key=attrgetter("timestamp"))
    ):
        assert np.array_equal(
            gt_state.state_vector, np.array([[10 + n], [20 + n], [30 + n]])
        )
        assert gt_state.timestamp.hour == 14
        assert gt_state.timestamp.minute == n
        assert gt_state.timestamp.date() == datetime.date(2018, 1, 1)

        assert isinstance(gt_state.metadata, dict)
        assert len(gt_state.metadata) == 2
        assert "identifier" in gt_state.metadata
        assert "meta/valid" in gt_state.metadata
        assert "meta/invalid" not in gt_state.metadata
        assert int(gt_state.metadata["meta/valid"]) == 40 + n


def test_hdf5_gt_multi_per_timestep(tmpdir):
    hdf5_gt_filename = tmpdir.join("test.hdf5")
    with h5py.File(hdf5_gt_filename, "w") as hdf5_file:
        hdf5_file.create_dataset("x", data=[10, 11, 12, 13, 14])
        hdf5_file.create_dataset("y", data=[20, 21, 22, 23, 24])
        hdf5_file.create_dataset("z", data=[30, 31, 32, 33, 34])
        hdf5_file.create_dataset(
            "identifier",
            data=[22018332, 22018332, 22018332, 32018332, 32018332],
            compression="gzip",
            compression_opts=1,
        )
        hdf5_file.create_dataset(
            "t",
            data=[
                "2018-01-01T14:00:00Z",
                "2018-01-01T14:01:00Z",
                "2018-01-01T14:02:00Z",
                "2018-01-01T14:02:00Z",
                "2018-01-01T14:03:00Z",
            ],
        )

    hdf5_reader = HDF5GroundTruthReader(
        hdf5_gt_filename.strpath,
        state_vector_fields=["x", "y"],
        time_field="t",
        path_id_field="identifier",
    )

    for time, ground_truth_paths in hdf5_reader:
        if time == datetime.datetime(2018, 1, 1, 14, 2):
            assert len(ground_truth_paths) == 2
        else:
            assert len(ground_truth_paths) == 1


@pytest.fixture()
def hdf5_det_filename(tmpdir):
    hdf5_filename = tmpdir.join("test.hdf5")

    with h5py.File(hdf5_filename, "w") as hdf5_file:
        hdf5_file.create_dataset("state/x", data=[10, 11, 12])
        hdf5_file.create_dataset("state/y", data=[20, 21, 22])
        hdf5_file.create_dataset("state/z", data=[30, 31, 32])
        hdf5_file.create_dataset(
            "identifier",
            data=["22018332", "22018332", "22018332"],
            compression="gzip",
            compression_opts=1,
        )
        hdf5_file.create_dataset("meta/valid", data=[40, 41, 42])
        hdf5_file.create_dataset("meta/invalid", data=[1, 2, 3, 4])
        hdf5_file.create_dataset(
            "t",
            data=[
                "2018-01-01T14:00:00Z",
                "2018-01-01T14:01:00Z",
                "2018-01-01T14:02:00Z",
            ],
        )
    return hdf5_filename


def test_hdf5_default(hdf5_det_filename):
    # run test with:
    #   - 'metadata_fields' for 'HDF5DetectionReader' == default
    #   - copy all metadata items
    hdf5_reader = HDF5DetectionReader(
        hdf5_det_filename.strpath, ["state/x", "state/y"], "t"
    )
    detections = [
        detection for _, detections in hdf5_reader for detection in detections
    ]

    for n, detection in enumerate(detections):
        assert np.array_equal(detection.state_vector, np.array([[10 + n], [20 + n]]))
        assert detection.timestamp.hour == 14
        assert detection.timestamp.minute == n
        assert detection.timestamp.date() == datetime.date(2018, 1, 1)
        assert isinstance(detection.metadata, dict)
        assert len(detection.metadata) == 3
        assert "state/z" in detection.metadata
        assert "identifier" in detection.metadata
        assert "meta/valid" in detection.metadata
        assert int(detection.metadata["state/z"]) == 30 + n
        assert detection.metadata["identifier"] == "22018332"
        assert int(detection.metadata["meta/valid"]) == 40 + n


def test_hdf5_metadata_time(hdf5_det_filename):
    # run test with:
    #   - 'metadata_fields' for 'HDF5DetectionReader' contains
    #       'z' but not 'identifier'
    #   - 'time_field_format' is specified
    hdf5_reader = HDF5DetectionReader(
        hdf5_det_filename.strpath,
        ["state/x", "state/y"],
        "t",
        time_field_format="%Y-%m-%dT%H:%M:%SZ",
        metadata_fields=["state/z"],
    )
    detections = [
        detection for _, detections in hdf5_reader for detection in detections
    ]

    for n, detection in enumerate(detections):
        assert np.array_equal(detection.state_vector, np.array([[10 + n], [20 + n]]))
        assert detection.timestamp.hour == 14
        assert detection.timestamp.minute == n
        assert detection.timestamp.date() == datetime.date(2018, 1, 1)
        assert isinstance(detection.metadata, dict)
        assert len(detection.metadata) == 1
        assert "state/z" in detection.metadata.keys()
        assert int(detection.metadata["state/z"]) == 30 + n


def test_hdf5_missing_metadata_timestamp(tmpdir):
    # run test with:
    #   - 'metadata_fields' for 'HDF5DetectionReader' contains
    #       field names that do not exist in hdf5 file
    #   - 'time' field represented as a Unix epoch timestamp
    hdf5_filename = tmpdir.join("test.hdf5")
    with h5py.File(hdf5_filename, "w") as hdf5_file:
        hdf5_file.create_dataset("state/x", data=[10, 11, 12])
        hdf5_file.create_dataset("state/y", data=[20, 21, 22])
        hdf5_file.create_dataset("state/z", data=[30, 31, 32])
        hdf5_file.create_dataset(
            "identifier",
            data=["22018332", "22018332", "22018332"],
            compression="gzip",
            compression_opts=1,
        )
        hdf5_file.create_dataset("t", data=[1514815200, 1514815260, 1514815320])

    hdf5_reader = HDF5DetectionReader(
        hdf5_filename.strpath,
        ["state/x", "state/y"],
        "t",
        metadata_fields=["heading"],
        timestamp=True,
    )
    detections = [
        detection for _, detections in hdf5_reader for detection in detections
    ]

    for n, detection in enumerate(detections):
        assert np.array_equal(detection.state_vector, np.array([[10 + n], [20 + n]]))
        assert detection.timestamp.hour == 14
        assert detection.timestamp.minute == n
        assert detection.timestamp.date() == datetime.date(2018, 1, 1)
        assert isinstance(detection.metadata, dict)
        assert len(detection.metadata) == 0


def test_hdf5_multi_per_timestep(tmpdir):
    hdf5_filename = tmpdir.join("test.hdf5")
    with h5py.File(hdf5_filename, "w") as hdf5_file:
        hdf5_file.create_dataset("state/x", data=[10, 11, 12, 13, 14])
        hdf5_file.create_dataset("state/y", data=[20, 21, 22, 23, 24])
        hdf5_file.create_dataset("state/z", data=[30, 31, 32, 33, 34])
        hdf5_file.create_dataset(
            "identifier",
            data=["22018332", "22018332", "22018332", "32018332", "32018332"],
            compression="gzip",
            compression_opts=1,
        )
        hdf5_file.create_dataset(
            "t",
            data=[
                "2018-01-01T14:00:00Z",
                "2018-01-01T14:01:00Z",
                "2018-01-01T14:02:00Z",
                "2018-01-01T14:02:00Z",
                "2018-01-01T14:03:00Z",
            ],
        )

    hdf5_reader = HDF5DetectionReader(
        hdf5_filename.strpath, ["state/x", "state/y"], "t"
    )

    for time, detections in hdf5_reader:
        if time == datetime.datetime(2018, 1, 1, 14, 2):
            assert len(detections) == 2
        else:
            assert len(detections) == 1


def test_hdf5_time_resolution(tmpdir):
    hdf5_filename = tmpdir.join("test.hdf5")
    with h5py.File(hdf5_filename, "w") as hdf5_file:
        hdf5_file.create_dataset("state/x", data=[10, 11, 12, 13, 14, 15])
        hdf5_file.create_dataset("state/y", data=[20, 21, 22, 23, 24, 25])
        hdf5_file.create_dataset(
            "t",
            data=[
                "2018-01-01T14:00:0.01Z",
                "2018-01-01T14:00:0.05Z",
                "2018-01-01T14:00:0.15Z",
                "2018-01-01T14:01:17Z",
                "2018-01-01T14:01:22Z",
                "2018-01-01T14:01:27Z",
            ],
        )

    hdf5_reader = HDF5DetectionReader(
        hdf5_filename.strpath, ["state/x", "state/y"], "t", time_res_micro=10000
    )

    for time, detections in hdf5_reader:
        if time == datetime.datetime(2018, 1, 1, 14, 0, 0):
            assert len(detections) == 2
        elif time == datetime.datetime(2018, 1, 1, 14, 0, 0, 10000):
            assert len(detections) == 1
        elif time == datetime.datetime(2018, 1, 1, 14, 1, 17):
            assert len(detections) == 1
        elif time == datetime.datetime(2018, 1, 1, 14, 1, 22):
            assert len(detections) == 1
        elif time == datetime.datetime(2018, 1, 1, 14, 1, 27):
            assert len(detections) == 1

    hdf5_reader = HDF5DetectionReader(
        hdf5_filename.strpath, ["state/x", "state/y"], "t", time_res_second=10
    )
    for time, detections in hdf5_reader:
        if time == datetime.datetime(2018, 1, 1, 14, 0):
            assert len(detections) == 3
        if time == datetime.datetime(2018, 1, 1, 14, 1, 10):
            assert len(detections) == 1
        elif time == datetime.datetime(2018, 1, 1, 14, 1, 20):
            assert len(detections) == 2
