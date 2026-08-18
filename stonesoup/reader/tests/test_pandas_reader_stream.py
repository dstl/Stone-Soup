import datetime

import numpy as np
import pytest

pytest.importorskip("pandas")
import pandas as pd  # noqa: E402

from ..pandas_reader import DataFrameDetectionReader  # noqa: E402


def test_detection_reader_accepts_dataframe_generator():
    first = pd.DataFrame({
        "x": [10, 11],
        "y": [20, 21],
        "t": [
            datetime.datetime(2026, 1, 1, 12, 0),
            datetime.datetime(2026, 1, 1, 12, 1),
        ],
    })
    second = pd.DataFrame({
        "x": [12, 13],
        "y": [22, 23],
        "t": [
            datetime.datetime(2026, 1, 1, 12, 1),
            datetime.datetime(2026, 1, 1, 12, 2),
        ],
    })

    def dataframe_generator():
        yield first
        yield second

    reader = DataFrameDetectionReader(
        dataframe=dataframe_generator(),
        state_vector_fields=["x", "y"],
        time_field="t",
    )

    steps = list(reader)

    assert [timestamp for timestamp, _ in steps] == [
        datetime.datetime(2026, 1, 1, 12, 0),
        datetime.datetime(2026, 1, 1, 12, 1),
        datetime.datetime(2026, 1, 1, 12, 2),
    ]
    assert [len(detections) for _, detections in steps] == [1, 2, 1]

    state_vectors = sorted(
        (tuple(np.asarray(detection.state_vector).ravel())
         for _, detections in steps for detection in detections)
    )
    assert state_vectors == [(10, 20), (11, 21), (12, 22), (13, 23)]


def test_detection_reader_rejects_non_dataframe_chunk():
    valid = pd.DataFrame({
        "x": [10],
        "y": [20],
        "t": [datetime.datetime(2026, 1, 1, 12, 0)],
    })

    reader = DataFrameDetectionReader(
        dataframe=iter((valid, {"x": [11]})),
        state_vector_fields=["x", "y"],
        time_field="t",
    )

    with pytest.raises(TypeError, match="iterable yielding DataFrames"):
        list(reader)
