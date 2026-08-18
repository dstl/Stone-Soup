"""
Streaming pandas DataFrames
===========================

Stone Soup's pandas readers can consume an iterable of DataFrames as well as one DataFrame held
entirely in memory. This is useful when data arrives in batches, for example as CSV files added to
a directory over time.

The reader consumes the iterable lazily. Each yielded DataFrame must use the same schema expected
by the reader and the sequence of rows across all DataFrames must be in time order.

This example uses a finite temporary directory so it is deterministic when built by Sphinx-Gallery.
In a live application, ``dataframe_batches`` can be replaced by a directory watcher, message-queue
consumer, database cursor, or another generator that yields new DataFrames as data arrives.
"""

# %%
# Create two small CSV batches. The second batch deliberately starts at the same timestamp at which
# the first batch ends, demonstrating that detections are still grouped by timestamp across a
# DataFrame boundary.
import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from stonesoup.reader.pandas_reader import DataFrameDetectionReader


with TemporaryDirectory() as directory_name:
    directory = Path(directory_name)

    pd.DataFrame({
        "x": [10, 11],
        "y": [20, 21],
        "time": [
            datetime.datetime(2026, 1, 1, 12, 0),
            datetime.datetime(2026, 1, 1, 12, 1),
        ],
    }).to_csv(directory / "batch_001.csv", index=False)

    pd.DataFrame({
        "x": [12, 13],
        "y": [22, 23],
        "time": [
            datetime.datetime(2026, 1, 1, 12, 1),
            datetime.datetime(2026, 1, 1, 12, 2),
        ],
    }).to_csv(directory / "batch_002.csv", index=False)

    # %%
    # The generator yields one DataFrame at a time instead of concatenating the whole dataset.
    # Sorting the file names provides deterministic batch order for this example. A live watcher
    # should provide the same ordering guarantee itself.
    def dataframe_batches(path):
        for csv_path in sorted(path.glob("*.csv")):
            yield pd.read_csv(csv_path, parse_dates=["time"])

    reader = DataFrameDetectionReader(
        dataframe=dataframe_batches(directory),
        state_vector_fields=["x", "y"],
        time_field="time",
    )

    # %%
    # Iteration is unchanged from a reader constructed with one in-memory DataFrame.
    for timestamp, detections in reader:
        print(timestamp, len(detections))

# %%
# The output contains one detection at 12:00, two detections at 12:01 (one from each DataFrame),
# and one detection at 12:02. The important constraint is that the generator must yield batches in
# chronological order; the reader does not sort or buffer arbitrary out-of-order batches.
