import datetime

import numpy as np
import matplotlib.figure
import pytest

from ..plotter import TwoDPlotter
from ...types.track import Track
from ...types.groundtruth import GroundTruthPath
from ...types.state import State
from ...types.detection import Detection


def test_twodplotter():
    plotter = TwoDPlotter(track_indices=[0, 1],
                          detection_indices=[0, 1],
                          gtruth_indices=[0, 1])
    timestamp1 = datetime.datetime.now()
    timestamp2 = timestamp1 + datetime.timedelta(seconds=10)

    tracks = {Track(states=[State(np.array([[1], [2]]),
                                  timestamp=timestamp1 + datetime.timedelta(
                                      seconds=i)) for i in range(11)])}
    truths = {GroundTruthPath(states=[State(np.array([[1], [2]]),
                                            timestamp=timestamp1 +
                                            datetime.timedelta(seconds=i))
                                      for i in range(11)])}
    dets = {Detection(np.array([[1], [2]]),
                      timestamp=timestamp1+datetime.timedelta(seconds=i))
            for i in range(11)}

    # Expect warning, as models not provided.
    with pytest.warns(UserWarning,
                      match="Measurement model type not specified for all detections"):
        metric = plotter.plot_tracks_truth_detections(tracks, truths, dets)

    assert metric.title == "Track plot"
    assert metric.generator == plotter
    assert type(metric.value) == matplotlib.figure.Figure
    assert metric.time_range.start_timestamp == timestamp1
    assert metric.time_range.end_timestamp == timestamp2
