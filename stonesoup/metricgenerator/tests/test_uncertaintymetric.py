"""Uncertainty metric tests."""
import datetime

import numpy as np

from ..manager import SimpleManager
from ..uncertaintymetric import UncertaintyMetric
from ...types.detection import Detection
from ...types.groundtruth import GroundTruthPath, GroundTruthState
from ...types.state import State, GaussianState
from ...types.track import Track


def test_uncertaintymetric_extractstates():
    """Test uncertainty metric extract states."""
    generator = UncertaintyMetric()

    # Test state extraction
    time_start = datetime.datetime.now()
    detections = [Detection(state_vector=np.array([[i]]), timestamp=time_start)
                  for i in range(5)]
    tracks = {Track(states=[State(state_vector=[[i]],
                                  timestamp=time_start)]) for i in range(5)}
    truths = {GroundTruthPath(states=[GroundTruthState(state_vector=[[i]],
                                                       timestamp=time_start)])
              for i in range(5)}

    det_states = generator.extract_states(detections)
    assert det_states.states == detections
    track_states = generator.extract_states(tracks)
    assert set(track_states) == set(t.states[0] for t in tracks)
    truth_states = generator.extract_states(truths)
    assert set(truth_states) == set(t.states[0] for t in truths)


def test_uncertaintymetric_compute_uncertainty():
    """Test uncertainty metric compute uncertainty."""
    generator = UncertaintyMetric()

    time = datetime.datetime.now()
    track = Track(states=[GaussianState(state_vector=[[1], [2], [1], [2]],
                                        timestamp=time,
                                        covar=np.diag([i, i, i, i]))
                          for i in range(5)])

    metric = generator.compute_uncertainty(track.states)

    assert metric.title == "Uncertainty Sum"
    assert metric.value == 20
    assert metric.timestamp == time
    assert metric.generator == generator


def test_uncertaintymetric_computemetric():
    """Test uncertainty compute metric."""
    generator = UncertaintyMetric()

    time = datetime.datetime.now()

    # Multiple tracks and truths present at two timesteps
    tracks = {Track(states=[GaussianState(state_vector=[[1], [2], [1], [2]], timestamp=time,
                                          covar=np.diag([i, i, i, i])),
                            GaussianState(state_vector=[[1.5], [2.5], [1.5], [2.5]],
                                          timestamp=time + datetime.timedelta(seconds=1),
                                          covar=np.diag([i+0.5, i+0.5, i+0.5, i+0.5]))])
              for i in range(5)}
    truths = {GroundTruthPath(states=[GroundTruthState(state_vector=[[0], [1], [0], [1]],
                                                       timestamp=time),
                                      GroundTruthState(state_vector=[[0.5], [1.5], [0.5], [1.5]],
                                                       timestamp=time + datetime.timedelta(
                                                           seconds=1))])
              for i in range(5)}

    manager = SimpleManager([generator])
    manager.add_data(truths, tracks)
    main_metric = generator.compute_metric(manager)
    first_association, second_association = main_metric.value

    assert main_metric.title == "Uncertainty Metric"
    assert main_metric.time_range.start_timestamp == time
    assert main_metric.time_range.end_timestamp == time + datetime.timedelta(
        seconds=1)

    assert first_association.title == "Uncertainty Sum"
    assert first_association.value == 20
    assert first_association.timestamp == time
    assert first_association.generator == generator

    assert second_association.title == "Uncertainty Sum"
    assert second_association.value == 25
    assert second_association.timestamp == time + datetime.timedelta(seconds=1)
    assert second_association.generator == generator
