"""Uncertainty metric tests."""
import datetime

import numpy as np
import pytest

from ..manager import SimpleManager
from ..uncertaintymetric import SumofCovarianceNormsMetric
from ...types.detection import Detection
from ...types.groundtruth import GroundTruthPath, GroundTruthState
from ...types.state import State, GaussianState
from ...types.track import Track


def test_uncertaintymetric_extractstates():
    """Test uncertainty metric extract states."""
    generator = SumofCovarianceNormsMetric()

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
    with pytest.raises(ValueError):
        generator.extract_states([1, 2, 3])


def test_uncertaintymetric_compute_sum_covariancenorms():
    """Test uncertainty metric compute uncertainty."""
    generator = SumofCovarianceNormsMetric()

    time = datetime.datetime.now()
    track = Track(states=[GaussianState(state_vector=[[1], [2], [1], [2]],
                                        timestamp=time,
                                        covar=np.diag([i, i, i, i]))
                          for i in range(5)])

    metric = generator.compute_sum_covariancenorms(track.states)

    assert metric.title == "Covariance Matrix Norm Sum"
    assert metric.value == 20
    assert metric.timestamp == time
    assert metric.generator == generator
    with pytest.raises(ValueError,
                       match="All states must be from the same time to compute total uncertainty"):
        generator.compute_sum_covariancenorms([
            GaussianState(state_vector=[[1], [2], [1], [2]],
                          timestamp=time,
                          covar=np.diag([0, 0, 0, 0])),
            GaussianState(state_vector=[[1], [2], [1], [2]],
                          timestamp=time+datetime.timedelta(seconds=1),
                          covar=np.diag([0, 0, 0, 0]))])


def test_uncertaintymetric_compute_metric():
    """Test uncertainty compute metric."""
    generator = SumofCovarianceNormsMetric()

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
                                                           seconds=1))])}

    manager = SimpleManager([generator])
    manager.add_data(truths, tracks)
    main_metric = generator.compute_metric(manager)
    first_association, second_association = main_metric.value

    assert main_metric.title == "Sum of Covariance Norms Metric"
    assert main_metric.time_range.start_timestamp == time
    assert main_metric.time_range.end_timestamp == time + datetime.timedelta(
        seconds=1)

    assert first_association.title == "Covariance Matrix Norm Sum"
    assert first_association.value == 20
    assert first_association.timestamp == time
    assert first_association.generator == generator

    assert second_association.title == "Covariance Matrix Norm Sum"
    assert second_association.value == 25
    assert second_association.timestamp == time + datetime.timedelta(seconds=1)
    assert second_association.generator == generator
