"""GOSPA/OSPA tests."""
import datetime

import numpy as np
import pytest

from ..manager import SimpleManager
from ..ospametric import GOSPAMetric, OSPAMetric
from ...types.detection import Detection
from ...types.groundtruth import GroundTruthPath, GroundTruthState
from ...types.state import State
from ...types.track import Track


def test_gospametric_extractstates():
    """Test GOSPA extract states."""
    generator = GOSPAMetric(
        c=10.0,
        p=1)
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


@pytest.mark.parametrize('num_states', (2, 5))
def test_gospametric_compute_assignments(num_states):
    """Test GOSPA assignment algorithm."""
    generator = GOSPAMetric(
        c=10.0,
        p=1)
    time_now = datetime.datetime.now()
    track_obj = Track([State(state_vector=[[i]], timestamp=time_now)
                      for i in range(num_states)])
    truth_obj = GroundTruthPath([State(state_vector=[[i]], timestamp=time_now)
                                for i in range(num_states)])
    cost_matrix = generator.compute_cost_matrix(track_obj.states,
                                                truth_obj.states)
    neg_cost_matrix = -1.*cost_matrix
    meas_to_truth, truth_to_meas, opt_cost =\
        generator.compute_assignments(neg_cost_matrix,
                                      10 * num_states * num_states)

    assert opt_cost == 0.0
    assert np.array_equal(meas_to_truth, truth_to_meas)
    assert np.array_equal(meas_to_truth,
                          np.array([i for i in range(num_states)]))

    # Missing 1 track
    cost_matrix = generator.compute_cost_matrix(track_obj.states[:-1],
                                                truth_obj.states)
    neg_cost_matrix = -1.*cost_matrix
    meas_to_truth, truth_to_meas, opt_cost = \
        generator.compute_assignments(neg_cost_matrix,
                                      10 * num_states * num_states)

    assert opt_cost == 0.0
    assert np.array_equal(meas_to_truth, truth_to_meas[:-1])
    assert truth_to_meas[-1] == -1
    assert np.array_equal(meas_to_truth,
                          np.array([i for i in range(num_states - 1)]))

    # Missing 1 truth
    cost_matrix = generator.compute_cost_matrix(track_obj.states,
                                                truth_obj.states[:-1])
    neg_cost_matrix = -1.*cost_matrix
    meas_to_truth, truth_to_meas, opt_cost = \
        generator.compute_assignments(neg_cost_matrix,
                                      10 * num_states * num_states)

    assert opt_cost == 0.0
    assert np.array_equal(meas_to_truth[:-1], truth_to_meas)
    assert meas_to_truth[-1] == -1
    assert np.array_equal(meas_to_truth[:-1],
                          np.array([i for i in range(num_states - 1)]))


def test_gospametric_cost_matrix():
    """Test GOSPA cost matrix. Also indirectly checks compute distance."""
    num_states = 5
    generator = GOSPAMetric(
        c=10.0,
        p=1)
    time_now = datetime.datetime.now()
    track_obj = Track([State(state_vector=[[i]], timestamp=time_now)
                      for i in range(num_states)])
    truth_obj = GroundTruthPath([State(state_vector=[[i]], timestamp=time_now)
                                for i in range(num_states)])
    cost_matrix = generator.compute_cost_matrix(track_obj.states,
                                                truth_obj.states)

    tmp_vec = np.arange(num_states)
    tmp_mat = np.zeros([num_states, num_states])
    for n in range(num_states):
        tmp_mat[n, :] = np.roll(tmp_vec, n)

    tmp_upper = np.triu(tmp_mat)
    test_matrix = tmp_upper + tmp_upper.transpose()
    assert np.array_equal(cost_matrix, test_matrix)


def test_gospametric_compute_gospa_metric():
    """Test compute GOSPA metric."""
    num_states = 5
    generator = GOSPAMetric(
        c=10.0,
        p=1)
    time_now = datetime.datetime.now()
    track_obj = Track([State(state_vector=[[i]], timestamp=time_now)
                      for i in range(num_states)])
    truth_obj = GroundTruthPath([State(state_vector=[[i]], timestamp=time_now)
                                for i in range(num_states)])
    single_time_metric, assignment_matrix =\
        generator.compute_gospa_metric(track_obj.states,
                                       truth_obj.states)
    gospa_metric = single_time_metric.value
    assert (gospa_metric['distance'] == 0.0)
    assert (gospa_metric['localisation'] == 0.0)
    assert (gospa_metric['missed'] == 0.0)
    assert (gospa_metric['false'] == 0.0)


def test_gospametric_computemetric():
    """Test GOSPA compute metric."""
    generator = GOSPAMetric(
        c=10.0,
        p=1)
    time = datetime.datetime.now()
    # Multiple tracks and truths present at two timesteps
    tracks = {Track(states=[State(state_vector=[[i + 0.5]], timestamp=time),
                            State(state_vector=[[i + 1]],
                                  timestamp=time + datetime.timedelta(
                                  seconds=1))])
              for i in range(5)}
    truths = {GroundTruthPath(
        states=[State(state_vector=[[i]], timestamp=time),
                GroundTruthState(state_vector=[[i]],
                                 timestamp=time + datetime.timedelta(
                                     seconds=1))])
              for i in range(5)}

    manager = SimpleManager([generator])
    manager.add_data(truths, tracks)
    main_metric = generator.compute_metric(manager)

    assert main_metric.title == "GOSPA Metrics"
    assert main_metric.time_range.start_timestamp == time
    assert main_metric.time_range.end_timestamp == time + datetime.timedelta(
        seconds=1)
    first_association = [i for i in main_metric.value
                         if i.timestamp == time][0]
    assert first_association.title == "GOSPA Metric"
    assert first_association.timestamp == time
    assert first_association.generator == generator
    # In the following, distance is divided by the cardinality of
    # of the set since GOSPA is not normalized.
    assert first_association.value['distance'] / 5. == 0.5
    second_association = [
        i for i in main_metric.value if
        i.timestamp == time + datetime.timedelta(seconds=1)][0]
    assert second_association.title == "GOSPA Metric"
    assert second_association.timestamp == time + datetime.timedelta(seconds=1)
    assert second_association.generator == generator
    assert second_association.value['distance'] / 5. == 1


def test_ospametric_extractstates():
    """Test OSPA metric extract states."""
    generator = OSPAMetric(
        c=10,
        p=1)

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


def test_ospametric_computecostmatrix():
    """Test OSPA metric compute cost matrix."""
    generator = OSPAMetric(
        c=10,
        p=1)

    time = datetime.datetime.now()
    track = Track(states=[
        State(state_vector=[[i]], timestamp=time)
        for i in range(5)])
    truth = GroundTruthPath(states=[
        State(state_vector=[[i]], timestamp=time)
        for i in range(5)])

    cost_matrix = generator.compute_cost_matrix(track.states, truth.states)

    assert np.array_equal(cost_matrix, np.array([[0., 1., 2., 3., 4.],
                                                 [1., 0., 1., 2., 3.],
                                                 [2., 1., 0., 1., 2.],
                                                 [3., 2., 1., 0., 1.],
                                                 [4., 3., 2., 1., 0.]]))

    cost_matrix = generator.compute_cost_matrix(track.states, truth.states[:-1])
    assert np.array_equal(cost_matrix, np.array([[0., 1., 2., 3.],
                                                 [1., 0., 1., 2.],
                                                 [2., 1., 0., 1.],
                                                 [3., 2., 1., 0.],
                                                 [4., 3., 2., 1.]]))

    # One more track than truths
    cost_matrix = generator.compute_cost_matrix(track.states, truth.states[:-1], complete=True)
    assert np.array_equal(cost_matrix, np.array([[0., 1., 2., 3., 10.],
                                                 [1., 0., 1., 2., 10.],
                                                 [2., 1., 0., 1., 10.],
                                                 [3., 2., 1., 0., 10.],
                                                 [4., 3., 2., 1., 10.]]))


def test_ospametric_computeospadistance():
    """Test OSPA metric compute OSPA distance."""
    generator = OSPAMetric(
        c=10,
        p=1)

    time = datetime.datetime.now()
    track = Track(states=[
        State(state_vector=[[i]], timestamp=time)
        for i in range(5)])
    truth = GroundTruthPath(states=[
        State(state_vector=[[i + 0.5]], timestamp=time)
        for i in range(5)])

    metric = generator.compute_OSPA_distance(track.states, truth.states)

    assert metric.title == "OSPA distance"
    assert metric.value == 0.5
    assert metric.timestamp == time
    assert metric.generator == generator


@pytest.mark.parametrize('p', (1, 2, np.inf), ids=('p=1', 'p=2', 'p=inf'))
def test_ospametric_computemetric(p):
    """Test OSPA compute metric."""
    generator = OSPAMetric(
        c=10,
        p=p)

    time = datetime.datetime.now()
    # Multiple tracks and truths present at two timesteps
    tracks = {Track(states=[State(state_vector=[[i + 0.5]], timestamp=time),
                            State(state_vector=[[i + 1.2]],
                                  timestamp=time + datetime.timedelta(
                                     seconds=1))])
              for i in range(5)}
    truths = {GroundTruthPath(
        states=[GroundTruthState(state_vector=[[i]], timestamp=time),
                GroundTruthState(state_vector=[[i + 1]],
                                 timestamp=time+datetime.timedelta(
                                     seconds=1))])
              for i in range(5)}

    manager = SimpleManager([generator])
    manager.add_data(truths, tracks)
    main_metric = generator.compute_metric(manager)
    first_association, second_association = main_metric.value

    assert main_metric.title == "OSPA distances"
    assert main_metric.time_range.start_timestamp == time
    assert main_metric.time_range.end_timestamp == time + datetime.timedelta(
        seconds=1)

    assert first_association.title == "OSPA distance"
    assert first_association.value == pytest.approx(0.5)
    assert first_association.timestamp == time
    assert first_association.generator == generator

    assert second_association.title == "OSPA distance"
    assert second_association.value == pytest.approx(0.2)
    assert second_association.timestamp == time + datetime.timedelta(seconds=1)
    assert second_association.generator == generator


@pytest.mark.parametrize(
    'p,first_value,second_value',
    ((1, 2.4, 2.16), (2, 4.49444, 4.47571), (np.inf, 10, 10)),
    ids=('p=1', 'p=2', 'p=inf'))
def test_ospa_computemetric_cardinality_error(p, first_value, second_value):
    generator = OSPAMetric(
        c=10,
        p=p)

    time = datetime.datetime.now()
    # Multiple tracks and truths present at two timesteps
    tracks = {Track(states=[State(state_vector=[[i + 0.5]], timestamp=time),
                            State(state_vector=[[i + 1.2]],
                                  timestamp=time + datetime.timedelta(seconds=1))])
              for i in range(4)}
    truths = {GroundTruthPath(
        states=[GroundTruthState(state_vector=[[i]], timestamp=time),
                GroundTruthState(state_vector=[[i + 1]],
                                 timestamp=time+datetime.timedelta(seconds=1))])
              for i in range(5)}

    manager = SimpleManager([generator])
    manager.add_data(truths, tracks)
    main_metric = generator.compute_metric(manager)
    first_association, second_association = main_metric.value

    assert main_metric.title == "OSPA distances"
    assert main_metric.time_range.start_timestamp == time
    assert main_metric.time_range.end_timestamp == time + datetime.timedelta(
        seconds=1)

    assert first_association.title == "OSPA distance"
    assert first_association.value == pytest.approx(first_value)
    assert first_association.timestamp == time
    assert first_association.generator == generator

    assert second_association.title == "OSPA distance"
    assert second_association.value == pytest.approx(second_value)
    assert second_association.timestamp == time + datetime.timedelta(seconds=1)
    assert second_association.generator == generator
