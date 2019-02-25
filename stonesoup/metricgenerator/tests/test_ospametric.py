import datetime

import numpy as np

from ..manager import SimpleManager
from ..ospametric import OSPAMetric
from ...models.measurement.linear import LinearGaussian
from ...types.detection import Detection
from ...types.groundtruth import GroundTruthPath, GroundTruthState
from ...types.state import State
from ...types.track import Track


def test_ospametric_extractstates():
    generator = OSPAMetric(
        c=10,
        p=1,
        measurement_model_truth=LinearGaussian(1, [0], None),
        measurement_model_track=LinearGaussian(1, [0], None))

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
    generator = OSPAMetric(
        c=10,
        p=1,
        measurement_model_truth=LinearGaussian(1, [0], None),
        measurement_model_track=LinearGaussian(1, [0], None))

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


def test_ospametric_computeospadistance():
    generator = OSPAMetric(
        c=10,
        p=1,
        measurement_model_truth=LinearGaussian(1, [0], None),
        measurement_model_track=LinearGaussian(1, [0], None))

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


def test_ospametric_computemetric():
    generator = OSPAMetric(
        c=10,
        p=1,
        measurement_model_truth=LinearGaussian(1, [0], None),
        measurement_model_track=LinearGaussian(1, [0], None))

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
                                 timestamp=time+datetime.timedelta(
                                     seconds=1))])
              for i in range(5)}

    manager = SimpleManager([generator])
    manager.add_data([tracks, truths])
    main_metric = generator.compute_metric(manager)

    assert main_metric.title == "OSPA distances"
    assert main_metric.time_range.start_timestamp == time
    assert main_metric.time_range.end_timestamp == time + datetime.timedelta(
        seconds=1)
    first_association = [i for i in main_metric.value
                         if i.timestamp == time][0]
    assert first_association.title == "OSPA distance"
    assert first_association.value == 0.5
    assert first_association.timestamp == time
    assert first_association.generator == generator
    second_association = [i for i in main_metric.value if
                          i.timestamp ==
                          time + datetime.timedelta(seconds=1)][
        0]
    assert second_association.title == "OSPA distance"
    assert second_association.value == 1
    assert second_association.timestamp == time + datetime.timedelta(seconds=1)
    assert second_association.generator == generator
