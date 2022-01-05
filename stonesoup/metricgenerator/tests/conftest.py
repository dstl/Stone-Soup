# -*- coding: utf-8 -*-
from datetime import datetime, timedelta

import numpy as np
import pytest

from ...metricgenerator.manager import SimpleManager
from ...types.association import TimeRangeAssociation, AssociationSet
from ...types.detection import Detection
from ...types.groundtruth import GroundTruthPath, GroundTruthState
from ...types.hypothesis import SingleDistanceHypothesis
from ...types.prediction import GaussianStatePrediction
from ...types.time import TimeRange
from ...types.track import Track
from ...types.update import GaussianStateUpdate
from ...types.array import CovarianceMatrix, StateVector
from ...models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity
from ...models.measurement.linear import LinearGaussian


@pytest.fixture()
def trial_timestamps():
    now = datetime.now()
    return [now + timedelta(seconds=i) for i in range(4)]


@pytest.fixture()
def trial_truths(trial_timestamps):
    return [
        GroundTruthPath([
            GroundTruthState(np.array([[0, 1, 0, 1]]), timestamp=trial_timestamps[0],
                             metadata={"colour": "red"}),
            GroundTruthState(np.array([[1, 1, 1, 1]]), timestamp=trial_timestamps[1],
                             metadata={"colour": "red"}),
            GroundTruthState(np.array([[2, 1, 2, 1]]), timestamp=trial_timestamps[2],
                             metadata={"colour": "red"}),
            GroundTruthState(np.array([[3, 1, 3, 1]]), timestamp=trial_timestamps[3],
                             metadata={"colour": "red"})
        ]),
        GroundTruthPath([
            GroundTruthState(np.array([[-2, 1, -2, 1]]), timestamp=trial_timestamps[0],
                             metadata={"colour": "green"}),
            GroundTruthState(np.array([[-1, 1, -1, 1]]), timestamp=trial_timestamps[1],
                             metadata={"colour": "green"}),
            GroundTruthState(np.array([[0, 1, 0, 1]]), timestamp=trial_timestamps[2],
                             metadata={"colour": "green"}),
            GroundTruthState(np.array([[2, 1, 2, 1]]), timestamp=trial_timestamps[3],
                             metadata={"colour": "green"})
        ]),
        GroundTruthPath([
            GroundTruthState(np.array([[-1, 1, 1, 0]]), timestamp=trial_timestamps[0],
                             metadata={"colour": "blue"}),
            GroundTruthState(np.array([[0, 1, 1, 0]]), timestamp=trial_timestamps[1],
                             metadata={"colour": "blue"}),
            GroundTruthState(np.array([[1, 1, 2, 0]]), timestamp=trial_timestamps[2],
                             metadata={"colour": "blue"}),
            GroundTruthState(np.array([[3, 1, 3, 0]]), timestamp=trial_timestamps[3],
                             metadata={"colour": "blue"})
        ])
    ]


@pytest.fixture()
def trial_tracks(trial_truths, trial_timestamps):
    return [
        Track([
            GaussianStateUpdate(np.array([[0.1, 1.2, 0.1, 1.2]]),
                                np.eye(4),
                                SingleDistanceHypothesis(None,
                                                         Detection(
                                                             np.array([[np.pi / 4, 0]]),
                                                             metadata={"colour": "red"}),
                                                         distance=1
                                                         ),
                                timestamp=trial_timestamps[0]),
            GaussianStateUpdate(np.array([[1.1, 1.2, 1.1, 1.2]]),
                                np.eye(4),
                                SingleDistanceHypothesis(None,
                                                         Detection(
                                                             np.array([[np.pi / 4, np.sqrt(2)]]),
                                                             metadata={"colour": "blue"}),
                                                         distance=1
                                                         ),
                                timestamp=trial_timestamps[1]),
            GaussianStateUpdate(np.array([[2.1, 1.2, 2.1, 1.2]]),
                                np.eye(4),
                                SingleDistanceHypothesis(None,
                                                         Detection(
                                                             np.array(
                                                                 [[np.pi / 4, 2 * np.sqrt(2)]]),
                                                             metadata={"colour": "red"}),
                                                         distance=1
                                                         ),
                                timestamp=trial_timestamps[2]),
            GaussianStateUpdate(np.array([[3.1, 1.2, 3.1, 1.2]]),
                                np.eye(4),
                                SingleDistanceHypothesis(None,
                                                         Detection(
                                                             np.array(
                                                                 [[np.pi / 4, 3 * np.sqrt(2)]]),
                                                             metadata={"colour": "red"}),
                                                         distance=1
                                                         ),
                                timestamp=trial_timestamps[3])
        ]),
        Track([
            GaussianStateUpdate(np.array([[-2.5, 1.6, -2.5, 1.6]]),
                                np.eye(4),
                                SingleDistanceHypothesis(None,
                                                         Detection(
                                                             np.array(
                                                                 [[-3 * np.pi / 4,
                                                                   2 * np.sqrt(2)]]),
                                                             metadata={"colour": "red"}),
                                                         distance=1
                                                         ),
                                timestamp=trial_timestamps[0]),
            GaussianStatePrediction(np.array([[-1.5, 1.6, -1.5, 1.6]]),
                                    np.eye(4),
                                    timestamp=trial_timestamps[1]),
            GaussianStateUpdate(np.array([[0.5, 1.6, 0.5, 1.6]]),
                                np.eye(4),
                                SingleDistanceHypothesis(None,
                                                         Detection(
                                                             np.array(
                                                                 [[np.pi / 4, 0]]),
                                                             metadata={"colour": "green"}),
                                                         distance=1
                                                         ),
                                timestamp=trial_timestamps[2]),
            GaussianStateUpdate(np.array([[1.5, 1.6, 1.5, 1.6]]),
                                np.eye(4),
                                SingleDistanceHypothesis(None,
                                                         Detection(
                                                             np.array(
                                                                 [[np.pi / 4, np.sqrt(2)]]),
                                                             metadata={"colour": "green"}),
                                                         distance=1
                                                         ),
                                timestamp=trial_timestamps[3])
        ]),
        Track([
            GaussianStateUpdate(np.array([[-1.99, 1.99, 1.99, 1.99]]),
                                np.eye(4),
                                SingleDistanceHypothesis(None,
                                                         Detection(
                                                             np.array(
                                                                 [[3 * np.pi / 4, np.sqrt(2)]]),
                                                             metadata={}),
                                                         distance=1
                                                         ),
                                timestamp=trial_timestamps[0]),
            GaussianStateUpdate(np.array([[0.99, 1.99, 1.99, 1.99]]),
                                np.eye(4),
                                SingleDistanceHypothesis(None,
                                                         Detection(
                                                             np.array([[np.pi / 2, 1]]),
                                                             metadata={}),
                                                         distance=1
                                                         ),
                                timestamp=trial_timestamps[1]),
            GaussianStateUpdate(np.array([[0.999, 1.99, 1.999, 1.99]]),
                                np.eye(4),
                                SingleDistanceHypothesis(None,
                                                         Detection(
                                                             np.array(
                                                                 [[np.pi / 2, 1]]),
                                                             metadata={}),
                                                         distance=1.1
                                                         ),
                                timestamp=trial_timestamps[1]),
            GaussianStateUpdate(np.array([[1.99, 1.99, 1.99, 1.99]]),
                                np.eye(4),
                                SingleDistanceHypothesis(None,
                                                         Detection(
                                                             np.array(
                                                                 [[np.pi / 4, np.sqrt(2)]]),
                                                             metadata={"colour": "blue"}),
                                                         distance=1
                                                         ),
                                timestamp=trial_timestamps[2]),
            GaussianStateUpdate(np.array([[2.99, 1.99, 1.99, 1.99]]),
                                np.eye(4),
                                SingleDistanceHypothesis(None,
                                                         Detection(
                                                             np.array(
                                                                 [[np.pi / 4, np.sqrt(2)]]),
                                                             metadata={"colour": "green"}),
                                                         distance=1
                                                         ),
                                timestamp=trial_timestamps[3])
        ])
    ]


@pytest.fixture()
def trial_associations(trial_truths, trial_tracks, trial_timestamps):
    return AssociationSet({
        TimeRangeAssociation(objects={trial_truths[0], trial_tracks[0]},
                             time_range=TimeRange(trial_timestamps[0], trial_timestamps[2])),
        TimeRangeAssociation(objects={trial_truths[1], trial_tracks[1]},
                             time_range=TimeRange(trial_timestamps[0], trial_timestamps[1])),
        TimeRangeAssociation(objects={trial_truths[1], trial_tracks[1]},
                             time_range=TimeRange(trial_timestamps[2], trial_timestamps[3])),
        TimeRangeAssociation(objects={trial_truths[2], trial_tracks[2]},
                             time_range=TimeRange(trial_timestamps[1], trial_timestamps[2])),
        TimeRangeAssociation(objects={trial_truths[0], trial_tracks[2]},
                             time_range=TimeRange(trial_timestamps[1], trial_timestamps[3]))
    })


@pytest.fixture()
def trial_manager(trial_truths, trial_tracks, trial_associations):
    manager = SimpleManager()
    manager.add_data(trial_truths, trial_tracks)
    manager.association_set = trial_associations

    return manager


@pytest.fixture()
def transition_model():
    return CombinedLinearGaussianTransitionModel([ConstantVelocity(0.05), ConstantVelocity(0.05)])


@pytest.fixture()
def measurement_model():
    return LinearGaussian(ndim_state=4, mapping=[0, 2],
                          noise_covar=CovarianceMatrix(np.diag([5., 5.])))


@pytest.fixture()
def groundtruth():
    now = datetime.now()
    init_sv = StateVector([0., 1., 0., 1.])
    increment_sv = StateVector([1., 0., 1., 0])
    states = [GroundTruthState(init_sv + i*increment_sv, timestamp=now+timedelta(seconds=i))
              for i in range(21)]
    path = GroundTruthPath(states)
    return path
