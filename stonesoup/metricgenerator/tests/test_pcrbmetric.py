"""PCRB tests."""

import numpy as np
import pytest

from ..manager import MultiManager
from ..pcrbmetric import PCRBMetric
from ...types.groundtruth import GroundTruthState
from ...types.array import StateVector, StateVectors, CovarianceMatrix
from ...types.state import GaussianState


@pytest.mark.parametrize(
    'inverse_j, velocity_mapping',
    [(
            np.array([[5., 0., 0., 0.],
                      [0., 10., 0., 0.],
                      [0., 0., 5., 0.],
                      [0., 0., 0., 10.]]),
            [1, 3]
    )]
)
def test_compute_vel_rmse(inverse_j, velocity_mapping):
    vel_rmse = PCRBMetric._compute_vel_rmse(inverse_j, velocity_mapping)
    assert np.allclose(vel_rmse, np.sqrt(20))


@pytest.mark.parametrize(
    'inverse_j, position_mapping',
    [(
            np.array([[5., 0., 0., 0.],
                      [0., 10., 0., 0.],
                      [0., 0., 5., 0.],
                      [0., 0., 0., 10.]]),
            [0, 2]
    )]
)
def test_compute_pos_rmse(inverse_j, position_mapping):
    pos_rmse = PCRBMetric._compute_pos_rmse(inverse_j, position_mapping)
    assert np.allclose(pos_rmse, np.sqrt(10))


@pytest.mark.parametrize(
    'state, sensor_locations, irf',
    [(
            GroundTruthState(
                StateVector([[1.17722347], [1.17454005], [1.00067575], [0.98556448]])),
            StateVectors([StateVector([[-10], [10]])]),
            np.array([1.])
    )]
)
def test_calculate_j_z(state, sensor_locations, measurement_model, irf):
    overall_j_z = PCRBMetric._calculate_j_z(state, sensor_locations, measurement_model, irf)
    assert np.allclose(overall_j_z, np.array([[0.2, 0., 0., 0.],
                                              [0., 0., 0., 0.],
                                              [0., 0., 0.2, 0.],
                                              [0., 0., 0., 0.]]))


@pytest.mark.parametrize(
    'prior, sensor_locations, irf_overall, position_mapping, velocity_mapping',
    [
        (
                GaussianState(StateVector([0, 0, 0, 0]),
                              CovarianceMatrix(np.diag([5., 10., 5., 10.]))),
                StateVectors([StateVector([[-10], [10]])]),
                1.,
                [0, 2],
                [1, 3]
        ),
        (
                GaussianState(StateVector([0, 0, 0, 0]),
                              CovarianceMatrix(np.diag([5., 10., 5., 10.]))),
                StateVectors([StateVector([[-10], [10]])]),
                1.,
                [0, 2],
                None
        )
    ]
)
def test_compute_pcrb_single(prior, transition_model, measurement_model, groundtruth,
                             sensor_locations, irf_overall, position_mapping, velocity_mapping):
    metric = PCRBMetric._compute_pcrb_single(prior, transition_model, measurement_model,
                                             groundtruth, sensor_locations, irf_overall,
                                             position_mapping, velocity_mapping)

    expected_pos_rmse = np.array([3.16227766, 2.73899281, 2.70945734, 2.57092567, 2.41880754,
                                  2.28452324, 2.1740353, 2.08692291, 2.02108617, 1.97375227,
                                  1.94170446, 1.92149958, 1.90979313, 1.90367124, 1.90086597,
                                  1.89980342, 1.8995155, 1.899484, 1.89947979, 1.89943311,
                                  1.89934607])
    expected_vel_rmse = np.array([4.47213595, 3.17148318, 2.02052469, 1.38909331, 1.04690571,
                                  0.85577243, 0.74853932, 0.68976806, 0.65903667, 0.64408669,
                                  0.63756245, 0.63517145, 0.63454123, 0.63447629, 0.63446495,
                                  0.6343556, 0.63415564, 0.63391957, 0.63369704, 0.63351646,
                                  0.63338613])

    # NOTE: Checking value of inverse_j is not necessary since rmse values are derived from
    # inverse_j, therefore if they are correct, then inverse_j is also correct.
    assert 'inverse_j' in metric
    assert metric['track'] == groundtruth
    assert np.allclose(metric['position_RMSE'], expected_pos_rmse)
    if velocity_mapping is not None:
        assert np.allclose(metric['velocity_RMSE'], expected_vel_rmse)
    else:
        assert 'velocity_RMSE' not in metric


@pytest.mark.parametrize(
    'prior, sensor_locations, irf_overall, position_mapping, velocity_mapping',
    [
        (
                GaussianState(StateVector([0, 0, 0, 0]),
                              CovarianceMatrix(np.diag([5., 10., 5., 10.]))),
                StateVectors([StateVector([[-10], [10]])]),
                1.,
                [0, 2],
                [1, 3]
        ),
        (
                GaussianState(StateVector([0, 0, 0, 0]),
                              CovarianceMatrix(np.diag([5., 10., 5., 10.]))),
                StateVectors([StateVector([[-10], [10]])]),
                1.,
                [0, 2],
                None
        )
    ]
)
def test_computemetric(prior, transition_model, measurement_model, groundtruth,
                       sensor_locations, irf_overall, position_mapping, velocity_mapping):
    pcrb = PCRBMetric(prior=prior,
                      transition_model=transition_model,
                      measurement_model=measurement_model,
                      sensor_locations=sensor_locations,
                      position_mapping=position_mapping,
                      velocity_mapping=velocity_mapping,
                      irf=irf_overall,
                      truths_key='truths',
                      generator_name='generator')

    manager = MultiManager([pcrb])

    manager.add_data({'truths': groundtruth})

    metric = pcrb.compute_metric(manager)[0]

    assert metric.title == 'PCRB Metrics'
    assert metric.time_range.start_timestamp == groundtruth.states[0].timestamp
    assert metric.time_range.end_timestamp == groundtruth.states[-1].timestamp
    assert metric.generator == pcrb

    expected_values = ['track', 'inverse_j', 'position_RMSE']
    if velocity_mapping:
        expected_values.append('velocity_RMSE')
    for value in expected_values:
        assert value in metric.value

    expected_pos_rmse = np.array([3.16227766, 2.73899281, 2.70945734, 2.57092567, 2.41880754,
                                  2.28452324, 2.1740353, 2.08692291, 2.02108617, 1.97375227,
                                  1.94170446, 1.92149958, 1.90979313, 1.90367124, 1.90086597,
                                  1.89980342, 1.8995155, 1.899484, 1.89947979, 1.89943311,
                                  1.89934607])
    expected_vel_rmse = np.array([4.47213595, 3.17148318, 2.02052469, 1.38909331, 1.04690571,
                                  0.85577243, 0.74853932, 0.68976806, 0.65903667, 0.64408669,
                                  0.63756245, 0.63517145, 0.63454123, 0.63447629, 0.63446495,
                                  0.6343556, 0.63415564, 0.63391957, 0.63369704, 0.63351646,
                                  0.63338613])

    # NOTE: Checking value of inverse_j is not necessary since rmse values are derived from
    # inverse_j, therefore if they are correct, then inverse_j is also correct.
    assert 'inverse_j' in metric.value
    # assert metric.value['track'] == groundtruth
    assert metric.value['track'].states == groundtruth.states
    assert np.allclose(metric.value['position_RMSE'], expected_pos_rmse)
    if velocity_mapping is not None:
        assert np.allclose(metric.value['velocity_RMSE'], expected_vel_rmse)
    else:
        assert 'velocity_RMSE' not in metric.value
