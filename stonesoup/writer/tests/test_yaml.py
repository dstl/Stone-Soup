from pathlib import Path
from textwrap import dedent

import pytest

import numpy as np
import datetime

from ..yaml import YAMLWriter, YAMLConfigWriter

from ...types.array import StateVector, CovarianceMatrix
from ...types.state import GaussianState
from ...models.transition.linear import (
    CombinedLinearGaussianTransitionModel, ConstantVelocity)
from ...simulator.simple import MultiTargetGroundTruthSimulator
from ...simulator.simple import SimpleDetectionSimulator
from ...models.measurement.linear import LinearGaussian
from ...predictor.kalman import KalmanPredictor
from ...updater.kalman import KalmanUpdater
from ...hypothesiser.distance import DistanceHypothesiser
from ...measures import Mahalanobis
from ...dataassociator.neighbour import GNNWith2DAssignment
from ...deleter.error import CovarianceBasedDeleter
from ...initiator.simple import MultiMeasurementInitiator
from ...tracker.simple import MultiTargetTracker
from ...metricgenerator.ospametric import OSPAMetric
from ...measures import Euclidean
from ...dataassociator.tracktotrack import TrackToTruth
from ...metricgenerator.manager import SimpleManager


initial_state = GaussianState(StateVector([[0], [0], [0], [0]]),
                              CovarianceMatrix(np.diag([4, 0.5, 4, 0.5])))
transition_model = CombinedLinearGaussianTransitionModel(
    [ConstantVelocity(0.05), ConstantVelocity(0.05)])
config_groundtruth_sim = MultiTargetGroundTruthSimulator(
    transition_model=transition_model,
    initial_state=initial_state,
    timestep=datetime.timedelta(seconds=5),
    number_steps=2,
    birth_rate=0.3,
    death_probability=0.05
)
measurement_model = LinearGaussian(4, [0, 2], np.diag([0.25, 0.25]))
config_detection_sim = SimpleDetectionSimulator(
    groundtruth=config_groundtruth_sim,
    measurement_model=measurement_model,
    detection_probability=0.9,
    meas_range=np.array([[-1, 1], [-1, 1]]) * 30,
    clutter_rate=1
)
predictor = KalmanPredictor(transition_model)
updater = KalmanUpdater(measurement_model)
hypothesiser = DistanceHypothesiser(predictor, updater, measure=Mahalanobis(),
                                    missed_distance=3)
data_associator = GNNWith2DAssignment(hypothesiser)
deleter = CovarianceBasedDeleter(covar_trace_thresh=2)
initiator = MultiMeasurementInitiator(
    prior_state=GaussianState([[0], [0], [0], [0]], np.diag([0.5, 0.15, 0.5, 0.15])),
    measurement_model=measurement_model,
    deleter=deleter,
    data_associator=data_associator,
    updater=updater,
    min_points=3
)
config_tracker = MultiTargetTracker(
    initiator=initiator,
    deleter=deleter,
    detector=config_detection_sim,
    data_associator=data_associator,
    updater=updater,
)
ospa_generator = OSPAMetric(c=10, p=1, measure=Euclidean([0, 2]))
associator = TrackToTruth(association_threshold=30)
config_metric_manager = SimpleManager({"ospa": ospa_generator},
                                      associator=associator)


def test_detections_yaml(detection_reader, tmpdir):
    filename = tmpdir.join("detections.yaml")

    with YAMLWriter(filename.strpath,
                    detections_source=detection_reader) as writer:
        writer.write()

    with filename.open('r') as yaml_file:
        generated_yaml = yaml_file.read()

    expected_yaml = dedent("""\
         ---
         time: 2018-01-01 14:00:00
         detections: !!set {}
         ...
         ---
         time: &id001 2018-01-01 14:01:00
         detections: !!set
           ? !stonesoup.types.detection.Detection
           - state_vector: !stonesoup.types.array.StateVector
             - [1]
           - timestamp: *id001
           - metadata: {}
           :
         ...
         ---
         time: &id001 2018-01-01 14:02:00
         detections: !!set
           ? !stonesoup.types.detection.Detection
           - state_vector: !stonesoup.types.array.StateVector
             - [2]
           - timestamp: *id001
           - metadata: {}
           :
           ? !stonesoup.types.detection.Detection
           - state_vector: !stonesoup.types.array.StateVector
             - [2]
           - timestamp: *id001
           - metadata: {}
           :
         ...
         """)

    assert generated_yaml == expected_yaml


def test_groundtruth_paths_yaml(groundtruth_reader, tmpdir):
    filename = tmpdir.join("groundtruth_paths.yaml")

    with YAMLWriter(filename.strpath,
                    groundtruth_source=groundtruth_reader) as writer:
        writer.write()

    with filename.open('r') as yaml_file:
        generated_yaml = yaml_file.read()

    expected_yaml = dedent("""\
        ---
        time: 2018-01-01 14:00:00
        groundtruth_paths: !!set {}
        ...
        ---
        time: &id001 2018-01-01 14:01:00
        groundtruth_paths: !!set
          ? !stonesoup.types.groundtruth.GroundTruthPath
          - states:
            - !stonesoup.types.groundtruth.GroundTruthState
              - state_vector: !stonesoup.types.array.StateVector
                - [1]
              - timestamp: *id001
              - metadata: {}
          - id: '0'
          :
        ...
        """)

    assert generated_yaml == expected_yaml


def test_tracks_yaml(tracker, tmpdir):
    filename = tmpdir.join("tracks.yaml")

    with YAMLWriter(filename.strpath, tracks_source=tracker) as writer:
        writer.write()

    with filename.open('r') as yaml_file:
        generated_yaml = yaml_file.read()

    expected_yaml = dedent("""\
        ---
        time: 2018-01-01 14:00:00
        tracks: !!set {}
        ...
        ---
        time: &id001 2018-01-01 14:01:00
        tracks: !!set
          ? !stonesoup.types.track.Track
          - states:
            - !stonesoup.types.state.State
              - state_vector: !stonesoup.types.array.StateVector
                - [1]
              - timestamp: *id001
          - id: '0'
          :
        ...
        """)

    assert generated_yaml == expected_yaml


def test_yaml_bad_init(tmpdir):
    filename = tmpdir.join("bad_init.yaml")
    with pytest.raises(ValueError, match="At least one source required"):
        YAMLWriter(filename.strpath)


def test_config_yaml(tmpdir,
                     tracker=config_tracker,
                     groundtruth_sim=config_groundtruth_sim,
                     detection_sim=config_detection_sim,
                     metric_manager=config_metric_manager):
    filename = tmpdir.join("config.yaml")

    with YAMLConfigWriter(filename.strpath,
                          tracker=tracker,
                          groundtruths=groundtruth_sim,
                          detections=detection_sim,
                          metricmanager=metric_manager) as writer:
        writer.write()

    with filename.open('r') as yaml_file:
        generated_yaml = yaml_file.read()

    expected_yaml = dedent("""\
tracker: !stonesoup.tracker.simple.MultiTargetTracker
- initiator: !stonesoup.initiator.simple.MultiMeasurementInitiator
  - prior_state: &id003 !stonesoup.types.state.GaussianState
    - state_vector: !stonesoup.types.array.StateVector
      - - 0
      - - 0
      - - 0
      - - 0
    - covar: !stonesoup.types.array.CovarianceMatrix
      - - 0.5
        - 0.0
        - 0.0
        - 0.0
      - - 0.0
        - 0.15
        - 0.0
        - 0.0
      - - 0.0
        - 0.0
        - 0.5
        - 0.0
      - - 0.0
        - 0.0
        - 0.0
        - 0.15
  - deleter: &id004 !stonesoup.deleter.error.CovarianceBasedDeleter
    - covar_trace_thresh: 2
  - data_associator: &id006 !stonesoup.dataassociator.neighbour.GNNWith2DAssignment
    - hypothesiser: !stonesoup.hypothesiser.distance.DistanceHypothesiser
      - predictor: !stonesoup.predictor.kalman.KalmanPredictor
        - transition_model: &id005 !stonesoup.models.transition.linear.CombinedLinearGaussianTransitionModel
          - model_list:
            - !stonesoup.models.transition.linear.ConstantVelocity
              - noise_diff_coeff: 0.05
            - !stonesoup.models.transition.linear.ConstantVelocity
              - noise_diff_coeff: 0.05
        - control_model: !stonesoup.models.control.linear.LinearControlModel
          - ndim_state: 4
          - mapping: []
          - control_vector: !numpy.ndarray
            - - 0.0
            - - 0.0
            - - 0.0
            - - 0.0
          - control_matrix: !numpy.ndarray
            - - 0.0
              - 0.0
              - 0.0
              - 0.0
            - - 0.0
              - 0.0
              - 0.0
              - 0.0
            - - 0.0
              - 0.0
              - 0.0
              - 0.0
            - - 0.0
              - 0.0
              - 0.0
              - 0.0
          - control_noise: !numpy.ndarray
            - - 0.0
              - 0.0
              - 0.0
              - 0.0
            - - 0.0
              - 0.0
              - 0.0
              - 0.0
            - - 0.0
              - 0.0
              - 0.0
              - 0.0
            - - 0.0
              - 0.0
              - 0.0
              - 0.0
      - updater: &id001 !stonesoup.updater.kalman.KalmanUpdater
        - measurement_model: &id002 !stonesoup.models.measurement.linear.LinearGaussian
          - ndim_state: 4
          - mapping:
            - 0
            - 2
          - noise_covar: !stonesoup.types.array.CovarianceMatrix
            - - 0.25
              - 0.0
            - - 0.0
              - 0.25
      - measure: !stonesoup.measures.Mahalanobis []
      - missed_distance: 3
  - updater: *id001
  - measurement_model: *id002
  - min_points: 3
  - initiator: !stonesoup.initiator.simple.SimpleMeasurementInitiator
    - prior_state: *id003
    - measurement_model: *id002
- deleter: *id004
- detector: !stonesoup.simulator.simple.SimpleDetectionSimulator
  - groundtruth: &id007 !stonesoup.simulator.simple.MultiTargetGroundTruthSimulator
    - transition_model: *id005
    - initial_state: !stonesoup.types.state.GaussianState
      - state_vector: !stonesoup.types.array.StateVector
        - - 0
        - - 0
        - - 0
        - - 0
      - covar: !stonesoup.types.array.CovarianceMatrix
        - - 4.0
          - 0.0
          - 0.0
          - 0.0
        - - 0.0
          - 0.5
          - 0.0
          - 0.0
        - - 0.0
          - 0.0
          - 4.0
          - 0.0
        - - 0.0
          - 0.0
          - 0.0
          - 0.5
    - timestep: !datetime.timedelta 5.0
    - number_steps: 2
    - birth_rate: 0.3
    - death_probability: 0.05
  - measurement_model: *id002
  - meas_range: !numpy.ndarray
    - - -30
      - 30
    - - -30
      - 30
  - detection_probability: 0.9
  - clutter_rate: 1
- data_associator: *id006
- updater: *id001
groundtruth: *id007
metric_manager: !stonesoup.metricgenerator.manager.SimpleManager
- generators:
    ospa: !stonesoup.metricgenerator.ospametric.OSPAMetric
    - p: 1
    - c: 10
    - measure: !stonesoup.measures.Euclidean
      - mapping: &id008
        - 0
        - 2
      - mapping2: *id008
- associator: !stonesoup.dataassociator.tracktotrack.TrackToTruth
  - association_threshold: 30
""")  # noqa

    assert generated_yaml == expected_yaml


def test_config_yaml_no_metricmanager(
        tmpdir,
        tracker=config_tracker,
        groundtruth_sim=config_groundtruth_sim,
        detection_sim=config_detection_sim,
        metric_manager=None):
    filename = tmpdir.join("config.yaml")

    with YAMLConfigWriter(filename.strpath,
                          tracker=tracker,
                          groundtruths=groundtruth_sim,
                          detections=detection_sim,
                          metricmanager=metric_manager) as writer:
        writer.write()

    with filename.open('r') as yaml_file:
        generated_yaml = yaml_file.read()

    expected_yaml = dedent("""\
tracker: !stonesoup.tracker.simple.MultiTargetTracker
- initiator: !stonesoup.initiator.simple.MultiMeasurementInitiator
  - prior_state: &id003 !stonesoup.types.state.GaussianState
    - state_vector: !stonesoup.types.array.StateVector
      - - 0
      - - 0
      - - 0
      - - 0
    - covar: !stonesoup.types.array.CovarianceMatrix
      - - 0.5
        - 0.0
        - 0.0
        - 0.0
      - - 0.0
        - 0.15
        - 0.0
        - 0.0
      - - 0.0
        - 0.0
        - 0.5
        - 0.0
      - - 0.0
        - 0.0
        - 0.0
        - 0.15
  - deleter: &id004 !stonesoup.deleter.error.CovarianceBasedDeleter
    - covar_trace_thresh: 2
  - data_associator: &id006 !stonesoup.dataassociator.neighbour.GNNWith2DAssignment
    - hypothesiser: !stonesoup.hypothesiser.distance.DistanceHypothesiser
      - predictor: !stonesoup.predictor.kalman.KalmanPredictor
        - transition_model: &id005 !stonesoup.models.transition.linear.CombinedLinearGaussianTransitionModel
          - model_list:
            - !stonesoup.models.transition.linear.ConstantVelocity
              - noise_diff_coeff: 0.05
            - !stonesoup.models.transition.linear.ConstantVelocity
              - noise_diff_coeff: 0.05
        - control_model: !stonesoup.models.control.linear.LinearControlModel
          - ndim_state: 4
          - mapping: []
          - control_vector: !numpy.ndarray
            - - 0.0
            - - 0.0
            - - 0.0
            - - 0.0
          - control_matrix: !numpy.ndarray
            - - 0.0
              - 0.0
              - 0.0
              - 0.0
            - - 0.0
              - 0.0
              - 0.0
              - 0.0
            - - 0.0
              - 0.0
              - 0.0
              - 0.0
            - - 0.0
              - 0.0
              - 0.0
              - 0.0
          - control_noise: !numpy.ndarray
            - - 0.0
              - 0.0
              - 0.0
              - 0.0
            - - 0.0
              - 0.0
              - 0.0
              - 0.0
            - - 0.0
              - 0.0
              - 0.0
              - 0.0
            - - 0.0
              - 0.0
              - 0.0
              - 0.0
      - updater: &id001 !stonesoup.updater.kalman.KalmanUpdater
        - measurement_model: &id002 !stonesoup.models.measurement.linear.LinearGaussian
          - ndim_state: 4
          - mapping:
            - 0
            - 2
          - noise_covar: !stonesoup.types.array.CovarianceMatrix
            - - 0.25
              - 0.0
            - - 0.0
              - 0.25
      - measure: !stonesoup.measures.Mahalanobis []
      - missed_distance: 3
  - updater: *id001
  - measurement_model: *id002
  - min_points: 3
  - initiator: !stonesoup.initiator.simple.SimpleMeasurementInitiator
    - prior_state: *id003
    - measurement_model: *id002
- deleter: *id004
- detector: !stonesoup.simulator.simple.SimpleDetectionSimulator
  - groundtruth: &id007 !stonesoup.simulator.simple.MultiTargetGroundTruthSimulator
    - transition_model: *id005
    - initial_state: !stonesoup.types.state.GaussianState
      - state_vector: !stonesoup.types.array.StateVector
        - - 0
        - - 0
        - - 0
        - - 0
      - covar: !stonesoup.types.array.CovarianceMatrix
        - - 4.0
          - 0.0
          - 0.0
          - 0.0
        - - 0.0
          - 0.5
          - 0.0
          - 0.0
        - - 0.0
          - 0.0
          - 4.0
          - 0.0
        - - 0.0
          - 0.0
          - 0.0
          - 0.5
    - timestep: !datetime.timedelta 5.0
    - number_steps: 2
    - birth_rate: 0.3
    - death_probability: 0.05
  - measurement_model: *id002
  - meas_range: !numpy.ndarray
    - - -30
      - 30
    - - -30
      - 30
  - detection_probability: 0.9
  - clutter_rate: 1
- data_associator: *id006
- updater: *id001
groundtruth: *id007
""")  # noqa

    assert generated_yaml == expected_yaml


def test_config_yaml_only_tracker(
        tmpdir,
        tracker=config_tracker,
        groundtruth_sim=None,
        detection_sim=None,
        metric_manager=None):
    filename = tmpdir.join("config.yaml")

    with YAMLConfigWriter(filename.strpath,
                          tracker=tracker,
                          groundtruths=groundtruth_sim,
                          detections=detection_sim,
                          metricmanager=metric_manager) as writer:
        writer.write()

    with filename.open('r') as yaml_file:
        generated_yaml = yaml_file.read()

    expected_yaml = dedent("""\
tracker: !stonesoup.tracker.simple.MultiTargetTracker
- initiator: !stonesoup.initiator.simple.MultiMeasurementInitiator
  - prior_state: &id003 !stonesoup.types.state.GaussianState
    - state_vector: !stonesoup.types.array.StateVector
      - - 0
      - - 0
      - - 0
      - - 0
    - covar: !stonesoup.types.array.CovarianceMatrix
      - - 0.5
        - 0.0
        - 0.0
        - 0.0
      - - 0.0
        - 0.15
        - 0.0
        - 0.0
      - - 0.0
        - 0.0
        - 0.5
        - 0.0
      - - 0.0
        - 0.0
        - 0.0
        - 0.15
  - deleter: &id004 !stonesoup.deleter.error.CovarianceBasedDeleter
    - covar_trace_thresh: 2
  - data_associator: &id006 !stonesoup.dataassociator.neighbour.GNNWith2DAssignment
    - hypothesiser: !stonesoup.hypothesiser.distance.DistanceHypothesiser
      - predictor: !stonesoup.predictor.kalman.KalmanPredictor
        - transition_model: &id005 !stonesoup.models.transition.linear.CombinedLinearGaussianTransitionModel
          - model_list:
            - !stonesoup.models.transition.linear.ConstantVelocity
              - noise_diff_coeff: 0.05
            - !stonesoup.models.transition.linear.ConstantVelocity
              - noise_diff_coeff: 0.05
        - control_model: !stonesoup.models.control.linear.LinearControlModel
          - ndim_state: 4
          - mapping: []
          - control_vector: !numpy.ndarray
            - - 0.0
            - - 0.0
            - - 0.0
            - - 0.0
          - control_matrix: !numpy.ndarray
            - - 0.0
              - 0.0
              - 0.0
              - 0.0
            - - 0.0
              - 0.0
              - 0.0
              - 0.0
            - - 0.0
              - 0.0
              - 0.0
              - 0.0
            - - 0.0
              - 0.0
              - 0.0
              - 0.0
          - control_noise: !numpy.ndarray
            - - 0.0
              - 0.0
              - 0.0
              - 0.0
            - - 0.0
              - 0.0
              - 0.0
              - 0.0
            - - 0.0
              - 0.0
              - 0.0
              - 0.0
            - - 0.0
              - 0.0
              - 0.0
              - 0.0
      - updater: &id001 !stonesoup.updater.kalman.KalmanUpdater
        - measurement_model: &id002 !stonesoup.models.measurement.linear.LinearGaussian
          - ndim_state: 4
          - mapping:
            - 0
            - 2
          - noise_covar: !stonesoup.types.array.CovarianceMatrix
            - - 0.25
              - 0.0
            - - 0.0
              - 0.25
      - measure: !stonesoup.measures.Mahalanobis []
      - missed_distance: 3
  - updater: *id001
  - measurement_model: *id002
  - min_points: 3
  - initiator: !stonesoup.initiator.simple.SimpleMeasurementInitiator
    - prior_state: *id003
    - measurement_model: *id002
- deleter: *id004
- detector: !stonesoup.simulator.simple.SimpleDetectionSimulator
  - groundtruth: !stonesoup.simulator.simple.MultiTargetGroundTruthSimulator
    - transition_model: *id005
    - initial_state: !stonesoup.types.state.GaussianState
      - state_vector: !stonesoup.types.array.StateVector
        - - 0
        - - 0
        - - 0
        - - 0
      - covar: !stonesoup.types.array.CovarianceMatrix
        - - 4.0
          - 0.0
          - 0.0
          - 0.0
        - - 0.0
          - 0.5
          - 0.0
          - 0.0
        - - 0.0
          - 0.0
          - 4.0
          - 0.0
        - - 0.0
          - 0.0
          - 0.0
          - 0.5
    - timestep: !datetime.timedelta 5.0
    - number_steps: 2
    - birth_rate: 0.3
    - death_probability: 0.05
  - measurement_model: *id002
  - meas_range: !numpy.ndarray
    - - -30
      - 30
    - - -30
      - 30
  - detection_probability: 0.9
  - clutter_rate: 1
- data_associator: *id006
- updater: *id001
""")  # noqa

    assert generated_yaml == expected_yaml


def test_config_yaml_no_groundtruth_sim(
        tmpdir,
        tracker=config_tracker,
        groundtruth_sim=None,
        detection_sim=config_detection_sim,
        metric_manager=config_metric_manager):
    filename = tmpdir.join("config.yaml")

    with YAMLConfigWriter(filename.strpath,
                          tracker=tracker,
                          groundtruths=groundtruth_sim,
                          detections=detection_sim,
                          metricmanager=metric_manager) as writer:
        writer.write()

    with filename.open('r') as yaml_file:
        generated_yaml = yaml_file.read()

    expected_yaml = dedent("""\
tracker: !stonesoup.tracker.simple.MultiTargetTracker
- initiator: !stonesoup.initiator.simple.MultiMeasurementInitiator
  - prior_state: &id003 !stonesoup.types.state.GaussianState
    - state_vector: !stonesoup.types.array.StateVector
      - - 0
      - - 0
      - - 0
      - - 0
    - covar: !stonesoup.types.array.CovarianceMatrix
      - - 0.5
        - 0.0
        - 0.0
        - 0.0
      - - 0.0
        - 0.15
        - 0.0
        - 0.0
      - - 0.0
        - 0.0
        - 0.5
        - 0.0
      - - 0.0
        - 0.0
        - 0.0
        - 0.15
  - deleter: &id004 !stonesoup.deleter.error.CovarianceBasedDeleter
    - covar_trace_thresh: 2
  - data_associator: &id006 !stonesoup.dataassociator.neighbour.GNNWith2DAssignment
    - hypothesiser: !stonesoup.hypothesiser.distance.DistanceHypothesiser
      - predictor: !stonesoup.predictor.kalman.KalmanPredictor
        - transition_model: &id005 !stonesoup.models.transition.linear.CombinedLinearGaussianTransitionModel
          - model_list:
            - !stonesoup.models.transition.linear.ConstantVelocity
              - noise_diff_coeff: 0.05
            - !stonesoup.models.transition.linear.ConstantVelocity
              - noise_diff_coeff: 0.05
        - control_model: !stonesoup.models.control.linear.LinearControlModel
          - ndim_state: 4
          - mapping: []
          - control_vector: !numpy.ndarray
            - - 0.0
            - - 0.0
            - - 0.0
            - - 0.0
          - control_matrix: !numpy.ndarray
            - - 0.0
              - 0.0
              - 0.0
              - 0.0
            - - 0.0
              - 0.0
              - 0.0
              - 0.0
            - - 0.0
              - 0.0
              - 0.0
              - 0.0
            - - 0.0
              - 0.0
              - 0.0
              - 0.0
          - control_noise: !numpy.ndarray
            - - 0.0
              - 0.0
              - 0.0
              - 0.0
            - - 0.0
              - 0.0
              - 0.0
              - 0.0
            - - 0.0
              - 0.0
              - 0.0
              - 0.0
            - - 0.0
              - 0.0
              - 0.0
              - 0.0
      - updater: &id001 !stonesoup.updater.kalman.KalmanUpdater
        - measurement_model: &id002 !stonesoup.models.measurement.linear.LinearGaussian
          - ndim_state: 4
          - mapping:
            - 0
            - 2
          - noise_covar: !stonesoup.types.array.CovarianceMatrix
            - - 0.25
              - 0.0
            - - 0.0
              - 0.25
      - measure: !stonesoup.measures.Mahalanobis []
      - missed_distance: 3
  - updater: *id001
  - measurement_model: *id002
  - min_points: 3
  - initiator: !stonesoup.initiator.simple.SimpleMeasurementInitiator
    - prior_state: *id003
    - measurement_model: *id002
- deleter: *id004
- detector: !stonesoup.simulator.simple.SimpleDetectionSimulator
  - groundtruth: !stonesoup.simulator.simple.MultiTargetGroundTruthSimulator
    - transition_model: *id005
    - initial_state: !stonesoup.types.state.GaussianState
      - state_vector: !stonesoup.types.array.StateVector
        - - 0
        - - 0
        - - 0
        - - 0
      - covar: !stonesoup.types.array.CovarianceMatrix
        - - 4.0
          - 0.0
          - 0.0
          - 0.0
        - - 0.0
          - 0.5
          - 0.0
          - 0.0
        - - 0.0
          - 0.0
          - 4.0
          - 0.0
        - - 0.0
          - 0.0
          - 0.0
          - 0.5
    - timestep: !datetime.timedelta 5.0
    - number_steps: 2
    - birth_rate: 0.3
    - death_probability: 0.05
  - measurement_model: *id002
  - meas_range: !numpy.ndarray
    - - -30
      - 30
    - - -30
      - 30
  - detection_probability: 0.9
  - clutter_rate: 1
- data_associator: *id006
- updater: *id001
metric_manager: !stonesoup.metricgenerator.manager.SimpleManager
- generators:
    ospa: !stonesoup.metricgenerator.ospametric.OSPAMetric
    - p: 1
    - c: 10
    - measure: !stonesoup.measures.Euclidean
      - mapping: &id007
        - 0
        - 2
      - mapping2: *id007
- associator: !stonesoup.dataassociator.tracktotrack.TrackToTruth
  - association_threshold: 30
""")  # noqa

    assert generated_yaml == expected_yaml


def test_config_yaml_no_tracker(
        tmpdir,
        tracker=None,
        groundtruth_sim=config_groundtruth_sim,
        detection_sim=config_detection_sim,
        metric_manager=config_metric_manager):
    filename = tmpdir.join("config.yaml")

    with pytest.raises(
            ValueError,
            match="Tracker object must be provided for the Run manager configuration file."):
        with YAMLConfigWriter(filename.strpath,
                              tracker=tracker,
                              groundtruths=groundtruth_sim,
                              detections=detection_sim,
                              metricmanager=metric_manager) as writer:
            writer.write()


def test_bad_config(tmpdir):
    filename = tmpdir.join("config.yaml")
    with pytest.raises(ValueError, match="At least one object required to write to YAML file."):
        YAMLConfigWriter(filename.strpath, None, None, None, None)
    with pytest.raises(ValueError, match="At least one object required to write to YAML file."):
        YAMLConfigWriter(Path(filename), None, None, None, None)
