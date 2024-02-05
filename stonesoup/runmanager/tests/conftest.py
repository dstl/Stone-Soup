import datetime
import os

import numpy as np
import pytest

from ..base import RunManager
from ...dataassociator.neighbour import GNNWith2DAssignment
from ...deleter.error import CovarianceBasedDeleter
from ...feeder.time import TimeBufferedFeeder
from ...hypothesiser.distance import DistanceHypothesiser
from ...measures import Mahalanobis
from ...metricgenerator.manager import SimpleManager
from ...metricgenerator.ospametric import OSPAMetric
# from ...models.control.linear import LinearControlModel
from ...models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity
from ...predictor.kalman import KalmanPredictor
from ...simulator.simple import SimpleDetectionSimulator, MultiTargetGroundTruthSimulator
from ...tracker.simple import MultiTargetTracker
from ...initiator.simple import MultiMeasurementInitiator, SimpleMeasurementInitiator
from ...types.array import CovarianceMatrix
from ...types.state import GaussianState
from ...models.measurement.linear import LinearGaussian
from ...updater.kalman import KalmanUpdater
from ...writer.json import JSONParameterWriter
from ...writer.yaml import YAMLConfigWriter


# Instantiate Stone Soup objects used to create config files
@pytest.fixture()
def config_parameters():
    prior_state = GaussianState(state_vector=[[0], [0], [0], [0]],
                                covar=np.diag([0, 0.5, 0, 0.5]))

    measurement_model = LinearGaussian(ndim_state=4,
                                       mapping=[0, 2],
                                       noise_covar=CovarianceMatrix([[0.25, 0],
                                                                     [0, 0.25]]))

    deleter = CovarianceBasedDeleter(covar_trace_thresh=2)

    transition_model = CombinedLinearGaussianTransitionModel(model_list=[ConstantVelocity(0.05),
                                                                         ConstantVelocity(0.05)])

    # control_model = LinearControlModel(ndim_state=4,
    #                                    mapping=[],
    #                                    control_vector=np.array([0, 0, 0, 0]),
    #                                    control_matrix=np.zeros([4, 4]),
    #                                    control_noise=np.zeros([4, 4]))

    predictor = KalmanPredictor(transition_model=transition_model,
                                # control_model=control_model
                                control_model=None)

    updater = KalmanUpdater(measurement_model=measurement_model)

    hypothesiser = DistanceHypothesiser(predictor=predictor,
                                        updater=updater,
                                        measure=Mahalanobis(),
                                        missed_distance=3)

    data_associator = GNNWith2DAssignment(hypothesiser=hypothesiser)

    initiator = MultiMeasurementInitiator(prior_state=prior_state,
                                          measurement_model=measurement_model,
                                          deleter=deleter,
                                          data_associator=data_associator,
                                          updater=updater,
                                          min_points=3,
                                          initiator=SimpleMeasurementInitiator(
                                              prior_state=prior_state,
                                              measurement_model=measurement_model))

    initial_state = GaussianState(state_vector=[[0], [0], [0], [0]],
                                  covar=[[4.0, 0.0, 0.0, 0.0],
                                         [0.0, 0.5, 0.0, 0.0],
                                         [0.0, 0.0, 4.0, 0.0],
                                         [0.0, 0.0, 0.0, 0.5]])

    ground_truth = MultiTargetGroundTruthSimulator(transition_model=transition_model,
                                                   initial_state=initial_state,
                                                   timestep=datetime.timedelta(5),
                                                   number_steps=20,
                                                   birth_rate=0.3,
                                                   death_probability=0.05)

    detector = SimpleDetectionSimulator(groundtruth=ground_truth,
                                        measurement_model=measurement_model,
                                        meas_range=np.array([[-30, 30],
                                                             [-30, 30]]),
                                        detection_probability=0.9,
                                        clutter_rate=1)

    detections = set()
    for time, dets in detector:
        detections |= dets

    generators = [OSPAMetric(p=1, c=100)]

    ospa_metric_manager = SimpleManager(generators=generators)

    tracker_w_gt = MultiTargetTracker(initiator=initiator,
                                      deleter=deleter,
                                      detector=detector,
                                      data_associator=data_associator,
                                      updater=updater)

    tracker_no_gt = MultiTargetTracker(initiator=initiator,
                                       deleter=deleter,
                                       detector=TimeBufferedFeeder(detections),
                                       data_associator=data_associator,
                                       updater=updater)
    return {'tracker_w_gt': tracker_w_gt,
            'tracker_no_gt': tracker_no_gt,
            'ground_truth': ground_truth,
            'detector': detector,
            'ospa_metric_manager': ospa_metric_manager}


@pytest.fixture()
def tempdirs(tmp_path):
    config_dir = tmp_path

    single_file_dir = os.path.join(config_dir, 'single_file_dir')
    os.mkdir(single_file_dir)

    additional_pairs_dir = os.path.join(config_dir, 'additional_pairs_dir')
    os.mkdir(additional_pairs_dir)

    wrong_order_dir = os.path.join(config_dir, 'wrong_order_dir')
    os.mkdir(wrong_order_dir)

    return {'config_dir': config_dir,
            'single_file_dir': single_file_dir,
            'additional_pairs_dir': additional_pairs_dir,
            'wrong_order_dir': wrong_order_dir}


@pytest.fixture()
def tempfiles(config_parameters, tempdirs):
    config_dir = tempdirs['config_dir']
    single_file_dir = tempdirs['single_file_dir']
    additional_pairs_dir = tempdirs['additional_pairs_dir']
    wrong_order_dir = tempdirs['wrong_order_dir']

    # YAML config: Tracker (containing GT) only
    config_tracker_only = os.path.join(config_dir, 'config_tracker_only.yaml')
    yaml_writer = YAMLConfigWriter(path=config_tracker_only,
                                   tracker=config_parameters['tracker_w_gt'],
                                   groundtruths=None,
                                   detections=None,
                                   metricmanager=None)
    yaml_writer.write()

    # YAML config: Tracker and GT
    config_tracker_gt = os.path.join(config_dir, 'config_tracker_gt.yaml')
    yaml_writer = YAMLConfigWriter(path=config_tracker_gt,
                                   tracker=config_parameters['tracker_w_gt'],
                                   groundtruths=config_parameters['ground_truth'],
                                   detections=None,
                                   metricmanager=None)
    yaml_writer.write()

    # YAML config: Tracker and detections
    config_tracker_dets = os.path.join(config_dir, 'config_tracker_dets.yaml')
    yaml_writer = YAMLConfigWriter(path=config_tracker_dets,
                                   tracker=config_parameters['tracker_w_gt'],
                                   groundtruths=None,
                                   detections=config_parameters['detector'],
                                   metricmanager=None)
    yaml_writer.write()

    # YAML config: Tracker, GT and detections
    config_tracker_gt_dets = os.path.join(config_dir, 'config_tracker_gt_dets.yaml')
    yaml_writer = YAMLConfigWriter(path=config_tracker_gt_dets,
                                   tracker=config_parameters['tracker_w_gt'],
                                   groundtruths=config_parameters['ground_truth'],
                                   detections=config_parameters['detector'],
                                   metricmanager=None)
    yaml_writer.write()

    # YAML config: Tracker (containing GT) and Metric Manager
    config_tracker_mm = os.path.join(config_dir, 'config_tracker_mm.yaml')
    yaml_writer = YAMLConfigWriter(path=config_tracker_mm,
                                   tracker=config_parameters['tracker_w_gt'],
                                   groundtruths=None,
                                   detections=None,
                                   metricmanager=config_parameters['ospa_metric_manager'])
    yaml_writer.write()

    # YAML config: Tracker, GT and Metric Manager
    config_tracker_gt_mm = os.path.join(config_dir, 'config_tracker_gt_mm.yaml')
    yaml_writer = YAMLConfigWriter(path=config_tracker_gt_mm,
                                   tracker=config_parameters['tracker_w_gt'],
                                   groundtruths=config_parameters['ground_truth'],
                                   detections=None,
                                   metricmanager=config_parameters['ospa_metric_manager'])
    yaml_writer.write()

    # YAML config: Tracker, detections and Metric Manager
    config_tracker_dets_mm = os.path.join(config_dir, 'config_tracker_dets_mm.yaml')
    yaml_writer = YAMLConfigWriter(path=config_tracker_dets_mm,
                                   tracker=config_parameters['tracker_w_gt'],
                                   groundtruths=config_parameters['ground_truth'],
                                   detections=config_parameters['detector'],
                                   metricmanager=config_parameters['ospa_metric_manager'])
    yaml_writer.write()

    # YAML config: Tracker, GT, detections and Metric Manager
    config_tracker_gt_dets_mm = os.path.join(config_dir, 'config_tracker_gt_dets_mm.yaml')
    yaml_writer = YAMLConfigWriter(path=config_tracker_gt_dets_mm,
                                   tracker=config_parameters['tracker_w_gt'],
                                   groundtruths=config_parameters['ground_truth'],
                                   detections=config_parameters['detector'],
                                   metricmanager=config_parameters['ospa_metric_manager'])
    yaml_writer.write()

    test_pair_yaml = os.path.join(config_dir, 'test_pair.yaml')
    yaml_writer = YAMLConfigWriter(path=test_pair_yaml,
                                   tracker=config_parameters['tracker_w_gt'],
                                   groundtruths=config_parameters['ground_truth'],
                                   detections=None,
                                   metricmanager=config_parameters['ospa_metric_manager'])
    yaml_writer.write()

    # Generate json config files
    test_pair_json = os.path.join(config_dir, 'test_pair.json')
    dictionary1 = JSONParameterWriter(test_pair_json)
    dictionary1.add_configuration(proc_num=1, runs_num=4)
    dictionary1.add_parameter([1, 0, 0, 0],
                              "tracker.initiator.initiator.prior_state.state_vector",
                              "StateVector",
                              [0, 0, 0, 0],
                              [1000, 100, 100, 100])
    dictionary1.write()

    test_json = os.path.join(config_dir, 'test_json.json')
    dictionary1 = JSONParameterWriter(test_json)
    dictionary1.add_configuration(proc_num=1, runs_num=4)
    dictionary1.add_parameter([1, 0, 0, 0],
                              "tracker.initiator.initiator.prior_state.state_vector",
                              "StateVector",
                              [0, 0, 0, 0],
                              [1000, 100, 100, 100])
    dictionary1.write()

    test_json_no_run = os.path.join(config_dir, 'test_json_no_run.json')
    dictionary1 = JSONParameterWriter(test_json_no_run)
    dictionary1.add_parameter([1, 0, 0, 0],
                              "tracker.initiator.initiator.prior_state.state_vector",
                              "StateVector",
                              [0, 0, 0, 0],
                              [1000, 100, 100, 100])

    dictionary1.write()

    # Write files to subdirectories
    # Single file subdir
    config_single_file = os.path.join(single_file_dir, 'config_single_file.yaml')
    yaml_writer = YAMLConfigWriter(path=config_single_file,
                                   tracker=config_parameters['tracker_w_gt'],
                                   groundtruths=config_parameters['ground_truth'],
                                   detections=config_parameters['detector'],
                                   metricmanager=config_parameters['ospa_metric_manager'])
    yaml_writer.write()

    # Additional pairs subdir
    config_additional_pairs = os.path.join(additional_pairs_dir, 'config_additional_pairs.yaml')
    yaml_writer = YAMLConfigWriter(path=config_additional_pairs,
                                   tracker=config_parameters['tracker_w_gt'],
                                   groundtruths=config_parameters['ground_truth'],
                                   detections=config_parameters['detector'],
                                   metricmanager=config_parameters['ospa_metric_manager'])
    yaml_writer.write()

    config_all = os.path.join(additional_pairs_dir, 'config_additional_pairs.json')
    dictionary1 = JSONParameterWriter(config_all)
    dictionary1.add_configuration(proc_num=1, runs_num=10)
    dictionary1.add_parameter([2, 2, 2, 2],
                              "tracker.initiator.initiator.prior_state.state_vector",
                              "StateVector",
                              [0, 0, 0, 0],
                              [1000, 100, 100, 100])
    dictionary1.write()

    # Wrong order subdir
    dummy_json = os.path.join(wrong_order_dir, 'dummy.json')
    dictionary1 = JSONParameterWriter(dummy_json)
    dictionary1.add_configuration(proc_num=1, runs_num=4)
    dictionary1.add_parameter([1, 0, 0, 0],
                              "tracker.initiator.initiator.prior_state.state_vector",
                              "StateVector",
                              [0, 0, 0, 0],
                              [1000, 100, 100, 100])
    dictionary1.write()

    config_dummy_yaml = os.path.join(wrong_order_dir, 'dummy.yaml')
    yaml_writer = YAMLConfigWriter(path=config_dummy_yaml,
                                   tracker=config_parameters['tracker_w_gt'],
                                   groundtruths=config_parameters['ground_truth'],
                                   detections=None,
                                   metricmanager=config_parameters['ospa_metric_manager'])
    yaml_writer.write()

    config_dummy1_yaml = os.path.join(wrong_order_dir, 'dummy1.yaml')
    yaml_writer = YAMLConfigWriter(path=config_dummy1_yaml,
                                   tracker=config_parameters['tracker_w_gt'],
                                   groundtruths=config_parameters['ground_truth'],
                                   detections=None,
                                   metricmanager=config_parameters['ospa_metric_manager'])
    yaml_writer.write()

    dummy_parameters_json = os.path.join(wrong_order_dir, 'dummy_parameters.json')
    dictionary1 = JSONParameterWriter(dummy_parameters_json)
    dictionary1.add_configuration(proc_num=1, runs_num=4)
    dictionary1.add_parameter([1, 0, 0, 0],
                              "tracker.initiator.initiator.prior_state.state_vector",
                              "StateVector",
                              [0, 0, 0, 0],
                              [1000, 100, 100, 100])
    dictionary1.write()

    return {'config_tracker_only': config_tracker_only,
            'config_tracker_gt': config_tracker_gt,
            'config_tracker_dets': config_tracker_dets,
            'config_tracker_gt_dets': config_tracker_gt_dets,
            'config_tracker_mm': config_tracker_mm,
            'config_tracker_gt_mm': config_tracker_gt_mm,
            'config_tracker_dets_mm': config_tracker_dets_mm,
            'config_tracker_gt_dets_mm': config_tracker_gt_dets_mm,
            'test_json': test_json,
            'config_all': config_all,
            'test_json_no_run': test_json_no_run,
            'test_pair_json': test_pair_json,
            'test_pair_yaml': test_pair_yaml,
            'config_single_file': config_single_file}


@pytest.fixture()
def runmanagers(tempfiles, tempdirs):

    test_json = tempfiles['test_json']

    test_rm_args0 = {"config": tempfiles['config_tracker_gt_mm'],
                     "parameters": test_json,
                     "nruns": 1,
                     "processes": 1}
    rmc = RunManager(test_rm_args0)

    test_rm_args1 = {"config": tempfiles['config_tracker_gt'],
                     "parameters": test_json,
                     "nruns": 1,
                     "processes": 1}
    rmc_nomm = RunManager(test_rm_args1)

    test_rm_args2 = {"config": None,
                     "parameters": None,
                     "config_dir": tempdirs['config_dir'],
                     "nruns": 1,
                     "processes": 1}
    rmc_config_dir = RunManager(test_rm_args2)

    test_rm_args3 = {"config": None,
                     "parameters": None,
                     "config_dir": tempdirs['wrong_order_dir'],
                     "nruns": 1,
                     "processes": 1}
    rmc_config_dir_w = RunManager(test_rm_args3)

    test_rm_args4 = {"config": tempfiles['config_tracker_gt_mm'],
                     "parameters": test_json,
                     "config_dir": tempdirs['additional_pairs_dir'],
                     "nruns": 1,
                     "processes": 1}
    rmc_config_params_dir = RunManager(test_rm_args4)

    test_rm_args5 = {"config": None,
                     "parameters": None,
                     "config_dir": tempdirs['single_file_dir'],
                     "nruns": 1,
                     "processes": 1}
    rmc_config_dir_single_file = RunManager(test_rm_args5)

    return {'rmc': rmc,
            'rmc_nomm': rmc_nomm,
            'rmc_config_dir': rmc_config_dir,
            'rmc_config_dir_w': rmc_config_dir_w,
            'rmc_config_params_dir': rmc_config_params_dir,
            'rmc_config_dir_single_file': rmc_config_dir_single_file}
