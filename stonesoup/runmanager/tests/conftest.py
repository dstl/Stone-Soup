# -*- coding: utf-8 -*-
import datetime
import pytest
import numpy as np

from stonesoup.models.transition.linear import \
    CombinedLinearGaussianTransitionModel, ConstantVelocity
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.simulator.simple import MultiTargetGroundTruthSimulator
from stonesoup.types.state import GaussianState
from stonesoup.types.array import StateVector, CovarianceMatrix
from stonesoup.simulator.simple import SimpleDetectionSimulator
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.updater.kalman import KalmanUpdater
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.measures import Mahalanobis
from stonesoup.dataassociator.neighbour import NearestNeighbour
from stonesoup.initiator.simple import SinglePointInitiator
from stonesoup.deleter.error import CovarianceBasedDeleter
from stonesoup.tracker.simple import MultiTargetTracker
from stonesoup.metricgenerator import BasicMetrics
from stonesoup.metricgenerator.ospametric import OSPAMetric
from stonesoup.metricgenerator.tracktotruthmetrics import SIAPMetrics
from stonesoup.dataassociator.tracktotrack import EuclideanTrackToTruth


@pytest.fixture()
def base_tracker():
    class BaseTracker(dict):
        def __init__(self):

            # -------------------------
            # Create a Tracker
            # -------------------------
            transition_model = \
                CombinedLinearGaussianTransitionModel((ConstantVelocity(1),
                                                       ConstantVelocity(1)))

            measurement_model = LinearGaussian(ndim_state=4, mapping=[0, 2],
                                               noise_covar=np.diag([10, 10]))

            groundtruth_sim = MultiTargetGroundTruthSimulator(
                transition_model=transition_model,
                initial_state=GaussianState(
                    StateVector([[0], [0], [0], [0]]),
                    CovarianceMatrix(np.diag([1000000, 10, 1000000, 10]))),
                timestep=datetime.timedelta(seconds=5),
                birth_rate=0.05,
                death_probability=0.05
            )

            detection_sim = SimpleDetectionSimulator(
                groundtruth=groundtruth_sim,
                measurement_model=measurement_model,
                meas_range=np.array([[-1, 1], [-1, 1]]) * 5000,  # cluttr area
                detection_probability=0.9,
                clutter_rate=0.3,
            )

            predictor = KalmanPredictor(transition_model)

            updater = KalmanUpdater(measurement_model)

            hypothesiser = \
                DistanceHypothesiser(predictor, updater,
                                     Mahalanobis(), missed_distance=3)

            data_associator = NearestNeighbour(hypothesiser)

            initiator = SinglePointInitiator(
                GaussianState(np.array([[0], [0], [0], [0]]),
                              np.diag([10000, 100, 10000, 1000])),
                measurement_model=measurement_model)

            deleter = CovarianceBasedDeleter(covar_trace_thresh=1E3)

            tracker = MultiTargetTracker(
                initiator=initiator,
                deleter=deleter,
                detector=detection_sim,
                data_associator=data_associator,
                updater=updater,
            )

            # --------------------------------------
            # Create metrics generators
            # --------------------------------------
            basic_calculator = BasicMetrics()
            ospa_calculator = \
                OSPAMetric(c=10, p=1,
                           measurement_model_track=measurement_model,
                           measurement_model_truth=measurement_model)
            siap_calculator = SIAPMetrics()

            associator = \
                EuclideanTrackToTruth(
                    measurement_model_truth=measurement_model,
                    measurement_model_track=measurement_model,
                    association_threshold=30)

            metrics_conditions = {'metric_01': basic_calculator,
                                  'metric_02': ospa_calculator,
                                  'metric_03': siap_calculator,
                                  'associator': associator}

            # ----------------------------------------------
            # encode experiment into Run Manager format
            # ----------------------------------------------
            self['tracker01'] = tracker
            self['components'] = {}
            self['conditions'] = {}
            self['metrics'] = metrics_conditions

        def get_base_tracker(self):
            return {key: value for key, value in self.items()}

    return BaseTracker()
