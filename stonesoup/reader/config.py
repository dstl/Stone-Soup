# -*- coding: utf-8 -*-
import copy as copy
import itertools

from .base import Reader
from ..dataassociator.neighbour import NearestNeighbour, GlobalNearestNeighbour
from ..deleter.error import CovarianceBasedDeleter
from ..deleter.time import UpdateTimeDeleter, UpdateTimeStepsDeleter
from ..hypothesiser.distance import MahalanobisDistanceHypothesiser
from ..hypothesiser.filtered import FilteredDetectionsHypothesiser
from ..initiator.simple import SinglePointInitiator,\
    LinearMeasurementInitiator, GaussianParticleInitiator
from ..models.measurement.linear import LinearGaussian
from ..models.measurement.nonlinear import RangeBearingElevationGaussianToCartesian,\
    RangeBearingGaussianToCartesian, BearingElevationGaussianToCartesian
from ..models.transition.linear import LinearGaussianTransitionModel,\
    CombinedLinearGaussianTransitionModel,\
    LinearGaussianTimeInvariantTransitionModel, ConstantVelocity,\
    ConstantAcceleration, Singer, SingerApproximate, ConstantTurn
from ..platform.simple import SensorPlatform
from ..predictor.kalman import KalmanPredictor, ExtendedKalmanPredictor
from ..predictor.particle import ParticlePredictor
from ..reader.aishub import JSON_AISDetectionReader
from ..reader.file import FileReader, TextFileReader, BinaryFileReader
from ..reader.generic import CSVDetectionReader
from..reader.yaml import YAMLReader
from ..resampler.particle import SystematicResampler
from ..sensor.radar import SimpleRadar
from ..simulator.simple import SingleTargetGroundTruthSimulator,\
    MultiTargetGroundTruthSimulator, SimpleDetectionSimulator
from ..smoother.lineargaussian import Backward
from ..tracker.simple import SingleTargetTracker, MultiTargetTracker
from ..updater.kalman import KalmanUpdater, ExtendedKalmanUpdater
from ..updater.particle import ParticleUpdater
from ..writer.yaml import YAMLWriter
from ..types.array import StateVector, CovarianceMatrix
from ..types.detection import Detection, GaussianDetection, Clutter
from ..types.groundtruth import GroundTruthState, GroundTruthPath
from ..types.hypothesis import Hypothesis, DistanceHypothesis, JointHypothesis,\
    DistanceJointHypothesis
from ..types.metric import Metric
from ..types.numeric import Probability
from ..types.particle import Particle
from ..types.prediction import Prediction, MeasurementPrediction,\
    StatePrediction, StateMeasurementPrediction, GaussianStatePrediction,\
    GaussianMeasurementPrediction, ParticleStatePrediction,\
    ParticleMeasurementPrediction
from ..types.sensordata import SensorData
from ..types.state import State, StateMutableSequence, GaussianState,\
    ParticleState
from ..types.track import Track
from ..types.update import Update, StateUpdate, GaussianStateUpdate,\
    ParticleStateUpdate

class ConfigConstituter(Reader):
    """Read and build experiment configurations"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def decode_config(self, config_in):
        return self.constitute_object(config_in['tracker'],
                                      config_in['conditions'])

    def constitute_object(self, kwargs_in, conditions):

        kwargs = copy.deepcopy(kwargs_in)

        # if 'kwargs' is already a (single or array of) constituted object(s), simply return it
        if not isinstance(kwargs, dict):
            if not isinstance(kwargs, list):
                return [kwargs]
            else:
                return kwargs

        # get class name of object to be constituted, process parameters
        class_name = None

        if 'class' in kwargs.keys():
            class_name = kwargs['class']
            del kwargs['class']
        elif 'options' in kwargs.keys():
            x = 1
        else:
            raise Exception('Experiment configuration is not well-formed!')

        # case where the current object is a class
        if class_name is not None:
            # constitute any class keyword arguments
            keys = []
            values = []

            for key, value in kwargs.items():

                # Apply RANGE conditions
                if 'RANGE' in conditions.keys():

                    # loop over all RANGE conditions
                    for range_condition in conditions['RANGE']:

                        # RANGE condition that specifies specific values
                        if 'var' in range_condition.keys():

                            # if the parameter (key) we are handling at the moment is part of a RANGE condition
                            if range_condition['var'] == key:

                                # instance where RANGE condition has specified values
                                possible_values = []
                                if isinstance(value, list):
                                    possible_values = [*copy.deepcopy(value),
                                                       *copy.deepcopy(
                                                           range_condition[
                                                               'values'])]
                                else:
                                    possible_values = copy.deepcopy(
                                        range_condition['values'])
                                    possible_values.append(value)
                                value = {'options': possible_values}

                            # instance where RANGE condition specifies min value, max value, step size
                            if 'min_value' in range_condition.keys() and \
                                            'max_value' in range_condition.keys() and \
                                            'step_size' in range_condition.keys():
                                # TODO - implement this logic
                                s = 1

                # Constitute the object
                # single class keyword argument
                if isinstance(value, dict) and 'class' in value.keys():
                    keys.append(key)
                    values.append(self.constitute_object(value, conditions))

                # class keyword argument is an array of possible keyword arguments
                elif isinstance(value, dict) and 'options' in value.keys():
                    keys.append(key)
                    values.append(
                        [self.constitute_object(inner_value, conditions)[0] for
                         inner_value in value['options']])

                # object is already constituted
                else:
                    keys.append(key)
                    values.append([value])

            kw_combos = list(itertools.product(*values))

            sub_configs = []

            for vals in kw_combos:
                sub_configs.append(dict(zip(keys, vals)))

            # constitute the object
            return [globals()[class_name](**subconfig) for subconfig in
                    sub_configs]

        # case where the current object is a set of possible objects
        else:
            return [self.constitute_object(local_value, conditions) for
                    _, local_value in kwargs.items()]
