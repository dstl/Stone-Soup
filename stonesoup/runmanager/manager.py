# -*- coding: utf-8 -*-
from .base import RunManager
from ..base import Property
from ..serialise import YAML
from ..metricgenerator.manager import SimpleManager

from stonesoup.models.transition import *#.linear import CombinedLinearGaussianTransitionModel, \
                                   #            ConstantVelocity
from stonesoup.models.measurement import * #linear import LinearGaussian
from stonesoup.simulator import *#simple import MultiTargetGroundTruthSimulator
from collections import defaultdict
from stonesoup.simulator import *#.simple import SimpleDetectionSimulator
import tempfile
import math
from stonesoup.predictor import *#.kalman import KalmanPredictor
from stonesoup.models.transition import *#.linear import ConstantVelocity
from stonesoup.types.state import * #GaussianState
from stonesoup.types.array import *#StateVector, CovarianceMatrix
from stonesoup.updater import *#.kalman import KalmanUpdater
from stonesoup.hypothesiser import *#.distance import MahalanobisDistanceHypothesiser
from stonesoup.dataassociator.neighbour import *#NearestNeighbour
from stonesoup.initiator import *#.simple import SinglePointInitiator
from stonesoup.deleter import *#simple import CovarianceBasedDeleter
from stonesoup.tracker import *#.simple import MultiTargetTracker
from stonesoup.serialise import YAML

import copy
import itertools


class SimpleRunManager(RunManager):
    """SimpleRunManager class for running multiple experiments

    """
    configuration = Property(dict, doc='The experiment configuration.',
                             default=None)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.experiment_results = dict()


    def load_config(self, *args, **kwargs):
        """Reads an experiment configuration and stores it in the
        RunManager object.

        kwargs Parameters
        ----------
        filename : String
            Name of file to be read which contains an experiment
            configuration in YAML format
        object : dict
            multi-level dict that defines an experiment configuration
        """

        # import an experiment configuration from a YAML file
        if 'filename' in kwargs:
            config = None
            with open(kwargs.get("filename"), 'r') as myfile:
                config = YAML().load(myfile.read())

            if isinstance(config, dict):
                self.configuration = config
            else:
                raise ValueError("File does not contain a valid experiment configuration!")

        # import an experiment configuration as a dict
        elif 'object' in kwargs:
            config = kwargs.get("object")
            if isinstance(config, dict):
                self.configuration = config
            else:
                raise ValueError("Object is not a valid experiment configuration!")

        # no experiment configuration soure indicated
        else:
            raise ValueError("No input experiment configuration specified!")

        # TODO Add some format checking for the experiment configuration rather than waiting until we are consituting it to see if it is in a valid format???


    def run_experiment(self, save_detections_tracks=False):
        """Run the experiment encoded in self.configuration

        Parameters
        ----------
        none
        """

        is_monte_carlo = False
        is_single_run = False
        num_monte_carlo_runs = 0

        # -----------------------------------------------------------
        # separate out individual trackers, conditions, and
        # metrics requirements
        # -----------------------------------------------------------
        trackers_config = [value for key, value in self.configuration.items() if 'tracker' in key]
        conditions_config = self.configuration['conditions']
        metrics_config = self.configuration['metrics']

        if len(trackers_config) > 1:
            is_single_run = True

        if 'REPEAT' in conditions_config.keys():
            is_monte_carlo = True
            num_monte_carlo_runs = conditions_config['REPEAT']['num_iter']

        if is_single_run and is_monte_carlo:
            raise AttributeError(
                'Invalid configuration: specifies Monte Carlo runs for '
                'multiple experiment configurations, cannot have both!')

        # ---------------------------------------------------------------
        # build the trackers that are specified in self.configuration
        # ---------------------------------------------------------------
        trackers = []

        for tracker_conf in trackers_config:

            local_trackers = self.constitute_object(tracker_conf, conditions_config)

            for tracker in local_trackers:
                trackers.append(tracker)

            if len(trackers) > 1:
                is_single_run = True

        if is_single_run and is_monte_carlo:
            raise AttributeError(
                'Invalid configuration: specifies Monte Carlo runs for '
                'multiple experiment configurations, cannot have both!')

        # --------------------------------------------------------
        # run the trackers and collect the results for metrics
        # --------------------------------------------------------
        num_experiments = len(trackers_config) if is_single_run else num_monte_carlo_runs
        experiment_name_length = int(math.ceil(math.log(num_experiments, 10)) + 1)

        metrics_generators, associator = self.decode_metrics(metrics_config)

        experiment_index = 1
        for tracker_instance in trackers:

            # multiple loops if this is a Monte Carlo experiment, single loop if not
            for loop_index in range(num_monte_carlo_runs if is_monte_carlo else 1):

                detector = tracker_instance.detector

                groundtruth_paths = set()
                detections = set()
                tracks = set()

                for time, ctracks in tracker_instance.tracks_gen():
                    detections |= detector.detections
                    groundtruth_paths |= detector.groundtruth.groundtruth_paths
                    tracks.update(ctracks)

                print("Tracker processed.")

                # --------------------------------------------------------
                # run metrics on the results of the trackers
                # --------------------------------------------------------
                metric_manager = SimpleManager(metrics_generators, associator=associator)
                metric_manager.add_data((tracks, groundtruth_paths, detections))

                metric_manager.associate_tracks()
                metrics_results = metric_manager.generate_metrics()

                metrics_dict = dict()
                num_metrics = len(metrics_results)
                metrics_name_length = int(math.ceil(math.log(num_metrics, 10)))
                metric_index = 1
                for metric in metrics_results:
                    metric_name = "Metric" + str(metric_index).zfill(metrics_name_length)
                    metrics_dict[metric_name] = metric
                    metric_index += 1

                experiment_name = "Experiment" + str(experiment_index).zfill(experiment_name_length)
                experiment_index += 1

                this_exp = dict()
                this_exp['tracker'] = tracker_instance
                this_exp['metrics'] = metrics_config
                this_exp['results'] = metrics_dict
                if save_detections_tracks:
                    this_exp['detections'] = detections
                    this_exp['groundtruth'] = groundtruth_paths
                    this_exp['tracks'] = tracks

                self.experiment_results[experiment_name] = this_exp


    def output_experiment_results(self):
        """Output experiment results to a text file or something

        Parameters
        ----------
        none
        """

        output = YAML().dumps(self.experiment_results)

        filename = "experiment_results.txt"

        file = open(filename, 'w')
        file.write(output)
        file.close()
        print(file.name)


    def constitute_object(self, tracker_sub_config, conditions):
        """Build the Tracker object specified in self.configuration.  This
        function is called recursively on each object in the configuration.

        Parameters
        ----------
        tracker_sub_config : dict
            a dict describing a Tracker or one of its sub-components that
            needs to be built
        conditions : dict
            the conditions from self.configuration - can include REPEAT,
            RANGE, etc. keywords
        """

        kwargs = copy.deepcopy(tracker_sub_config)

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


    def decode_metrics(self, metrics_config):
        """Extract the Metrics objects specified in self.configuration.  They
        will be used to construct a MetricManager.

        Parameters
        ----------
        metrics_config : dict
            the encoded metrics to be run

        Returns
        -------
        metrics_generators : [MetricGenerator]
            the metric generators that will be used by the MetricManager
        associator : TrackToTrackAssociator
            the associator that will be used by the MetricManager

        """

        metric_generators = []

        # extract all the metric generators
        for item in (metrics_config[key] for key, _ in metrics_config.items() if 'metric' in key):

            metric_generators.append(self.constitute_object(item, None)[0])

        # extract the associator
        associator = self.constitute_object(metrics_config['associator'], None)[0]

        return metric_generators, associator