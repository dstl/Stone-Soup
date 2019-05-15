# -*- coding: utf-8 -*-
from .base import RunManager
from ..base import Property
from ..serialise import YAML
from ..metricgenerator.manager import SimpleManager

from stonesoup.models.transition import *
from stonesoup.models.measurement import *
from stonesoup.simulator import *
from collections import defaultdict
from stonesoup.simulator import *
import tempfile
import math
from stonesoup.predictor import *
from stonesoup.models.transition import *
from stonesoup.types.state import *
from stonesoup.types.array import *
from stonesoup.updater import *
from stonesoup.hypothesiser import *
from stonesoup.dataassociator.neighbour import *
from stonesoup.initiator import *
from stonesoup.deleter import *
from stonesoup.tracker import *
from stonesoup.reader import *
from stonesoup.serialise import YAML
from stonesoup.writer.yaml import YAMLWriter

import copy
import itertools
import random
import string


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
        trackers_config = {key:value for key, value in self.configuration.items() if 'tracker' in key}
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
        trackers = dict()
        detectors = dict()

        for tracker_name, tracker_conf in trackers_config.iteritems():

            local_trackers = self.constitute_object(tracker_conf, conditions_config)
            detectors[tracker_name] = local_trackers.detector

            for tracker in local_trackers:
                trackers[].append(tracker)

            if len(trackers) > 1:
                is_single_run = True

        # if no explicit indication if this experiment is a single run or a
        # Monte Carlo, then it is a single run
        if not is_single_run and not is_monte_carlo:
            is_single_run = True

        if is_single_run and is_monte_carlo:
            raise AttributeError(
                'Invalid configuration: specifies Monte Carlo runs for '
                'multiple experiment configurations, cannot have both!')

        # --------------------------------------------------------------------
        # if multiple trackers have the same 'detector' and that 'detector'
        # is a simulator, then this indicates that the two trackers should
        # run on the SAME simulated data -
        # this is accomplished by exporting the simulated data from the first
        # tracker to a file and importing that file into the second tracker
        # --------------------------------------------------------------------
        #present_detectors = {tracker.detector for tracker in trackers}
        #repeated_detectors = {detector for detector in present_detectors if isinstance(detector, str)}#(present_detectors.count(detector) > 1 and isinstance(detector, DetectionSimulator))}
        #repeated_detector_output = dict()

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

                # ------------------------------------------------------------
                # if 'detector' of 'tracker_instance' is in
                # 'repeated_detectors', then save data from simulator on first
                # use of 'detector', and re-use this data on subsequent uses
                # of 'detector'
                # ------------------------------------------------------------
                detector = tracker_instance.detector
                if detector in repeated_detectors:

                    # first use of this 'detector' - generate data and output
                    if detector not in repeated_detector_output.items():
                        output_file_name = 'output_data_'.join(random.choice(string.ascii_letters) for i in range(8)).join('.txt')

                        output_file = open(output_file_name, 'w')

                        with YAMLWriter(path=output_file.name, groundtruth_source=detector.groundtruth,
                                        detections_source=detector) as writer:
                            writer.write()

                        repeated_detector_output[detector] = output_file_name

                    # first or subsequent use of this 'detector' - use generated data
                    tracker_instance['detector'] = YAMLReader(repeated_detector_output[detector])

                # ------------------------------------------------------------

                groundtruth_paths = set()
                detections = set()
                tracks = set()

                for time, ctracks in tracker_instance.tracks_gen():
                    detections |= detector.detections
                    tracks.update(ctracks)

                    # sometimes 'detector' will not include groundtruth paths,
                    # e.g. if the data comes from real-world sensors
                    try:
                        groundtruth_paths |= detector.groundtruth.groundtruth_paths
                    except AttributeError:
                        pass

                print("Tracker processed.")

                # --------------------------------------------------------
                # run metrics on the results of the trackers
                # --------------------------------------------------------
                if metrics_generators is not None and associator is not None:
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

                return


    def output_experiment_results(self):
        """Output experiment results to a text file or something

        Parameters
        ----------
        none
        """

        if self.experiment_results:
            output = YAML().dumps(self.experiment_results)

            filename = "experiment_results.txt"

            file = open(filename, 'w')
            file.write(output)
            file.close()
            print(file.name)

        return

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

                #----------------------------
                # Apply RANGE conditions
                # ----------------------------
                if 'RANGE' in conditions.keys():

                    # loop over all RANGE conditions
                    for range_condition in conditions['RANGE']:

                        # if the parameter (key) we are handling at the moment is part of a RANGE condition
                        if range_condition['var'] == key:

                            # RANGE condition that specifies specific values
                            if 'values' in range_condition.keys():

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

                            # RANGE condition that specifies a range of values
                            elif 'min_value' in range_condition.keys() and \
                                    'max_value' in range_condition.keys() and \
                                    'step_size' in range_condition.keys():

                                max_value = range_condition['max_value']
                                min_value = range_condition['min_value']
                                step_size = range_condition['step_size']

                                # sanity check 'max_value','min_value','step_size'
                                if max_value < min_value:
                                    raise Exception('Max_value cannot be less than min_value!')
                                if step_size > (max_value - min_value):
                                    raise Exception('Step_size cannot be greater than difference between max_value and min_value!')

                                possible_values = np.arange(min_value, max_value+step_size, step_size)
                                if value not in possible_values:
                                    possible_values = np.append(possible_values, value)
                                value = {'options': possible_values}

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

        try:
            # extract all the metric generators
            for item in (metrics_config[key] for key, _ in metrics_config.items() if 'metric' in key):

                metric_generators.append(self.constitute_object(item, None)[0])

            # extract the associator
            associator = self.constitute_object(metrics_config['associator'], None)[0]

            return metric_generators, associator

        except KeyError:
            return None, None

# =========================================
#
#   HELPER FUNCTIONS
#
# =========================================

def scan_model_to_config(model):
    """Scans a model (whether it be a Tracker or one of its sub-components)
    and returns a configuration file text string representation
    """

    output_config = {}

    # add the class type to 'output_config'
    class_name = model.__class__.__name__
    output_config['class'] = class_name

    # get the parameters (and other Properties) of the model
    try:
        config = vars(model)
    except TypeError:
        return model

    # save in 'output_config' only items in 'config' whose name begins with '_parameter_'
    for key, value in config.items():
        if key.startswith('_property_'):
            output_config[key[10:]] = scan_model_to_config(value)

    # if 'model' has no 'properties', then treat this as an atomic object - simply return the object
    if len(output_config) <= 1:
        return model
    else:
        return output_config
