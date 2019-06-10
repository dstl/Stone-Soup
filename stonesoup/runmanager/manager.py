# -*- coding: utf-8 -*-
from .base import RunManager
from ..metricgenerator.manager import SimpleManager

import math
import collections
import os
import datetime
import functools
from stonesoup.types.update import Update
from ..base import Property
from stonesoup.serialise import YAML
from stonesoup.writer.yaml import YAMLWriter
from stonesoup.reader.yaml import YAMLReader
from stonesoup.simulator import Simulator, GroundTruthSimulator, \
    DetectionSimulator

import copy
import itertools

output_folder = 'run_manager_results'
experiment_result_filename = "experiment_results.txt"
data_filename = "data.txt"
shared_data_filename = "shared_data.txt"


class SimpleRunManager(RunManager):
    """SimpleRunManager class for running multiple experiments

    """
    configuration = Property(dict, doc='The experiment configuration.',
                             default=None)
    run_checks = Property(bool, doc='Run format checks on self.configuration '
                                    'before running trackers.')
    illegal_attribute_values = Property(list,
                                        doc='Set of illegal tracker component'
                                            ' attribute values.',
                                        default=set([None]))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.experiment_results = dict()
        self.experiment_results_filename = ""
        self.saved_data_files = list()
        self.experiment_time = ""
        self.failed_checks = False

        if not isinstance(self.illegal_attribute_values, list):
            self.illegal_attribute_values = [self.illegal_attribute_values]

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
                raise ValueError("File does not contain a valid "
                                 "experiment configuration!")

        # import an experiment configuration as a dict
        elif 'object' in kwargs:
            config = kwargs.get("object")
            if isinstance(config, dict):
                self.configuration = config
            else:
                raise ValueError("Object is not a valid "
                                 "experiment configuration!")

        # no experiment configuration soure indicated
        else:
            raise ValueError("No input experiment configuration specified!")

    def check_experiment_config(self):
        """Check self.configuration for obvious errors

        Parameters
        ----------
        none
        """

        # -----------------------------------------------------------
        #   raise a warning if self.configuration['metrics']
        #   does not have 'associator'
        # -----------------------------------------------------------
        if 'associator' not in self.configuration['metrics'].keys():
            print("=============================================")
            print("WARNING: self.configuration['metrics'] does not contain "
                  "an 'associator'.  Metrics that include groundtruth in "
                  "their calculations may fail!")
            print("=============================================")
            raise SystemExit(-1)

        # -------------------------------------------------------------------
        #   loop through all the defined Tracker variants
        # -------------------------------------------------------------------
        trackers = {key: value for key, value
                    in self.configuration.items() if 'tracker' in key}
        try:
            local_trackers = copy.deepcopy(trackers)
        except TypeError:
            local_trackers = copy.copy(trackers)
        components = self.configuration['components']
        local_components = copy.deepcopy(components)

        simulator_filenames = dict()

        is_monte_carlo = False
        if "monte_carlo" in self.configuration['conditions'].keys():
            is_monte_carlo = True

        for tracker_name, tracker_instance in local_trackers.items():

            # identify all of the parameters of "tracker_instance"
            # that have multiple values
            iter_params = extract_iter_parameters(tracker_instance,
                                                  top_level=True)
            iter_params, simulator_filenames = \
                merge_iter_params(tracker_instance, tracker_name, iter_params,
                                  local_components, simulator_filenames)

            iter_combos = list()
            keys = iter_params.keys()
            for element in itertools.product(*iter_params.values()):
                iter_combos.append(dict(zip(keys, element)))

            # if none of the parameters have multiple values, then just run
            # one permutation of "tracker_instance"
            if not iter_combos:
                iter_combos = [None]

            # ------------------------------------------------------------
            #   Monte Carlo experiments can only have one variant
            # ------------------------------------------------------------
            if is_monte_carlo and (len(iter_combos) > 1 or len(trackers) > 1):
                print("=============================================")
                print("WARNING: Monte Carlo experiments only allow one "
                      "Tracker variant!")
                print("=============================================")
                raise SystemExit(-1)

            # iterate through all defined Tracker combinations
            for combo in iter_combos:

                try:
                    local_tracker_instance = copy.deepcopy(tracker_instance)
                except TypeError:
                    local_tracker_instance = copy.copy(tracker_instance)

                # ------------------------------------------------------------
                # check "local_tracker_instance" to see if the parameters
                # specified in "combo" actually exist
                # ------------------------------------------------------------
                for param_key, param_value in combo.items():
                    if not hasattr(local_tracker_instance, param_key):
                        print("Notice: ", tracker_name,
                              " does not have parameter ", param_key,
                              ", but it is specified in ''conditions''. "
                              "This may be an incorrect assignment.")

                # make the parameter changes specific to this permutation
                if combo:
                    for param_key, param_value in combo.items():
                        rsetattr(local_tracker_instance,
                                 param_key, param_value)

                # ------------------------------------------------------------
                # Congratulations! The Tracker variant has been constituted!
                # ------------------------------------------------------------

                # ------------------------------------------------------------
                #   check all attributes of Tracker/sub-components - warning
                #   if any match items in "self.illegal_attribute_values"
                # ------------------------------------------------------------
                self.check_all_component_attributes(local_tracker_instance,
                                                    tracker_name)

                # ------------------------------------------------------------
                #   if Monte Carlo and NOT using simulated data - FAIL
                # ------------------------------------------------------------
                if is_monte_carlo:
                    if not isinstance(local_tracker_instance.detector,
                                      Simulator):
                        print("=============================================")
                        print("WARNING: Monte Carlo simulations MUST use "
                              "simulated data for the Detector!")
                        print("=============================================")
                        raise SystemExit(-1)
                    elif local_tracker_instance.detector.groundtruth and not \
                            isinstance(
                                local_tracker_instance.detector.groundtruth,
                                Simulator):
                        print("=============================================")
                        print("WARNING: Monte Carlo simulations MUST use "
                              "simulated data for the Groundtruth!")
                        print("=============================================")
                        raise SystemExit(-1)

        return

    def check_all_component_attributes(self, component, component_name):
        """Check attributes of sub-components of a tracker against certain rules

        This function is designed to be called recursively to explore all
        sub-components of a tracker.

        Parameters
        -------------
        illegal_values(list): none of the tracker sub-components or their
            attributes may take on this value
        """

        # check 'component' against 'self.illegal_attribute_values'
        if any(component is value for value in self.illegal_attribute_values):
            print("Notice: ", component_name, " has illegal value ", component)

        component_attribute_names = dir(component)

        for attribute_name in component_attribute_names:
            if not attribute_name.startswith("_") and \
                    "_property_" + attribute_name in component_attribute_names:
                self.check_all_component_attributes(
                    getattr(component, attribute_name),
                    component_name + "." + attribute_name)

        return

    def run_experiment(self, save_detections_tracks=False):
        """Run the experiment encoded in self.configuration

        Parameters
        ----------
        none
        """

        # --------------------------------------------------------------------
        #   perform checks on self.configuration, print warnings if there are
        #   any illegal/questionable situations - it is up to the user to fix
        #   these situations before running the Run Manager - if they wat to
        # --------------------------------------------------------------------
        if self.run_checks:
            self.check_experiment_config()

        # -----------------------------------------------------------
        # separate out individual trackers, conditions, and
        # metrics requirements
        # -----------------------------------------------------------
        trackers = {key: value for key, value in self.configuration.items()
                    if 'tracker' in key}
        components = self.configuration['components']
        conditions = self.configuration['conditions']
        metrics_generators = {key: value for key, value in
                              self.configuration['metrics'].items()
                              if 'metric' in key}
        metrics_associator = self.configuration['metrics']['associator']

        # --------------------------------------------------------------------
        # identify whether this is a Monte Carlo experiment
        # --------------------------------------------------------------------
        is_monte_carlo = False
        num_monte_carlo_iterations = 1
        if "monte_carlo" in conditions.keys():
            is_monte_carlo = True
            num_monte_carlo_iterations = conditions['monte_carlo']

        # --------------------------------------------------------
        # run the trackers and collect the results for metrics
        # --------------------------------------------------------
        experiment_name_length = 5
        self.experiment_time = datetime.datetime.today().\
            strftime('%Y-%m-%d_%H-%M')

        simulator_filenames = dict()

        experiment_index = 1
        for tracker_name, tracker_instance in trackers.items():

            # ---------------------------------------------------------
            # identify all of the parameters of "tracker_instance" that
            # have multiple values
            # ---------------------------------------------------------
            iter_params = extract_iter_parameters(tracker_instance,
                                                  top_level=True)
            iter_params, simulator_filenames = merge_iter_params(
                tracker_instance, tracker_name, iter_params,
                components, simulator_filenames)

            iter_combos = list()
            keys = iter_params.keys()
            for element in itertools.product(*iter_params.values()):
                iter_combos.append(dict(zip(keys, element)))

            # if none of the parameters have multiple values, then just run
            # one permutation of "tracker_instance"
            if not iter_combos or is_monte_carlo:
                iter_combos = [None]

            # ----------------------------------------------------------------
            # loop through all of the permutations of the parameters of
            # "tracker_instance" - if this is not a Monte Carlo experiment,
            # then "num_monte_carlo_iterations" is 1, loops only once
            # ----------------------------------------------------------------
            for loop in range(num_monte_carlo_iterations):

                for combo in iter_combos:

                    experiment_name = "Experiment" + str(experiment_index).\
                        zfill(experiment_name_length)
                    experiment_index += 1

                    try:
                        local_tracker_instance = copy.deepcopy(
                            tracker_instance)
                    except TypeError:
                        local_tracker_instance = copy.copy(tracker_instance)

                    # make the parameter changes specific to this permutation
                    if combo:
                        for param_key, param_value in combo.items():
                            rsetattr(local_tracker_instance,
                                     param_key, param_value)

                    # -----------------------------------------------------
                    # output data to file, if indicated - also replace
                    # "detections source" with a reader pointing to this
                    # output file
                    # -----------------------------------------------------
                    if "output_data" in conditions and \
                            conditions["output_data"] is True:
                        filename = os.path.join(
                            output_folder,
                            self.experiment_time + "_" + experiment_name
                            + "_" + data_filename)

                        with YAMLWriter(
                                path=os.path.normpath(filename),
                                groundtruth_source=local_tracker_instance.
                                detector.groundtruth,
                                detections_source=local_tracker_instance.
                                detector) as writer:
                            writer.write()

                        self.saved_data_files.append(filename)

                        local_tracker_instance.detector = YAMLReader(filename)

                    # -------------------------
                    # run the tracker
                    # -------------------------
                    groundtruth_paths = set()
                    detections = set()
                    tracks = set()

                    for time, ctracks in local_tracker_instance.tracks_gen():
                        detections |= \
                            local_tracker_instance.detector.detections
                        tracks.update(ctracks)

                        # sometimes 'detector' will not include groundtruth
                        # paths, e.g. if the data comes from real-world
                        # sensors
                        try:
                            groundtruth_paths |= \
                                local_tracker_instance.detector.groundtruth.\
                                groundtruth_paths
                        except AttributeError:
                            try:
                                groundtruth_paths |= \
                                    local_tracker_instance.detector.\
                                    groundtruth_paths
                            except AttributeError:
                                pass

                    # Remove tracks that weren't updated after initialisation,
                    # as these would have been created from clutter.
                    tracks = {track for track in tracks
                              if any(isinstance(state, Update)
                                     for state in track[1:])}

                    print("Tracker processed.")

                    # --------------------------------------------------------
                    # run metrics on the results of the trackers
                    # --------------------------------------------------------
                    if metrics_generators is not None \
                            and metrics_associator is not None:
                        metric_manager = SimpleManager(
                            [generator for generator in
                             metrics_generators.values()],
                            associator=metrics_associator)
                        metric_manager.add_data((tracks, groundtruth_paths,
                                                 detections))

                        metric_manager.associate_tracks()
                        metrics_results = metric_manager.generate_metrics()

                        metrics_dict = dict()
                        num_metrics = len(metrics_results)
                        metrics_name_length = int(math.ceil(
                            math.log(num_metrics, 10)))
                        metric_index = 1
                        for metric in metrics_results:
                            metric_name = "Metric" + str(metric_index).\
                                zfill(metrics_name_length)
                            metrics_dict[metric_name] = metric
                            metric_index += 1

                        # ----------------------------------------
                        # export the tracker experiment results
                        # ----------------------------------------
                        this_exp = dict()
                        this_exp['tracker'] = local_tracker_instance
                        this_exp['metrics'] = self.configuration['metrics']
                        this_exp['results'] = metrics_dict
                        if save_detections_tracks:
                            this_exp['detections'] = detections
                            this_exp['groundtruth'] = groundtruth_paths
                            this_exp['tracks'] = tracks

                        # in a situation where multiple Trackers shared a
                        # simulated "detector", the detector in
                        # "local_tracker_instance" is currently a YAML file
                        # reader - replace "detector" with the original
                        # version, to make the experiment results more clear
                        if isinstance(local_tracker_instance.detector,
                                      YAMLReader) and \
                                local_tracker_instance.detector.path in \
                                simulator_filenames.values():
                            component_name = list(simulator_filenames.keys())
                            [list(simulator_filenames.values()
                                  ).index(local_tracker_instance.path)][0]
                            local_tracker_instance.detector = \
                                components[component_name]

                        self.experiment_results[experiment_name] = this_exp

        return

    def output_experiment_results(self):
        """Output experiment results to a text file or something

        Parameters
        ----------
        none
        """

        if self.experiment_results:

            filename = os.path.join(
                output_folder,
                self.experiment_time + "_" + experiment_result_filename)
            output_data(filename, self.experiment_results)

            self.experiment_results_filename = filename

        return


# =========================================
#
#   HELPER FUNCTIONS
#
# =========================================
def extract_iter_parameters(model, model_pathname="", top_level=False,
                            iter_dict=dict()):
    """Takes a model (tracker or other component) and returns a dictionary of
    the iterable components contained in that model
    """

    # when calling this function multiple times for different trackers, it is
    # necessary to reset the variables
    if top_level:
        iter_dict = dict()

    # get the parameters (and other Properties) of the model
    try:
        config = vars(model)
    except TypeError:
        return iter_dict

    # save in 'iter_dict' Properties that are iterables
    for key, value in config.items():
        if key.startswith('_property_'):
            if isinstance(value, collections.Iterator):
                iter_dict[model_pathname +
                          ("." if len(model_pathname) > 0 else "") +
                          key[10:]] = value
            else:
                iter_dict = extract_iter_parameters(
                    value, model_pathname +
                    ("." if len(model_pathname) > 0 else "") +
                    key[10:], top_level=False, iter_dict=iter_dict)

    return iter_dict


def merge_iter_params(tracker, tracker_name, iter_params,
                      components, simulator_filenames):
    """Merges two sets of Tracker property iterators

    iter_params: the various values for properties that were specified as
    "iter" in tracker components: values for properties specified outside
    the Tracker, always as "iter"

    values for a given property can be specified in the Tracker and in
    "components", either as a single value or as and "iter".  If they are
    defined in the Tracker not as an "iter", then they will not appear in
    "iter_params"; if they are defined in the Tracker as an "iter", they will
    appear in "iter_params"

    keys in "components" can take 2 different forms:
    Type 1: tracker01.deleter.x - specifies a specific component for a
        specific tracker
    Type 2: detector01 (another component is tracker01.detector: detector01)
       - specify a generic component that can be assigned to a tracker

    Scenario 1: tracker.deleter.x is specified
        in "tracker" (and therefore "iter_params"): iter(2, 3)
        in "components": iter([4, 5])
        result -> tracker.deleter.x in "iter_params" updated to
        iter([2, 3, 4, 5])
    Scenario 2: tracker.deleter.x is specified
        in "tracker" (and therefore NOT in "iter_params"): 3
        in "components": iter([4, 5])
        result -> tracker.deleter.x in "iter_params" updated to iter([3, 4, 5])
    Scenario 3: tracker.deleter.x is specified
        in "tracker" (and therefore "iter_params"): iter([2,3])
        in "components": 5
        result -> tracker.deleter.x in "iter_params" updated to iter([2, 3, 5])
    Scenario 4: tracker.deleter.x is specified
        in "tracker" (and therefore NOT in "iter_params"): None
        in "components": 5
        result -> tracker.deleter.x updated to 5

    """

    for long_name, value in components.items():

        # determine if this 'component' is of Type 1 or Type 2, and pull out
        # the keys (component names)
        component_tracker_name = long_name.split('.', 1)[0]
        # Component Type 2
        if component_tracker_name == long_name:
            continue
        # Component Type 1
        else:
            component_name = long_name.split('.', 1)[1]

        # --------------------------------
        # process Type 1 components
        # --------------------------------

        # --------------------------------------------------------------------
        # handle instances where 'value' is a reference to another "component"
        # --------------------------------------------------------------------
        # 'value' is a single component
        if isinstance(value, str) and value in components.keys():
            value, simulator_filenames = \
                constitute_referenced_component(value, components[value],
                                                simulator_filenames)
        # 'value' is an iterator - check each element in iterator
        elif isinstance(value, collections.Iterator):
            interim_value = list(value)
            value = []
            for local_value in interim_value:
                if isinstance(local_value, str) and \
                        local_value in components.keys():
                    local_value, simulator_filenames = \
                        constitute_referenced_component(
                            local_value, components[local_value],
                            simulator_filenames)
                    value.append(local_value)
                else:
                    value.append(local_value)

        # component's tracker name must match "tracker_name" for this
        # component to apply to "tracker"
        if component_tracker_name != tracker_name:
            continue

        # correctly format "component_value" as an "iter()"
        if isinstance(value, collections.Iterator):
            component_value = value
        elif isinstance(value, list):
            component_value = iter(value)
        else:
            component_value = iter([value])

        # if "component_name" appears in "iter_params", then join
        # "tracker.component_name" and "iter_params[component_name]" as a
        # single "iter()" in "iter_params" (Scenario 1 or 3 above)
        if component_name in iter_params.keys():
            iter_params[component_name] = itertools.chain(
                iter_params[component_name], component_value)

        # if "component_name" DOES NOT appear in "iter_params", then find the
        # value of "tracker.component_name" and join it with "component_value"
        # as a new "iter()" stored at "iter_params[component_name]"
        # (Scenario 2 above)
        else:
            tracker_component_value = rgetattr(tracker, component_name)
            if tracker_component_value is not None:
                iter_params[component_name] = itertools.chain(
                    iter([rgetattr(tracker, component_name)]), component_value)
            else:
                iter_params[component_name] = component_value

    return iter_params, simulator_filenames


def output_data(local_filename, data):
    """Output experiment results to a text file or something

    Parameters
    ----------
    none
    """

    output = YAML().dumps(data)

    try:
        os.makedirs(os.path.dirname(local_filename), exist_ok=True)
    except FileNotFoundError:
        pass

    local_file = open(local_filename, 'w')
    local_file.write(output)
    local_file.close()

    return


def constitute_referenced_component(ref_component_name,
                                    referenced_component,
                                    simulator_filenames):
    """Constitutes a tracker sub-component that is a reference to a component
    defined in self.configuration['components']

    This function exists because 'GroundTruthSimulator' and
    'DetectionSimulator' have unique handling - to ensure that
    they generate the same data for multiple experiments
    """

    # --------------------------------------------------------------------
    # if 'referenced_component' is not a 'GroundTruthSimulator' or a
    # 'DetectionSimulator', then just return it
    # --------------------------------------------------------------------
    if not isinstance(referenced_component, GroundTruthSimulator) and \
            not isinstance(referenced_component, DetectionSimulator):

        return referenced_component, simulator_filenames

    # -----------------------------------------------------------------------
    # handle case where 'referenced_component' is a 'GroundTruthSimulator'
    # or a 'DetectionSimulator'
    # -----------------------------------------------------------------------
    else:
        if ref_component_name in simulator_filenames.keys():
            return YAMLReader(
                 simulator_filenames[ref_component_name]), \
                    simulator_filenames
        else:
            experiment_time = datetime.datetime\
                .today().strftime('%Y-%m-%d_%H-%M-%S')

            filename = os.path.join(
                output_folder,
                experiment_time + "_" + shared_data_filename)

            if isinstance(referenced_component, GroundTruthSimulator):
                with YAMLWriter(
                        path=os.path.normpath(filename),
                        groundtruth_source=referenced_component) \
                        as writer:
                    writer.write()
            elif isinstance(referenced_component, DetectionSimulator):
                with YAMLWriter(
                        path=os.path.normpath(filename),
                        groundtruth_source=referenced_component.groundtruth,
                        detections_source=referenced_component) \
                        as writer:
                    writer.write()

            simulator_filenames[ref_component_name] = filename

            return YAMLReader(filename), simulator_filenames


# ----------------------------------------------------------------------------
def rsetattr(obj, attr, val):
    """setattr() function for setting attributes several layers down in a
    nested object.
    """
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    """getattr() function for getting attributes several layers down in a
    nested object.
    """
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))
# ----------------------------------------------------------------------------
