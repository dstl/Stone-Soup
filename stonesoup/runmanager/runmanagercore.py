from os import mkdir
from stonesoup.serialise import YAML
import numpy as np
import copy
import json
import itertools
import pandas as pd
from itertools import chain
from runmanagermetrics import RunmanagerMetrics
import sys
from operator import attrgetter
from datetime import datetime
import os

def read_json(json_input):
    with open(json_input) as json_file:
        json_data = json.load(json_file)
        # print(json.dumps(json_data, indent=4, sort_keys=True))
        return json_data


# Calculate the steps for each item in a list
def iterations(min_value, max_value, num_samples):
    temp = []
    difference = max_value - min_value
    factor = difference / (num_samples - 1)
    for x in range(num_samples):
        temp.append(min_value + (x * factor))
    return temp

# gets the combinations for one tracker and stores in list
# Once you have steps created from iterations, generate step combinations for one parameter
def get_trackers_list(iterations_container_list, value_min):
    temp =[]
    for x in range(0, len(value_min)):
        temp.append(iterations_container_list[x])
    list_combinations = list(itertools.product(*temp))
    set_combinations = set(list_combinations)
    set_combinations = list(set_combinations)
    for idx, elem in enumerate(set_combinations):
        set_combinations[idx]=list(elem)
        set_combinations[idx]=np.c_[set_combinations[idx]].astype(int)
        
    
    return list(set_combinations)

# Generates all of the combinations between different parameters
def generate_all_combos(trackers_dict):
    """Generates all of the combinations between different parameters

    Args:
        trackers_dict (dict): Dictionary of all the parameters with all the possible values

    Returns:
        dict: Dictionary of all the parameters combined each other
    """
    keys = trackers_dict.keys()
    values = (trackers_dict[key] for key in keys)
    combinations = [dict(zip(keys, combination)) for combination in itertools.product(*values)]
    return combinations


def run(config_path, parameters_path, output_path = None):
    """Run the run manager

    Args:
        config_path : Path of the config file
        parameters_path : Path of the parameters file
    """
    json_data = read_json(parameters_path)
    trackers_combination_dict = generate_parameters_combinations(json_data["parameters"])
    combo_dict = generate_all_combos(trackers_combination_dict)

    with open(config_path, 'r') as file:
        tracker, ground_truth, metric_manager = read_config_file(file)

    trackers = []
    ground_truths = []
    metric_managers = []


    trackers, ground_truths, metric_managers = set_trackers(combo_dict,tracker, ground_truth, metric_manager )
    root_path = " C:/Users/gbellant/Documents/Projects/Serapis/"
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    for idx in range(0, len(trackers)):
        for runs_num in range(0,json_data["runs_num"]):
            dir_name = "metrics_{}".format(dt_string)+"/simulation_{}".format(idx)+"/run_{}".format(runs_num)
            run_simulation(trackers[idx],metric_managers[idx],dir_name)

"""Start the simulation

    Args:
        tracker: Tracker
        groundtruth: GroundTruth
        metric_manager: Metric Manager
    """
def run_simulation(tracker,metric_manager,dir_name):

    detector = tracker.detector
    try:
        groundtruth = set()
        detections = set()
        tracks = set()

        for time, ctracks in tracker:
            #Update groundtruth, tracks and detections
            groundtruth.update(tracker.detector.groundtruth.groundtruth_paths)
            tracks.update(ctracks)
            detections.update(tracker.detector.detections)

            RunmanagerMetrics.tracks_to_csv(dir_name,ctracks)
            metric_manager.add_data(tracker.detector.groundtruth.groundtruth_paths,ctracks,tracker.detector.detections)

        RunmanagerMetrics.groundtruth_to_csv(dir_name, groundtruth)
        RunmanagerMetrics.detection_to_csv(dir_name, detections)

        #Add the data to the metric_manager
 
        #Generate the metrics
        metrics = metric_manager.generate_metrics()                            
        RunmanagerMetrics.metrics_to_csv(dir_name, metrics)

        ##Save the data in csv file

    except Exception as e:
        print(f'Failure: {e}', flush=True)
        # return None
    else:
        print('Success!', flush=True)


def generate_parameters_combinations(parameters):
    """[summary]
    From a list of parameters with, min, max and n_samples values generate all the possible values

    Args:
        parameters ([type]): [list of parameters used to calculate all the possible parameters]

    Returns:
        [dict]: [dictionary of all the combinations]
    """
    combination_dict = {}
    combo_list = {}
    int_list = {}
    iters = []

    for param in parameters:
        for key, val in param.items():
            path = param["path"]

            if type(val) is list and key == "value_min":
                for x in range(len(val)):
                    iters.append(iterations(param["value_min"][x], param["value_max"][x], param["n_samples"]))
                combo_list[path] = get_trackers_list(iters, param["value_min"])
                combination_dict.update(combo_list)

            if type(val) is int and key == "value_min":
                path = param["path"]
                int_iterations = iterations(param["value_min"], param["value_max"], param["n_samples"])
                int_list[path] = [int(x) for x in int_iterations]
                combination_dict.update(int_list)

    return combination_dict


def set_trackers(combo_dict,tracker, ground_truth, metric_manager ):
    """Set the trackers, groundtruths and metricmanagers list (stonesoup objects)

    Args:
        combo_dict (dict): dictionary of all the possible combinations of values
        
        tracker (tracker): stonesoup tracker
        
        groundtruth (groundtruth): stonesoup groundtruth
        
        metric_manager (metricmanager): stonesoup metric_manager

    Returns:
        list: list of trackers
        list: list of groundtruths
        list: list of metric managers
    """
    trackers = []
    ground_truths = []
    metric_managers = []

    for parameter in combo_dict:
        tracker_copy, ground_truth_copy, metric_manager_copy = copy.deepcopy(
            (tracker, ground_truth, metric_manager))
        for k, v in parameter.items():
            split_path = k.split('.')
            path_param = '.'.join(split_path[1::])
            split_path =split_path[1::]

            # setattr(tracker_copy.initiator, split_path[-1], v)
            set_param(split_path,tracker_copy,v)
        trackers.append(tracker_copy)
        ground_truths.append(ground_truth_copy)
        metric_managers.append(metric_manager_copy)

    return trackers,ground_truths,metric_managers




def set_param(split_path,el,value):
    """[summary]

    Args:
        split_path ([type]): [description]
        el ([type]): [description]
        value ([type]): [description]
    """
    if len(split_path)>1:
       # print(split_path[0])
        newEl = getattr(el,split_path[0])
        set_param(split_path[1::],newEl,value)
    else:
        # print(value)
        # print(getattr(el,split_path[0]))

        setattr(el,split_path[0],value)
        # print(el)


def read_config_file(config_file):
    """Read the configuration file

    Args:
        config_file (file path): file path of the configuration file 

    Returns:
        trackers,ground_truth,metric_manager: trackers, ground_truth and metric manager stonesoup structure
    """
    config_string = config_file.read()
    tracker, ground_truth, metric_manager = YAML('safe').load(config_string)
    return tracker, ground_truth, metric_manager


if __name__ == "__main__":
    args = sys.argv[1:]

    
    try:
        configInput = args[0] 
    except:
        configInput= "C:\\Users\\Davidb1\\Documents\\Python\\data\\config.yaml" 
    

    
    try:
        parametersInput = args[1] 
    except:
        parametersInput= "C:\\Users\\Davidb1\\Documents\\Python\\data\\dummy2.json" 
    
    run(configInput, parametersInput)