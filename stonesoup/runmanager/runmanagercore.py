from os import mkdir
from stonesoup.serialise import YAML
import copy
import json
from runmanagermetrics import RunmanagerMetrics
from inputmanager import InputManager
import sys
from datetime import datetime



def read_json(json_input):
    with open(json_input) as json_file:
        json_data = json.load(json_file)
        # print(json.dumps(json_data, indent=4, sort_keys=True))
        return json_data


def run(config_path, parameters_path, output_path = None):
    """Run the run manager

    Args:
        config_path : Path of the config file
        parameters_path : Path of the parameters file
    """
    input_manager=InputManager()
    json_data = read_json(parameters_path)
    trackers_combination_dict = input_manager.generate_parameters_combinations(json_data["parameters"])
    combo_dict = input_manager.generate_all_combos(trackers_combination_dict)

    with open(config_path, 'r') as file:
        tracker, ground_truth, metric_manager = read_config_file(file)

    trackers = []
    ground_truths = []
    metric_managers = []

    trackers, ground_truths, metric_managers = set_trackers(combo_dict,tracker, ground_truth, metric_manager )
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    for idx in range(0, len(trackers)):
        for runs_num in range(0,json_data["runs_num"]):
            dir_name = "metrics_{}".format(dt_string)+"/simulation_{}".format(idx)+"/run_{}".format(runs_num)
            run_simulation(trackers[idx],metric_managers[idx],dir_name)

def run_simulation(tracker,metric_manager,dir_name):
    """Start the simulation

    Args:
        tracker: Tracker
        groundtruth: GroundTruth
        metric_manager: Metric Manager
    """

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

        #Generate the metrics
        metrics = metric_manager.generate_metrics()    
 
    except Exception as e:
        print(f'Failed to run Simulation: {e}', flush=True)

    else:
        RunmanagerMetrics.groundtruth_to_csv(dir_name, groundtruth)
        RunmanagerMetrics.detection_to_csv(dir_name, detections)
        RunmanagerMetrics.metrics_to_csv(dir_name, metrics)
        print('Success!', flush=True)


def set_trackers(combo_dict,tracker, ground_truth, metric_manager):
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