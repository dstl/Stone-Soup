from stonesoup.serialise import YAML
import numpy as np
import copy
import json
import itertools

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


def run():
    # Initiating by reading in the files
    num_runs = 1  # TEMPORARY VALUE. NEEDS TO BE READ FROM JSON
    path_input = 'C:\\Users\\gbellant\\Documents\\Projects\\Serapis\\dummy2.json'
    config_path = "C:\\Users\gbellant\Documents\Projects\Serapis\\config.yaml"
    json_data = read_json(path_input)
    
    combo_list = {}
    int_list = {}


    trackers_combination_dict = generate_parameters_combinations(json_data["parameters"])

    combo_dict = generate_all_combos(trackers_combination_dict)

    # Everything from this point onwards is from original runmanager
    # This current code still uses the yaml file to run the monte carlo
    from operator import attrgetter

    with open(config_path, 'r') as file:
        tracker, ground_truth, metric_manager = read_config_file(file)

    trackers = []
    ground_truths = []
    metric_managers = []


    trackers, ground_truths, metric_managers = set_trackers(combo_dict,tracker, ground_truth, metric_manager )



    # for i in range(0, json_data["runs_num"]): 

    for idx in range(0, len(trackers)):
        for runs_num in range(0,json_data["runs_num"]):
            try:
                tracks = set()

                for n, (time, ctracks) in enumerate(trackers[idx], 1):  # , 1):
                    tracks.update(ctracks)

                print("\n",trackers[idx].initiator.initiator.prior_state.state_vector)        
                print("\n",trackers[idx].initiator.number_particles)        

                metric_managers[idx].add_data(ground_truths[1], tracks)

                metrics = metric_managers[idx].generate_metrics()
            except Exception as e:
                print(f'Failure: {e}', flush=True)
                print("\n",trackers[idx].initiator.initiator.prior_state.state_vector)        
                print("\n",trackers[idx].initiator.number_particles)        
                # return None
            else:
                print('Success!', flush=True)
        
                # metricsList.append(metrics)

    # for trac in trackers:
    #     print("\n",trac.initiator.initiator.prior_state.state_vector)

    #  print(el)
    # for trac in trackers:
    #     print("\n",trac.initiator.number_particles)
    #print(trackers)
    #trackers()
    # Initialise the tracker
    #  tracker_copy, ground_truth_copy, metric_manager_copy = copy.deepcopy((tracker, ground_truth, metric_manager))
    # tracker_min, ground_truth_min, metric_manager_min = copy.deepcopy((tracker, ground_truth, metric_manager))
    # tracker_max, ground_truth_max, metric_manager_max = copy.deepcopy((tracker, ground_truth, metric_manager))
    # tracker_step, ground_truth_step, metric_manager_step = copy.deepcopy((tracker, ground_truth, metric_manager)) """
    # metricsList = []
    # for i in range(0, json_data["runs_num"]):
        
    #     try:
    #         tracks = set()

    #         for n, (time, ctracks) in enumerate(trackers[i], 1):  # , 1):
    #             tracks.update(ctracks)

    #         # print(tracks)
    #         metric_managers[i].add_data(ground_truths[i], tracks)

    #         metrics = metric_managers[i].generate_metrics()
    #     except Exception as e:
    #         print(f'Failure: {e}', flush=True)
    #         # return None
    #     else:
    #         print('Success!', flush=True)
    #         metricsList.append(metrics)

    # values, labels = plot(metricsList, len(metricsList))




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
          #  print(split_path)

            # setattr(tracker_copy.initiator, split_path[-1], v)
            set_param(split_path,tracker_copy,v)
           # print(tracker_copy)
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
    config_string = config_file.read()
    tracker, ground_truth, metric_manager = YAML('safe').load(config_string)
    return tracker, ground_truth, metric_manager


if __name__ == "__main__":
    run()