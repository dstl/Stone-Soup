
import itertools
import json
from pathlib import Path
from itertools import product, starmap
from collections import namedtuple

#from stonesoup.runmanager.runmanager import index

def read_json(json_input):
    with open(json_input) as json_file:
        json_data = json.load(json_file)
        return json_data


def iterations(min_value, max_value, num_samples):
    temp = []
    difference = max_value - min_value
    factor = difference / (num_samples - 1)
    for x in range(num_samples):
        temp.append(min_value + (x * factor))
    return temp


def get_trackers_list(iterations_container_list, value_min):
    temp =[]
    for x in range(0, len(value_min)):
        temp.append(iterations_container_list[x])
    list_combinations = itertools.product(*temp)
    set_combinations = set(list_combinations)

    return set_combinations

def all_combos(**items):
    Combo = namedtuple('_params', items.keys())
    return starmap(Combo, product(*items.values()))

def run():
    # Initiating by reading in the files
    # pathInput = input("Input json directory: ")
    # path = input("Input config yaml directory: ")
    pathInput = 'C:\\Users\\Davidb1\\Documents\\Python\\data\\dummy2.json'
    json_data = read_json(pathInput)
    sim_dir = Path('sims/')
    trackers = []
    tracker = []
    iters = []
    # For each parameter in the json, calculate the increments of each step.
    # Need to sort out bools will do in future
    for param in json_data["parameters"]:
        
        if param["type"] == "bool":
            bool_list = [True, False]
            tracker.append(param["var_name"])
            tracker.append(False)
            trackers.append(tracker)
            tracker = []
            tracker.append(param["var_name"])
            tracker.append(True)
            trackers.append(tracker)
            tracker = []               

        elif param["type"] == "vector":
            for x in range(0, len(param["value_min"])):
                iters.append(iterations(param["value_min"][x], param["value_max"][x], param["n_samples"]))
            combo_list = get_trackers_list(iters, param["value_min"])
            for vector in combo_list:
                tracker.append(param["var_name"])
                tracker.append(vector)
                trackers.append(tracker)
                tracker = []

        elif param["type"] == "int":
            int_list = iterations(param["value_min"], param["value_max"], param["n_samples"])
            int_list = [int(x) for x in int_list]
            for i in int_list:
                tracker.append(param["var_name"])
                tracker.append(int(i))
                trackers.append(tracker)
                tracker = []

        elif param["type"] == "float":
            float_list = iterations(param["value_min"], param["value_max"], param["n_samples"])
            for f in float_list:
                tracker.append(param["var_name"])
                tracker.append(float(f))
                trackers.append(tracker)
                tracker = []
        else:
            print("nothing")

    combis = all_combos(velocity=combo_list, num_particles=int_list, total_weight=float_list, normalise=bool_list)
    
    print(combis)
    for idx, output in enumerate(combis):
        #_track = str(output).replace("(", "_").replace(" ", "_").replace(",", "").replace(")", "")
        _track = str(output)
        print('tracker{:0>8d}{}'.format(idx, _track))
    
    #print(type(combis))
    # Construct all of the trackers with correct step increments
    #combos = set(itertools.product(combo_list, int_list, float_list, bool_list))
    #sim_dir.mkdir(exist_ok=True)
    
   
if __name__ == "__main__":
    run()
