import itertools
import json
from os import path
from pathlib import Path
from itertools import product, starmap
from collections import namedtuple

def read_json(json_input):
    with open(json_input) as json_file:
        json_data = json.load(json_file)
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
    return set_combinations

# Returns the product of all combinations from the trackers list
# Generates all of the combinations for n parameters
def all_combos(**items):
    Combo = namedtuple('_params', items.keys())
    return starmap(Combo, product(*items.values()))

def generate_all_combos(trackers_dict):
    keys = trackers_dict.keys()
    values = (trackers_dict[key] for key in keys)
    combinations = [dict(zip(keys, combination)) for combination in itertools.product(*values)]
    return combinations

def run():
    # Initiating by reading in the files
    # pathInput = input("Input json directory: ")
    # path = input("Input config yaml directory: ")
    pathInput = 'C:\\Users\\Davidb1\\Documents\\Python\\data\\dummy2.json'
    json_data = read_json(pathInput)
    sim_dir = Path('sims/')
    #trackers = []
    tracker = []
    iters = []
    trackers = {}
    values = []
    combo_list = {}
    int_list = {}
    #all_trackers = {}
    # For each parameter in the json, calculate the increments of each step.
    # Essentially, for each parameter calculate the steps and iterations, add to tracker list
    # then combine all to generate combinations.
    for param in json_data["parameters"]:
        for key, val in param.items():
            path = param["path"]

            if type(val) is list and key == "value_min":
                for x in range(len(val)):
                    iters.append(iterations(param["value_min"][x], param["value_max"][x], param["n_samples"]))
                combo_list[path] = get_trackers_list(iters, param["value_min"])
                trackers.update(combo_list)

            if type(val) is int and key == "value_min":
                path = param["path"]
                int_iterations = iterations(param["value_min"], param["value_max"], param["n_samples"])
                int_list[path] = [int(x) for x in int_iterations]
                trackers.update(int_list)

    print(generate_all_combos(trackers))

"""
    for i in json_data["parameters"]:
        print ("\n", i) """


   
if __name__ == "__main__":
    run()
