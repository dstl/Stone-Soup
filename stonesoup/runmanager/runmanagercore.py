import os
import sys

from stonesoup.runmanager.parameters import Parameters
from stonesoup.serialise import YAML
from stonesoup.types.array import CovarianceMatrix
from numpy.random.mtrand import randint
import numpy as np

from stonesoup.initiator.simple import GaussianParticleInitiator, SinglePointInitiator
from stonesoup.platform.base import MovingPlatform, MultiTransitionMovingPlatform, Platform
from stonesoup.base import Base

import random
import inspect
import importlib
import pkgutil
import warnings
from stonesoup.tracker.base import Tracker
import copy
import datetime
from stonesoup.metricgenerator.ospametric import OSPAMetric
from stonesoup.measures import Euclidean
from stonesoup.metricgenerator.manager import SimpleManager
from matplotlib import pyplot as plt
import json
import itertools

# UPLOAD_FOLDER = '/path/to/the/uploads'
# JSON PATH = "C:/temp/SERAPIS/dummy2.json"
# CONFIG PATH = "C:/temp/SERAPIS/config.yaml"
path = ""
pathInput = ""


def read_json(json_input):
    with open(json_input) as json_file:
        json_data = json.load(json_file)
        # print(json.dumps(json_data, indent=4, sort_keys=True))
        return json_data


def calculateDifferencesList(minValueList, maxValueList):
    x = 0
    count = 0
    while x < len(minValueList):
        if minValueList[x] - maxValueList[x] != 0:
            count += 1
        x += 1
    return count


def calculateDifferences(minValue, maxValue):
    x = 0
    count = 0
    if minValue - maxValue != 0:
        count += 1
    return count


def iterations(minValue, maxValue, num_samples, iteration_level):
    x = 0
    returnVar = []
    difference = maxValue - minValue
    factor = difference / (num_samples - 1)
    while x < num_samples:
        returnVar.append(minValue + (x * factor))
        print(returnVar)
        x += 1
    print("RANGE VALUES CALCULATED FOR SAMPLES", num_samples, "AT ITERATION", iteration_level, "IS: ", returnVar)
    return returnVar


def getTrackersList(c, listCount, iterationsContainerList, minValues):
    iterationLength = len(minValues[c])
    finalLocation = listCount + iterationLength
    tempContainer = []
    print("listcoutn", listCount)
    print("iteration_lenth", iterationLength)
    

    #print("listCombinations", iterationsContainerList[0])
    while listCount < finalLocation:
        tempContainer.append(iterationsContainerList[listCount])
        #print("listCombinations", iterationsContainerList[listCount])
        listCount += 1
    #print("tempContainer", tempContainer)
    listCombinations = list(itertools.product(*tempContainer))
    setCombinations = list(set(listCombinations))
    print("tempContainer", setCombinations)
    # This removes duplicate rows, as sets aren't allowed to have duplicates in python
    return setCombinations

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

# Generates all of the combinations between different parameters
def generate_all_combos(trackers_dict):
    keys = trackers_dict.keys()
    values = (trackers_dict[key] for key in keys)
    combinations = [dict(zip(keys, combination)) for combination in itertools.product(*values)]
    return combinations
def run():
    # Initiating by reading in the files
    num_runs = 1  # TEMPORARY VALUE. NEEDS TO BE READ FROM JSON
    path_input = 'C:\\Users\\gbellant.LIVAD\\Documents\\Projects\\serapis\\Serapis C38 LOT 1\\dummy2.json'
    config_path = "C:\\Users\\gbellant.LIVAD\\Documents\\Projects\\serapis\\Serapis C38 LOT 1\\config.yaml"
    json_data = read_json(path_input)
    
    iters = []
    trackers_combination_dict = {}
    combo_list = {}
    int_list = {}

    for param in json_data["parameters"]:
        for key, val in param.items():
            path = param["path"]

            if type(val) is list and key == "value_min":
                for x in range(len(val)):
                    iters.append(iterations(param["value_min"][x], param["value_max"][x], param["n_samples"]))
                combo_list[path] = get_trackers_list(iters, param["value_min"])
                trackers_combination_dict.update(combo_list)

            if type(val) is int and key == "value_min":
                path = param["path"]
                int_iterations = iterations(param["value_min"], param["value_max"], param["n_samples"])
                int_list[path] = [int(x) for x in int_iterations]
                trackers_combination_dict.update(int_list)

    the_combo_dict = generate_all_combos(trackers_combination_dict)

    # Everything from this point onwards is from original runmanager
    # This current code still uses the yaml file to run the monte carlo
    from operator import attrgetter

    with open(config_path, 'r') as file:
        tracker, ground_truth, metric_manager = read_config_file(file)

    trackers = []
    ground_truths = []
    metric_managers = []

    for parameter in the_combo_dict:
        for k, v in parameter.items():
            split_path = k.split('.')
            path_param = '.'.join(split_path[1::])
            split_path =split_path[1::]
          #  print(split_path)
            tracker_copy, ground_truth_copy, metric_manager_copy = copy.deepcopy(
            (tracker, ground_truth, metric_manager))
            # setattr(tracker_copy.initiator, split_path[-1], v)
            setParam(split_path,tracker_copy,v)
           # print(tracker_copy)
        trackers.append(tracker_copy)
        ground_truths.append(ground_truth_copy)
        metric_managers.append(metric_manager_copy)
       

    for trac in trackers:
        print("\n",trac.initiator.number_particles)


def setParam(split_path,el,value):
    if len(split_path)>1:
       # print(split_path[0])
        newEl = getattr(el,split_path[0])
        setParam(split_path[1::],newEl,value)
    else:
        setattr(el,split_path[0],value)
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

# return render_template("result.html", labels=labels, values=values)

def index():
    return 'index.html'


# Set the data for the plot
def plot(metricsList, num_sims):
    metric_values = []
    time_values = []

    for metric in metricsList[0]:
        for var in metric.value:
            time_values.append(var.timestamp.strftime('%Y-%m-%d %H:%M:%S.%f'))

    for simulation in range(len(metricsList)):
        metric = []
        for m in metricsList[simulation]:
            for val in m.value:
                metric.append(val.value)

        metric_values.append(metric)
        # print(metric_values[simulation])

    return metric_values, time_values
    # for i in range(num_steps):


# Print the tree on the console
def printTree(data, treeNodes, deep=-1):
    deep = deep + 1
    print(str(data.property) + ' Deep ' + str(deep))
    deep = deep + 1
    for node in treeNodes:
        if hasattr(node, "children"):
            if len(node.children) > 0:
                printTree(node.data, node.children, deep - 1)
            else:
                print(str(node.data.property) + ' Deep ' + str(deep))


# Generate a tree
def generate_tree(object, parentNode, propertyName):
    if type(object) is not list and type(object) is not tuple:
        node: Tree = Tree(NodeData(object.__class__.__name__, object, propertyName))

        parentNode.children.append(node)

    if hasattr(object.__class__, "properties"):
        properties = object.__class__.properties
        if len(properties) > 0:
            for property in properties:
                generate_tree(getattr(object, property), node, property)

        else:
            print(object.__class__)
            parentNode.children.remove(node)
            return node
    else:
        if type(object) is list or type(object) is tuple:
            for i, el in enumerate(object):
                generate_tree(el, parentNode, propertyName + '[' + str(i) + ']')

        else:
            return node


def read_config_file(config_file):
    config_string = config_file.read()
    tracker, ground_truth, metric_manager = YAML('safe').load(config_string)
    return tracker, ground_truth, metric_manager


# Navigate inside all the tracker list given a property
def navigate_tracker(property, trackers, isSet=False, idx=0):
    newTrackers = []
    for i in range(0, len(trackers)):
        if not isSet:
            newTrackers.append(getattr(trackers[i], property))
        else:
            newTrackers.append(getattr(list(trackers[i])[idx], property))

    return newTrackers


# Navigate inside all the tracker list given a list/tuple
def navigate_tracker_list(index, trackers, isSet=False, idx=0):
    newTrackers = []
    for i in range(0, len(trackers)):
        if not isSet:
            newTrackers.append(trackers[i][index])
        else:
            newTrackers.append(list(trackers[i])[idx][index])
    return newTrackers


# Compare two tree to check if they are similar
def compare_trees(tree1, tree2, result):
    if (result == False):
        return False

    if hasattr(tree1.__class__, "properties"):
        properties = tree1.__class__.properties
        if len(properties) > 0:
            for property in properties:
                #  trackers = get_data(getattr(object,property),getattr(object_min,property),getattr(object_max,property),getattr(object_steps,property),propertyName+"."+property,request,gen_object_array(getattr(object,property),len(trackers)))
                result = compare_trees(getattr(tree1, property), getattr(tree2, property), result)
        else:
            if tree1 == tree2:
                print("TRUE")
                return True
            else:
                print(type(tree1))
                print(type(tree2))
                print("tree1 " + str(tree1) + " tree2 " + str(tree2))
                return True
    else:
        if (type(tree1) is list or type(tree1) is tuple):
            for i, el in enumerate(tree1):

                # trackers =get_data(el,object_min[i],object_max[i],object_steps[i],propertyName+'['+str(i)+']',request,gen_object_array(el,len(trackers)))
                try:
                    result = compare_trees(el, tree2[i], result)
                except:
                    print("el " + str(el) + " tree2 " + str(tree2[i]))
                    return False
        else:
            className = tree1.__class__.__name__
            if (className == "ndarray" or className == "StateVector" or className == "CovarianceMatrix"):
                if (tree1 == tree2).all():
                    print("TRUE")
                    return True
                else:
                    print("abc")
                    print(className)
                    print("tree1 " + str(tree1) + " tree2 " + str(tree2))
                    return False
            elif (
                    className == "timedelta" or className == "datetime" or className == "int" or className == "float" or className == "bool" or className == "NoneType"):
                if (tree1 == tree2):
                    return True
                else:
                    print("tree1 " + str(tree1) + " tree2 " + str(tree2))
                    return False
            else:
                print(className)
                return False
    return result


# return trackers

# Get the data from the POST request
def get_data(object, object_min, object_max, object_steps, propertyName, request, trackers, isSet=False, idx=0):
    if hasattr(object.__class__, "properties"):
        properties = object.__class__.properties
        if len(properties) > 0:
            for single_property in properties:
                new_trackers = get_data(getattr(object, single_property), getattr(object_min, single_property),
                                        getattr(object_max, single_property), getattr(object_steps, single_property),
                                        propertyName + "." + single_property, request,
                                        navigate_tracker(single_property, trackers, isSet, idx))
        else:
            return trackers
    else:
        if type(object) is list or type(object) is tuple:
            for i, el in enumerate(object):
                new_trackers = get_data(el, object_min[i], object_max[i], object_steps[i],
                                        propertyName + '[' + str(i) + ']', request, navigate_tracker_list(i, trackers))

        else:
            set_tracker_data(object, object_min, object_max, object_steps, propertyName, request, trackers)


# Set the data value inside the trackers
def set_tracker_data(object, object_min, object_max, object_steps, type, request, trackers):
    if object.__class__.__name__ == 'StateVector':
        set_state_vector(object, object_min, object_max, object_steps, type, request, trackers)
    elif object.__class__.__name__ == 'CovarianceMatrix':
        set_covar(object, object_min, object_max, object_steps, type, request, trackers)
    elif object.__class__.__name__ == 'datetime':
        set_datetimes(object, object_min, object_max, object_steps, type, request, trackers)
    elif object.__class__.__name__ == 'ndarray':
        set_nd_array(object, object_min, object_max, object_steps, type, request, trackers)
    elif (
            object.__class__.__name__ == 'int' or object.__class__.__name__ == 'float' or object.__class__.__name__ == 'NoneType' or object.__class__.__name__ == 'bool'):
        set_single_value(object, object_min, object_max, object_steps, type, request, trackers)


def set_bool(object, objectMin, objectMax, objectSteps, type, request, trackers):
    objectType = object.__class__.__name__

    object = request.form.get(type)
    objectMin = request.form.get(type + '_min_range')
    objectMax = request.form.get(type + '_max_range')
    objectSteps = request.form.get(type + '_step')

    for k in range(0, len(trackers)):
        trackers[k] = bool(object)


# STATE VECTOR
def set_state_vector(object, objectMin, objectMax, objectSteps, type, request, trackers):
    state_vector = request.form.getlist(type + '[]')
    state_vector_min = request.form.getlist(type + '_min_range[]')
    state_vector_max = request.form.getlist(type + '_max_range[]')
    state_vector_step = request.form.getlist(type + '_step[]')

    for i, val in enumerate(state_vector):
        object[i] = val
        objectMin[i] = state_vector_min[i]
        objectMax[i] = state_vector_max[i]
        objectSteps[i] = state_vector_step[i]

        for k in range(0, len(trackers)):
            trackers[k][i] = generate_random(object[i], objectMin[i], objectMax[i], objectSteps[i], k)


def set_covar(object, objectMin, objectMax, objectSteps, type, request, trackers):
    covar = request.form.getlist(type + '[]')
    covar_min = request.form.getlist(type + '_min_range[]')
    covar_max = request.form.getlist(type + '_max_range[]')
    covar_step = request.form.getlist(type + '_step[]')
    # print(covar)
    for i in range(0, len(object)):
        for j in range(0, len(object)):
            object[i, j] = covar[j + len(object) * i]
            objectMin[i, j] = covar_min[j + len(object) * i]
            objectMax[i, j] = covar_max[j + len(object) * i]
            objectSteps[i, j] = covar_step[j + len(object) * i]

            for k in range(0, len(trackers)):
                trackers[k][i, j] = generate_random(object[i, j], objectMin[i, j], objectMax[i, j], objectSteps[i, j],
                                                    k)


def set_nd_array(object, objectMin, objectMax, objectSteps, type, request, trackers):
    nd_array = request.form.getlist(type + '[]')
    nd_array_min = request.form.getlist(type + '_min_range[]')
    nd_array_max = request.form.getlist(type + '_max_range[]')
    nd_array_step = request.form.getlist(type + '_step[]')

    for i, val in enumerate(nd_array):
        object[i] = val
        objectMin[i] = nd_array_min[i]
        objectMax[i] = nd_array_max[i]
        objectSteps[i] = nd_array_step[i]


def set_single_value(object, objectMin, objectMax, objectSteps, type, request, trackers):
    objectType = object.__class__.__name__

    object = request.form.get(type)
    objectMin = request.form.get(type + '_min_range')
    objectMax = request.form.get(type + '_max_range')
    objectSteps = request.form.get(type + '_step')

    for k in range(0, len(trackers)):
        if (objectType == "int"):
            trackers[k] = int(generate_random_int(int(object), int(objectMin), int(objectMax), int(objectSteps), k))
        elif (objectType == 'float'):
            try:
                trackers[k] = float(
                    generate_random(float(object), float(objectMin), float(objectMax), float(objectSteps), k))
            except ValueError:
                trackers[k] = 0.0
        elif (objectType == 'bool'):
            trackers[k] = bool(bool(object))
        else:
            if (object == "None"):
                trackers[k] = None
            else:
                trackers[k] = object


def set_datetimes(object, objectMin, objectMax, objectSteps, type, request, trackers):
    object = request.form.get(type)
    objectMin = request.form.get(type + '_min_range')
    objectMax = request.form.get(type + '_max_range')
    objectSteps = request.form.get(type + '_step')
    for k in range(0, len(trackers)):
        if object == None:
            trackers[k] = object
        else:
            trackers[k] = datetime.datetime.strptime(object, '%Y-%m-%d %H:%M:%S.%f')


def generate_random(val, valMin, valMax, valSteps, index_run):
    if (valSteps != 0):
        val = valMin + valSteps * index_run
        if val > valMax:
            return valMax
        else:
            return float(val)
    elif valMin < valMax:
        return random.uniform(valMin, valMax)
    else:
        return float(val)


def generate_random_int(val, valMin, valMax, valSteps, index_run):
    if (valSteps != 0):
        val = valMin + valSteps * index_run
        if val > valMax:
            return valMax
        else:
            return int(val)
    elif valMin < valMax:
        return random.randint(valMin, valMax)
    else:
        return int(val)


if __name__ == "__main__":
    run()