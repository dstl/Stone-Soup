#!/usr/bin/env python
# coding: utf-8
import gzip
import sys
from multiprocessing import Pool
from multiprocessing import freeze_support

from stonesoup.serialise import YAML
import time
from stonesoup.types.array import StateVector
import numpy as np
from numpy.random.mtrand import randint
from itertools import product
from stonesoup.types.metric import PlottingMetric


def metric_generator(args):#(config_string,matrix):
    config_string = args[0]
    matrix = args[1] 
   ## config_string,matrix = args[1:]
    tracker, ground_truth, metric_manager = YAML('safe').load(config_string)
    #print(tracker.initiator.initiator.prior_state)

 #   randVector = np.array([[randint(matrix[0][0],matrix[0][1])],[randint(matrix[1][0],matrix[1][1])],[randint(matrix[2][0],matrix[2][1])],[randint(matrix[3][0],matrix[3][1])],[randint(matrix[4][0],matrix[4][1])],[randint(matrix[5][0],matrix[5][1])]])
#  tracker.initiator.initiator.prior_state.state_vector = StateVector(randVector)
    #print("state vector ")
    print(tracker.initiator.initiator.prior_state.state_vector)

    try:
        tracks = set()

        for n, (time, ctracks) in enumerate(tracker, 1):
            tracks.update(ctracks)

        metric_manager.add_data(ground_truth, tracks)
        metrics = metric_manager.generate_metrics()
    except Exception as e:
        print(f'Failure: {e}', flush=True)
        return None
    else:
        print('Success!', flush=True)
    
    for val in metrics: 
        print("metric")
        for metric in val.value:
            print(metric.value)


    return YAML().dumps(metrics)






def start_run(matrix,config_file, num_runs, num_processes, filename):
#if __name__ == '__main__':
  #  freeze_support()
    #config_file, num_runs, num_processes, filename = sys.argv[1:]
    with open(config_file, 'r') as file:
        config_string = file.read()
    
    with Pool(processes=int(num_processes)) as pool:
        configs = (config_string for _ in range(int(num_runs)))
        matrixRange = (matrix for _ in range(int(num_runs)))

        rangeCM = ([config_string,matrix] for _ in range(int(num_runs)))
     #   print(matrixRange)

        collated_metrics = (
            YAML('safe').load(metric)
            for metric in pool.imap_unordered(metric_generator, rangeCM)
                )
        
        with gzip.open(filename, 'wt') as file:
            YAML().dump_all(collated_metrics, file)

