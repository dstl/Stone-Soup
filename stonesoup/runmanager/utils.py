from numpy.random.mtrand import randint
import numpy as np
import time
from datetime import datetime,timedelta
import matplotlib.pyplot as plt

def generate_covar(covarList,length):
    covar = np.zeros((length,length))

    for i in range (0,length):
        for j in range(0,length):
            covar[i,j] = covarList[i*length+j]
    
    return covar


def generate_random_array(min_value,max_value, n_values):
    array = np.zeros((n_values))
    actualValue = min_value
    for i in range(0,n_values):
        if(min_value[i]<max_value[i]):
            array[i]=(randint(min_value[i],max_value[i]))
        else:
             array[i]=0

    return array


def generate_random_covar(min_value,max_value,n_values):
    matrix=np.zeros((n_values,n_values))

    for i in range(0,n_values):
        for j in range(0,n_values):
            if(min_value[0][i,j]<max_value[0][i,j]):
                matrix[i,j]=randint(min_value[0][i,j],max_value[0][i,j])
    
    return matrix



def generate_random_int(min_value,max_value):
    value = 0
    if min_value<max_value:
        value=randint(min_value,max_value)

    return value 


class Tree:
    def __init__(self, data):
        self.children = []
        self.data:nodeData =data


class NodeData:
    def __init__(self,type,value,property):
        self.type = type
        self.value = value 
        self.property = property



def TicTocGen():
    ti =0
    tf = time.time()
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti
TicToc = TicTocGen()

def toc(tempBool=True):
    tempTimeInterval = next(TicToc)
    if tempBool:
        print(tempTimeInterval)
def tic():
    toc(False)
    
def plot_ospa_gospa_for_multi_run(multi_run_metrics, start_time, num_sims,num_steps):
    ospa_values = []
    time_values = [start_time+ timedelta(seconds=i) for i in range(num_steps)]
    for simulation in range(num_sims):
        ospa_values.append([m.value for m in multi_run_metrics["OSPA distances"][simulation]])
    plotting_ospa_mean = []
    plotting_ospa_sd = []
    for i in range(num_steps):
        plotting_ospa_mean.append(np.mean([m[i] for m in ospa_values]))
        plotting_ospa_sd.append(np.sqrt(np.var([m[i] for m in ospa_values])))
        gospa_values = []
    time_values = [start_time + timedelta(seconds=i) for i in range(num_steps)]
    for simulation in range(num_sims):
        gospa_values.append([m.value for m in multi_run_metrics["GOSPA Metrics"][simulation]])
    plotting_gospa_mean = []
    plotting_gospa_sd = []
    for i in range(num_steps):
        plotting_gospa_mean.append(np.mean([m[i]["distance"] for m in gospa_values]))
        plotting_gospa_sd.append(np.sqrt(np.var([m[i]["distance"] for m in gospa_values])))
    plt.figure(figsize=(15,9))
    plt.plot(time_values, plotting_ospa_mean, "-b", label = "OSPA")
    plt.plot(time_values, plotting_gospa_mean, "-r", label = "GOSPA")
    plt.fill_between(time_values, 
                     [m-1.96*s for m, s in zip(plotting_ospa_mean, plotting_ospa_sd)], 
                     [m+1.96*s for m, s in zip(plotting_ospa_mean, plotting_ospa_sd)], 
                     color= "blue", alpha = 0.2)
    plt.fill_between(time_values, 
                     [m-1.96*s for m, s in zip(plotting_gospa_mean, plotting_gospa_sd)], 
                     [m+1.96*s for m, s in zip(plotting_gospa_mean, plotting_gospa_sd)], 
                     color= "red", alpha = 0.2)
    plt.xticks(rotation=90)
    plt.legend(loc="lower right")