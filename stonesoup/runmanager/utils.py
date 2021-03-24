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
