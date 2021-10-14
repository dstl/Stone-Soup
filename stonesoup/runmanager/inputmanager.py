from typing import Tuple
from base import RunManager
import numpy as np
import itertools
from stonesoup.types.array import StateVector, CovarianceMatrix
from stonesoup.types.numeric import Probability
from datetime import datetime, timedelta
class InputManager(RunManager):

    def set_stateVector(self, list_state_vector):
        """Get a list and return a state vector

        Parameters:
            list_state_vector: State vector list

        Returns:
            StateVector: state vector 
        """
        vector_list=[]
        for idx, elem in enumerate(list_state_vector):
            vector_list.append(StateVector(elem))
        return vector_list

    def set_int(self, input_int):
        input_int=int(input_int)
        return input_int
    
    def set_float(self, input_float):
        input_float=float(input_float)
        return input_float

    def set_covariance(self, covar):
        covar=np.array(covar)
        return covar

    def set_tuple(self, list_tuple):
        tuple_list=[]
        for idx, elem in enumerate(list_tuple):
            tuple_list.append(tuple(elem))
        return tuple_list

    def set_bool():
        raise NotImplementedError

    def set_ndArray():
        raise NotImplementedError

    def set_time_delta(self, time_delta):
        return timedelta(time_delta)
    
    def set_coordinate_system():
        raise NotImplementedError

    def set_probability():
        raise NotImplementedError

    def generate_parameters_combinations(self, parameters):
        """[summary]
        From a list of parameters with, min, max and n_samples values generate all the possible values

        Args:
            parameters ([type]): [list of parameters used to calculate all the possible parameters]

        Returns:
            [dict]: [dictionary of all the combinations]
        """
        combination_dict = {}

        for param in parameters:
            for key, val in param.items():
                path = param["path"]
                combination_list = {}
                iteration_list=[]
                if param["type"] == "StateVector" and key == "value_min":
                    for x in range(len(val)):
                        iteration_list.append(self.iterations(param["value_min"][x], param["value_max"][x], param["n_samples"]))
                    print("STATE VECTOR ", iteration_list)
                    combination_list[path] = self.set_stateVector(self.get_trackers_list(iteration_list, param["value_min"]))
                    combination_dict.update(combination_list)

                if param["type"] == "int" and key == "value_min":
                    iteration_list = self.iterations(param["value_min"], param["value_max"], param["n_samples"])
                    combination_list[path] = [int(x) for x in iteration_list]
                    combination_dict.update(combination_list)

                if param["type"] == "float" and key == "value_min":
                    iteration_list = self.iterations(param["value_min"], param["value_max"], param["n_samples"])
                    combination_list[path] = [float(x) for x in iteration_list]
                    combination_dict.update(combination_list)

                if param["type"] == "Probability" and key == "value_min":
                    iteration_list = self.iterations(param["value_min"], param["value_max"], param["n_samples"])
                    combination_list[path] = [Probability(x) for x in iteration_list]
                    combination_dict.update(combination_list)

                if param["type"] == 'bool' and key == "value_min":
                    combination_list[path] = [True, False]
                    combination_dict.update(combination_list)

                if param["type"] == "CovarianceMatrix" and key == "value_min":
                    covar_min=CovarianceMatrix(param["value_min"])
                    covar_max=CovarianceMatrix(param["value_max"])
                    covar_diag_min=covar_min.diagonal()
                    covar_diag_max=covar_max.diagonal()
                    
                    for x in range(len(val)):
                        iteration_list.append(self.iterations(covar_diag_min[x], covar_diag_max[x], param["n_samples"]))
                    combination_list[path]=self.get_covar_trackers_list(iteration_list, covar_min)
                    combination_dict.update(combination_list)

                if param["type"] == "DateTime" and key == "value_min":
                    min_date=datetime.strptime(param["value_min"], '%Y-%m-%d %H:%M:%S.%f')
                    max_date=datetime.strptime(param["value_max"], '%Y-%m-%d %H:%M:%S.%f')                  
                    iteration_list = self.iterations(min_date, max_date, param["n_samples"])
                    combination_list[path]=iteration_list
                    combination_dict.update(combination_list)
                
                if param["type"] == "Tuple" and key == "value_min":
                    for x in range(len(val)):
                        iteration_list.append(self.iterations(param["value_min"][x], param["value_max"][x], param["n_samples"]))
                    combination_list[path] = self.set_tuple(self.get_trackers_list(iteration_list, param["value_min"]))
                    combination_dict.update(combination_list)

                if param["type"] == "timedelta" and key == "value_min":
                    iteration_list = self.iterations(param["value_min"], param["value_max"], param["n_samples"])
                    combination_list[path] = [self.set_time_delta(x) for x in iteration_list]
                    combination_dict.update(combination_list)
                
                if param["type"] == "ndarray" and key == "value_min":
                    val_min=param["value_min"]
                    val_max=param["value_max"]
                    # for x in range(len(val)):
                    #      for y in range(len(val[x])):
                    #         iteration_list.append(self.iterations(val_min[x][y], val_max[x][y], param["n_samples"]))
                    self.darray_navigator(val, val_min, val_max, iteration_list,param["n_samples"])
                    print("iteration_list")
                    print(iteration_list)
                    combination_list[path] = self.get_ndarray_trackers_list(iteration_list, param["value_min"])
                    combination_dict.update(combination_list)
                    print("combination_list ",combination_dict)
                

        return combination_dict



    def darray_navigator(self,val,val_min,val_max,iteration_list,n_samples):
        if(type(val) is list):
            for x in range(len(val)):
                new_iteration_list = []
                iteration_list.append(new_iteration_list)
                self.darray_navigator(val[x],val_min[x],val_max[x],new_iteration_list,n_samples)
        else:
             iteration_list.append(self.iterations(val_min, val_max, n_samples))
        

    # Calculate the steps for each item in a list
    def iterations(self, min_value, max_value, num_samples, index=0):
        """ Calculates the step different between the min 
            and max value given in the parameter file.
        Args:
            self : self
            min_value : Minimum parameter value
            maz_value : Maximum parameter value
        """
        temp = []
        difference = max_value - min_value
        factor = difference / (num_samples - 1)
        for x in range(num_samples):
            temp.append(min_value + (x * factor))
        return temp

    # gets the combinations for one tracker and stores in list
    # Once you have steps created from iterations, generate step combinations for one parameter
    def get_trackers_list(self, iterations_container_list, value_min):
        temp =[]
        for x in range(0, len(value_min)):
            temp.append(iterations_container_list[x])
        list_combinations = list(itertools.product(*temp))

        #Using a set to remove any duplicates
        set_combinations = list(set(list_combinations))
                
        return set_combinations

    def get_ndarray_trackers_list(self, iterations_container_list, value_min):
        temp =[]
        for x in range(0, len(value_min)):
            temp.append(iterations_container_list[x])
        list_combinations = [list(tup) for tup in itertools.product(*temp)]

        #Using a set to remove any duplicates
        #set_combinations = list(set(list_combinations))
                
        return list_combinations

    def get_covar_trackers_list(self, iteration_list, value_min):
        temp =[]
        combinations = []
        array_size=len(value_min)
        for x in range(0, len(value_min)):
            temp.append(iteration_list[x])
        list_combinations = list(itertools.product(*temp))
        set_combinations = np.array(list(set(list_combinations)))
        for y in set_combinations:
            temp_array=np.empty((array_size,array_size), dtype=int)
            np.fill_diagonal(temp_array,y)
            combinations.append(temp_array)
        return combinations

    # Generates all of the combinations between different parameters
    def generate_all_combos(self, trackers_dict):
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
  
