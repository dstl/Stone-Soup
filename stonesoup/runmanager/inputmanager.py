
from .base import RunManager
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
        vector_list = []
        for idx, elem in enumerate(list_state_vector):
            vector_list.append(StateVector(elem))
        return vector_list

    def set_int(self, input_int):
        """
        Set int

        Parameters:
            input_int: value to convert

        Returns:
            int: int value
        """
        input_int = int(input_int)
        return input_int

    def set_float(self, input_float):
        """
        Set float

        Parameters:
            input_float: value to convert

        Returns:
            float: float value
        """
        input_float = float(input_float)
        return input_float

    def set_covariance(self, covar):
        """Get a list and return a covar

        Parameters:
            list_state_covar: covar vector list

        Returns:
            CovarianceMatrix: covariance
        """
        covar = np.array(covar)
        return covar

    def set_tuple(self, list_tuple):
        """
        Set tuple

        Parameters:
            input_tuple: list of tuple

        Returns:
            tuple: tuple
        """
        tuple_list = []
        for idx, elem in enumerate(list_tuple):
            tuple_list.append(tuple(elem))
        return tuple_list

    def set_bool():
        """
        Set bool

        Parameters:
            input_bool: value to convert

        Returns:
            bool: bool value
        """
        raise NotImplementedError

    def set_ndArray(self, arr):
        """
        Set ndArray

        Parameters:
            input_ndarray: value to convert

        Returns:
            ndarray: ndarray value
        """
        return np.array(arr)

    def set_time_delta(self, time_delta):
        """
        Set timedelta

        Parameters:
            input_timedelta: value to convert

        Returns:
            timedelta: timedelta value
        """
        return timedelta(time_delta)

    def set_probability():
        """
        Set probability

        Parameters:
            input_probability: value to convert

        Returns:
            Probability: probability value
        """
        raise NotImplementedError

    def generate_parameters_combinations(self, parameters):
        """[summary]
        From a list of parameters with, min, max and n_samples values
        generate all the possible values

        Parameters:
            parameters ([type]): [list of parameters used to calculate
                                  all the possible parameters]

        Returns:
            [dict]: [dictionary of all the combinations]
        """
        combination_dict = {}
        for param in parameters:
            for key, val in param.items():
                path = param["path"]
                combination_list = {}
                iteration_list = []
                if param["type"] == "StateVector" and key == "value_min":
                    if len(param['value_min']) > 0 and len(param['value_max']) > 0:
                        if type(param["n_samples"]) is list:
                            for x in range(len(val)):
                                iteration_list.append(self.iterations(param["value_min"][x],
                                                                      param["value_max"][x],
                                                                      param["n_samples"][x]))
                        else:
                            for x in range(len(val)):
                                iteration_list.append(self.iterations(param["value_min"][x],
                                                                      param["value_max"][x],
                                                                      param["n_samples"]))

                        combination_list[path] = self.set_stateVector(
                            self.get_array_list(iteration_list, len(param["value_min"])))

                    combination_dict.update(combination_list)

                if param["type"] == "int" and key == "value_min":
                    iteration_list = self.iterations(param["value_min"],
                                                     param["value_max"],
                                                     param["n_samples"])
                    combination_list[path] = [int(x) for x in iteration_list]
                    combination_dict.update(combination_list)

                if param["type"] == "float" and key == "value_min":
                    iteration_list = self.iterations(param["value_min"],
                                                     param["value_max"],
                                                     param["n_samples"])
                    combination_list[path] = [float(x) for x in iteration_list]
                    combination_dict.update(combination_list)

                if param["type"] == "Probability" and key == "value_min":
                    iteration_list = self.iterations(param["value_min"],
                                                     param["value_max"],
                                                     param["n_samples"])
                    combination_list[path] = [Probability(x) for x in iteration_list]
                    combination_dict.update(combination_list)

                if param["type"] == 'bool' and key == "value_min":
                    combination_list[path] = [True, False]
                    combination_dict.update(combination_list)

                if param["type"] == "CovarianceMatrix" and key == "value_min":
                    if (type(param["value_min"]) is not list and
                            type(param["value_max"]) is not list):
                        break
                    if len(param["value_min"]) > 0 and len(param["value_max"]) > 0:
                        covar_min = CovarianceMatrix(param["value_min"])
                        covar_max = CovarianceMatrix(param["value_max"])
                        covar_diag_min = covar_min.diagonal()
                        covar_diag_max = covar_max.diagonal()
                        if type(param["n_samples"]) is list:
                            for x in range(len(val)):
                                iteration_list.append(self.iterations(covar_diag_min[x],
                                                                      covar_diag_max[x],
                                                                      param["n_samples"][x][x]))
                        else:
                            iteration_list.append(self.iterations(covar_diag_min[x],
                                                                  covar_diag_max[x],
                                                                  param["n_samples"]))
                        combination_list[path] = self.get_covar_trackers_list(iteration_list,
                                                                              len(covar_min))
                        combination_dict.update(combination_list)

                if param["type"] == "DateTime" and key == "value_min":
                    min_date = datetime.strptime(param["value_min"], '%Y-%m-%d %H:%M:%S.%f')
                    max_date = datetime.strptime(param["value_max"], '%Y-%m-%d %H:%M:%S.%f')
                    iteration_list = self.iterations(min_date, max_date, param["n_samples"])
                    combination_list[path] = iteration_list
                    combination_dict.update(combination_list)

                if (param["type"] == "Tuple" or param["type"] == "list") and key == "value_min":
                    if len(param['value_min']) > 0 and len(param['value_max']) > 0:
                        for x in range(len(val)):
                            iteration_list.append(self.iterations(param["value_min"][x],
                                                                  param["value_max"][x],
                                                                  param["n_samples"][x]))
                        combination_list[path] = self.get_array_list(iteration_list,
                                                                     len(param["value_min"]))

                        if param["type"] == "Tuple":
                            combination_list[path] = self.set_tuple(combination_list[path])
                        if param["type"] == "list":
                            combination_list[path] = [list(i) for i in combination_list[path]]

                    combination_dict.update(combination_list)

                if param["type"] == "timedelta" and key == "value_min":
                    iteration_list = self.iterations(param["value_min"],
                                                     param["value_max"],
                                                     param["n_samples"])
                    combination_list[path] = [self.set_time_delta(x) for x in iteration_list]
                    combination_dict.update(combination_list)

                if param["type"] == "ndarray" and key == "value_min":
                    if param["value_min"].size > 0 and param["value_max"].size > 0:

                        for x in range(len(val)):
                            iteration_list.append(self.iterations(param["value_min"][x],
                                                                  param["value_max"][x],
                                                                  param["n_samples"][x]))

                        combination_list[path] = self.set_ndArray(self.get_array_list(
                                                                    iteration_list,
                                                                    len(param["value_min"])))

                        combination_dict.update(combination_list)

        return combination_dict

    def darray_navigator(self, val, val_min, val_max, iteration_list, n_samples):
        """Not used at the moment. Navigate inside the ndarray with a n depth and
        calculate all the iterations

        Args:
            val ([type]): [description]
            val_min ([type]): [description]
            val_max ([type]): [description]
            iteration_list ([type]): [description]
            n_samples ([type]): [description]
        """
        if(type(val) is list):
            for x in range(len(val)):
                new_iteration_list = []
                iteration_list.append(new_iteration_list)
                self.darray_navigator(val[x], val_min[x], val_max[x],
                                      new_iteration_list, n_samples)
        else:
            iteration_list.append(self.iterations(val_min, val_max, n_samples))

    # Calculate the steps for each item in a list
    def iterations(self, min_value, max_value, num_samples, index=0):
        """ Calculates the step different between the min
            and max value given in the parameter file.
            If n_samples is 0 return 1 value, if it is >=1 return num_samples+2 values
        Args:
            self : self
            min_value : Minimum parameter value
            maz_value : Maximum parameter value
        """
        temp = []

        # If num_samples is 0 or less don't calculate any
        if num_samples <= 0 or min_value == max_value or num_samples is None:
            temp.append(min_value)
            return temp

        else:
            difference = max_value - min_value
            factor = difference / (num_samples+1)
            # Calculate n_samples different samples plus min_value and max_value
            for x in range(0, num_samples+2):
                temp.append(min_value + (x * factor))
            return temp

    def get_array_list(self, iterations_container_list, n):
        """Gets the combinations for one list of state vector and stores in list
           Once you have steps created from iterations, generate step combinations
           for one parameter

        Parameters:
            iterations_container_list (list): list all the possible values
            n (list): list of value min

        Returns:
            list: list all the possible combinations
        """

        temp = []
        for x in range(0, n):
            temp.append(iterations_container_list[x])

        list_combinations = list(itertools.product(*temp))
        # Using a set to remove any duplicates
        set_combinations = list(set(list_combinations))
        return set_combinations

    def get_ndarray_trackers_list(self, iterations_container_list, n):
        """Gets the combinations for one list of ndarray and stores in list
           Once you have steps created from iterations, generate step combinations
           for one parameter

        Parameters:
            iterations_container_list (list): list all the possible values
            n (list): list of value min

        Returns:
            list: list all the possible combinations
        """
        temp = []
        for x in range(0, n):
            temp.append(iterations_container_list[x])

        list_combinations = [list(tup) for tup in itertools.product(*temp)]
        # Using a set to remove any duplicates
        # set_combinations = list(set(list_combinations))

        return list_combinations

    def get_covar_trackers_list(self, iteration_list, n):
        """Gets the combinations for one list of ndarray and stores in list
           Once you have steps created from iterations, generate step combinations
           for one parameter

        Parameters:
            iteration_list (list): list all the possible values
            value_min ([type]): [description]

        Returns:
            list: list all the possible combinations
        """
        temp = []
        combinations = []
        for x in range(0, n):
            temp.append(iteration_list[x])
        list_combinations = list(itertools.product(*temp))
        set_combinations = np.array(list(set(list_combinations)))
        for y in set_combinations:
            temp_array = np.empty((n, n), dtype=int)
            np.fill_diagonal(temp_array, y)
            combinations.append(temp_array)
        return combinations

    # Generates all of the combinations between different parameters
    def generate_all_combos(self, trackers_dict):
        """Generates all of the combinations between different parameters

        Parameters:
            trackers_dict (dict): Dictionary of all the parameters with all the possible values

        Returns:
            dict: Dictionary of all the parameters combined each other
        """
        keys = trackers_dict.keys()
        values = (trackers_dict[key] for key in keys)
        combinations = [dict(zip(keys, combination)) for combination in itertools.product(*values)]
        return combinations
