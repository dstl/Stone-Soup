
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
        """From a list of parameters with, min, max and n_samples values
        generate all the possible values

        Parameters
        ----------
        parameters : list
            list of parameters used to calculate all the possible combinations

        Returns
        -------
        dict:
            dictionary of all the combinations
        """
        combination_dict = {}
        for parameter in parameters:
            if parameter["type"] == "StateVector":
                combination_list = self.generate_state_vector_combinations(parameter)
                combination_dict.update(combination_list)

            if parameter["type"] == "int":
                combination_list = self.generate_int_combinations(parameter)
                combination_dict.update(combination_list)

            if parameter["type"] == "float":
                combination_list = self.generate_float_combinations(parameter)
                combination_dict.update(combination_list)

            if parameter["type"] == "Probability":
                combination_list = self.generate_probability_combinations(parameter)
                combination_dict.update(combination_list)

            if parameter["type"] == 'bool':
                combination_list = self.generate_bool_combinations(parameter)
                combination_dict.update(combination_list)

            if parameter["type"] == "CovarianceMatrix":
                combination_list = self.generate_covariance_combinations(parameter)
                combination_dict.update(combination_list)

            if parameter["type"] == "DateTime":
                combination_list = self.generate_date_time_combinations(parameter)
                combination_dict.update(combination_list)

            if (parameter["type"] == "Tuple" or parameter["type"] == "list"):
                combination_list = self.generate_tuple_list_combinations(parameter)
                combination_dict.update(combination_list)

            if parameter["type"] == "timedelta":
                combination_list = self.generate_timedelta_combinations(parameter)
                combination_dict.update(combination_list)

            if parameter["type"] == "ndarray":
                combination_list = self.generate_ndarray_combinations(parameter)
                combination_dict.update(combination_list)

        return combination_dict

    def generate_ndarray_combinations(self, parameter):
        """Generate combinations of ndarray

        Parameters
        ----------
        parameter : dict
            dictionary of the parameter with value_max, value_min, n_samples and path

        Returns
        -------
        set
            set of all the possible values
        """
        path = parameter["path"]
        combination_list = {}
        iteration_list = []
        if parameter["value_min"].size > 0 and parameter["value_max"].size > 0:
            for x in range(len(parameter["value_min"])):
                iteration_list.append(self.iterations(parameter["value_min"][x],
                                                                parameter["value_max"][x],
                                                                parameter["n_samples"][x]))

            combination_list[path] = self.set_ndArray(self.get_array_list(
                                                                iteration_list,
                                                                len(parameter["value_min"])))
                                                        
        return combination_list

    def generate_timedelta_combinations(self, parameter):
        """Generate combinations of timedelta

        Parameters
        ----------
        parameter : dict
            dictionary of the parameter with value_max, value_min, n_samples and path

        Returns
        -------
        set
            set of all the possible values
        """
        path = parameter["path"]
        combination_list = {}
        iteration_list = self.iterations(parameter["value_min"],
                                                    parameter["value_max"],
                                                    parameter["n_samples"])
        combination_list[path] = [self.set_time_delta(x) for x in iteration_list]
        return combination_list

    def generate_tuple_list_combinations(self, parameter):
        """Generate combinations of tuple or list

        Parameters
        ----------
        parameter : dict
            dictionary of the parameter with value_max, value_min, n_samples and path

        Returns
        -------
        set
            set of all the possible values
        """
        path = parameter["path"]
        combination_list = {}
        iteration_list = []
        if len(parameter['value_min']) > 0 and len(parameter['value_max']) > 0:
            for x in range(len(parameter["value_min"])):
                iteration_list.append(self.iterations(parameter["value_min"][x],
                                                                parameter["value_max"][x],
                                                                parameter["n_samples"][x]))
            combination_list[path] = self.get_array_list(iteration_list,
                                                                    len(parameter["value_min"]))

            if parameter["type"] == "Tuple":
                combination_list[path] = self.set_tuple(combination_list[path])
            if parameter["type"] == "list":
                combination_list[path] = [list(i) for i in combination_list[path]]
        return combination_list

    def generate_date_time_combinations(self, parameter):
        """Generate combinations of date time

        Parameters
        ----------
        parameter : dict
            dictionary of the parameter with value_max, value_min, n_samples and path

        Returns
        -------
        set
            set of all the possible values
        """
        path= parameter["path"]
        combination_list = {}
        min_date = datetime.strptime(parameter["value_min"], '%Y-%m-%d %H:%M:%S.%f')
        max_date = datetime.strptime(parameter["value_max"], '%Y-%m-%d %H:%M:%S.%f')
        iteration_list = self.iterations(min_date, max_date, parameter["n_samples"])
        combination_list[path] = iteration_list
        return combination_list


    def generate_covariance_combinations(self, parameter):
        """Generate combinations of covariance matrix

        Parameters
        ----------
        parameter : dict
            dictionary of the parameter with value_max, value_min, n_samples and path

        Returns
        -------
        set
            set of all the possible values
        """
        path = parameter["path"]
        combination_list = {}
        iteration_list = []

        if (type(parameter["value_min"]) is  list and
                type(parameter["value_max"]) is  list):
            covar_min = CovarianceMatrix(parameter["value_min"])
            covar_max = CovarianceMatrix(parameter["value_max"])
            covar_diag_min = covar_min.diagonal()
            covar_diag_max = covar_max.diagonal()
            if type(parameter["n_samples"]) is list:
                #Check if we have a list of list
                for x in range(len(parameter["value_min"])):
                    iteration_list.append(self.iterations(covar_diag_min[x],
                                                            covar_diag_max[x],
                                                            parameter["n_samples"][x][x]))
            else:
                iteration_list.append(self.iterations(covar_diag_min[x],
                                                        covar_diag_max[x],
                                                        parameter["n_samples"]))
            combination_list[path] = self.get_covar_trackers_list(iteration_list,
                                                                    len(covar_min))
        return combination_list
    
    def generate_bool_combinations(self, parameter):
        """Generate combinations of bool 

        Parameters
        ----------
        parameter : dict
            dictionary of the parameter with value_max, value_min, n_samples and path

        Returns
        -------
        set
            set of all the possible values
        """
        path = parameter["path"]
        combination_list = {}
        combination_list[path] = [True, False]
        return combination_list

    def generate_probability_combinations(self, parameter):
        """Generate combinations of probability

        Parameters
        ----------
        parameter : dict
            dictionary of the parameter with value_max, value_min, n_samples and path

        Returns
        -------
        set
            set of all the possible values
        """
        path = parameter["path"]
        combination_list = {}
        iteration_list = self.iterations(parameter["value_min"],
                                                    parameter["value_max"],
                                                    parameter["n_samples"])
        combination_list[path] = [Probability(x) for x in iteration_list]
        return combination_list

    def generate_float_combinations(self, parameter):
        """Generate combinations of float

        Parameters
        ----------
        parameter : dict
            dictionary of the parameter with value_max, value_min, n_samples and path

        Returns
        -------
        set
            set of all the possible values
        """
        path = parameter["path"]
        combination_list = {}
        iteration_list = self.iterations(parameter["value_min"],
                                                    parameter["value_max"],
                                                    parameter["n_samples"])
        combination_list[path] = [float(x) for x in iteration_list]
        return combination_list

    def generate_int_combinations(self, parameter):
        """Generate combinations of int

        Parameters
        ----------
        parameter : dict
            dictionary of the parameter with value_max, value_min, n_samples and path

        Returns
        -------
        set
            set of all the possible values
        """
        path = parameter["path"]
        combination_list = {}
                # iteration_list = []
        iteration_list = self.iterations(parameter["value_min"],
                                                    parameter["value_max"],
                                                    parameter["n_samples"])
        combination_list[path] = [int(x) for x in iteration_list]
        return combination_list

    def generate_state_vector_combinations(self, parameter):
        """Generate combinations of state vector

        Parameters
        ----------
        parameter : dict
            dictionary of the parameter with value_max, value_min, n_samples and path

        Returns
        -------
        set
            set of all the possible values
        """
        if len(parameter['value_min']) > 0 and len(parameter['value_max']) > 0:
            path = parameter["path"]
            combination_list = {}
            iteration_list = []
            if type(parameter["n_samples"]) is list:
                for x in range(len(parameter['value_min'])):
                    iteration_list.append(self.iterations(parameter["value_min"][x],
                                                                    parameter["value_max"][x],
                                                                    parameter["n_samples"][x]))
            else:
                for x in range(parameter['value_min']):
                    iteration_list.append(self.iterations(parameter["value_min"][x],
                                                                    parameter["value_max"][x],
                                                                    parameter["n_samples"]))

            combination_list[path] = self.set_stateVector(
                        self.get_array_list(iteration_list, len(parameter["value_min"])))
                
        return combination_list

    def darray_navigator(self, val, val_min, val_max, iteration_list, n_samples):
        """Not used at the moment. Navigate inside the ndarray with a n depth and
        calculate all the iterations

        Parameters
        ----------
        val : [type]
            [description]
        val_min : [type]
            [description]
        val_max : [type]
            [description]
        iteration_list : [type]
            [description]
        n_samples : int
            number of parameter combinations
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
        """Calculates the step different between the min
            and max value given in the parameter file.
            If n_samples is 0 return 1 value, if it is >=1 return num_samples+2 values

        Parameters
        ----------
        min_value : Object
            Minimum parameter value
        max_value : Object
            Minimum parameter value
        num_samples : int
            number of parameter samples to calculate
        index : int, optional
            [description], by default 0

        Returns
        -------
        list
            list of steps required for the monte-carlo run
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
