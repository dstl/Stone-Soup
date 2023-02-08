import numpy as np
import itertools

from ..types.array import StateVector, CovarianceMatrix
from ..types.numeric import Probability


class InputManager:
    """
    The Input Manager is a component of the Run Manager that handles
    the inputs of parameters by setting the inputs to their correct
    types and generates the combinations of parameters to be ran
    in the RunManagerCore simulations.

    Parameters
    ----------

    """

    @staticmethod
    def set_stateVector(list_state_vector):
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

    @staticmethod
    def set_int(input_int):
        """
        Sets input value to int type

        Parameters:
            input_int: value to convert

        Returns:
            int: int value
        """
        return int(input_int)

    @staticmethod
    def set_float(input_float):
        """
        Set input value to float type

        Parameters:
            input_float: value to convert

        Returns:
            float: float value
        """
        return float(input_float)

    @staticmethod
    def set_covariance(list_state_covar):
        """Get a list and return a covar

        Parameters:
            list_state_covar: covar vector list

        Returns:
            CovarianceMatrix: covariance
        """
        covar_list = []
        for idx, elem in enumerate(list_state_covar):
            covariance_matrix = np.zeros((len(elem), len(elem)), dtype=int)
            np.fill_diagonal(covariance_matrix, list(elem))
            covar_list.append(CovarianceMatrix(covariance_matrix))
        return covar_list

    @staticmethod
    def set_tuple(input_list_tuple):
        """
        Set list input to tuple type

        Parameters:
            input_list_tuple: list of tuple

        Returns:
            tuple: tuple
        """
        tuple_list = []
        for idx, elem in enumerate(input_list_tuple):
            tuple_list.append(tuple(elem))
        return tuple_list

    @staticmethod
    def set_bool(input_bool):
        """
        Set bool, Not Yet Implemented

        Parameters:
            input_bool: value to convert

        Returns:
            bool: bool value
        """
        if input_bool in [0, 1, True, False]:
            return bool(input_bool)
        else:
            raise ValueError("input_bool must be of boolean type.")

    @staticmethod
    def set_ndArray(input_ndarray):
        """
        Gets an array and sets it to numpy array

        Parameters:
            input_ndarray: value to convert

        Returns:
            ndarray: ndarray value
        """
        return np.array(input_ndarray)

    @staticmethod
    def set_time_delta(input_timedelta):
        """
        Set time input to timedelta type

        Parameters:
            input_timedelta: value to convert

        Returns:
            timedelta: timedelta value
        """
        return timedelta(input_timedelta)

    @staticmethod
    def set_probability(input_probability):
        """
        Set probability, Not Yet Implemented

        Parameters:
            input_probability: value to convert

        Returns:
            Probability: probability value
        """
        return Probability(input_probability)

    def generate_parameters_combinations(self, parameters):
        """Generates all the possible combination values from a list
        of parameters with, min, max and n_samples values.

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
            # combination_list = {}
            try:
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

                if parameter["type"] == "Tuple":
                    combination_list = self.generate_tuple_list_combinations(parameter)
                    combination_dict.update(combination_list)

                if parameter["type"] == "timedelta":
                    combination_list = self.generate_timedelta_combinations(parameter)
                    combination_dict.update(combination_list)

                if parameter["type"] == "ndarray":
                    combination_list = self.generate_ndarray_combinations(parameter)
                    combination_dict.update(combination_list)
            except KeyError:
                pass
        return combination_dict

    def generate_ndarray_combinations(self, parameter):
        """Generate list of combinations to set in an ndarray

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
                iteration_list, len(parameter["value_min"])))
        return combination_list

    def generate_timedelta_combinations(self, parameter):
        """
        Generate list of timedelta combinations

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
        path = parameter["path"]
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
        # path = parameter["path"]
        combination_list = {}
        # iteration_list = []
        n_samples = parameter["n_samples"]
        n_samples_matrix = np.zeros((len(n_samples), len(n_samples)), dtype=int)
        np.fill_diagonal(n_samples_matrix, n_samples)
        min_val = parameter["value_min"]
        covar_min_array = np.zeros((len(min_val), len(min_val)), dtype=int)
        np.fill_diagonal(covar_min_array, min_val)
        max_val = parameter["value_max"]
        covar_max_array = np.zeros((len(max_val), len(max_val)), dtype=int)
        np.fill_diagonal(covar_max_array, max_val)

        if len(parameter['value_min']) > 0 and len(parameter['value_max']) > 0:
            path = parameter["path"]
            iteration_list = []
            if type(parameter["n_samples"]) is list:
                for x in range(len(parameter['value_min'])):
                    iteration_list.append(self.iterations(parameter["value_min"][x],
                                                          parameter["value_max"][x],
                                                          parameter["n_samples"][x]))
            combination_list[path] = self.set_covariance(self.get_array_list(iteration_list,
                                                         len(parameter["value_min"])))
        return combination_list

    @staticmethod
    def generate_bool_combinations(parameter):
        """Generate combinations of booleans

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
        combination_list = dict()
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
        combination_list = {}

        if len(parameter['value_min']) > 0 and len(parameter['value_max']) > 0:
            path = parameter["path"]
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
        if type(val) is list:
            for x in range(len(val)):
                new_iteration_list = []
                iteration_list.append(new_iteration_list)
                self.darray_navigator(val[x], val_min[x], val_max[x],
                                      new_iteration_list, n_samples)
        else:
            iteration_list.append(self.iterations(val_min, val_max, n_samples))

    # Calculate the steps for each item in a list
    @staticmethod
    def iterations(min_value, max_value, num_samples):
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

        Returns
        -------
        list
            list of steps required for the monte-carlo run
        """
        temp = []

        # If num_samples is 0 or less don't calculate any
        if num_samples <= 0 or min_value == max_value or num_samples is None:
            temp.append(min_value)
            if min_value != max_value:
                temp.append(max_value)
            return temp

        else:
            difference = max_value - min_value
            factor = difference / (num_samples+1)
            # Calculate n_samples different samples plus min_value and max_value
            for x in range(0, num_samples+2):
                temp.append(min_value + (x * factor))
            return temp

    @staticmethod
    def get_array_list(iterations_container_list, n):
        """Gets the combinations for one list of state vector and stores in list
           Once you have steps created from iterations, generate step combinations
           for one parameter

        Parameters:
            iterations_container_list (list): list all the possible values
            n (list): list of value min

        Returns:
            list: list all the possible combinations
        """

        array_list = []
        for x in range(0, n):
            array_list.append(iterations_container_list[x])

        list_combinations = list(itertools.product(*array_list))
        # Using a set to remove any duplicates
        set_combinations = list(set(list_combinations))
        return set_combinations

    @staticmethod
    def get_ndarray_trackers_list(iterations_container_list, n):
        """Gets the combinations for one list of ndarray and stores in list
           Once you have steps created from iterations, generate step combinations
           for one parameter

        Parameters:
            iterations_container_list (list): list all the possible values
            n (list): list of value min

        Returns:
            list: list all the possible combinations
        """
        array_list = []
        for x in range(0, n):
            array_list.append(iterations_container_list[x])

        list_combinations = [list(tup) for tup in itertools.product(*array_list)]
        # Using a set to remove any duplicates
        # set_combinations = list(set(list_combinations))

        return list_combinations

    @staticmethod
    def get_covar_trackers_list(iteration_list, n):
        """Gets the combinations for one list of ndarray and stores in list
           Once you have steps created from iterations, generate step combinations
           for one parameter

        Parameters:
            iteration_list (list): list all the possible values

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
            temp_array = np.zeros((n, n), dtype=int)
            np.fill_diagonal(temp_array, y)
            combinations.append(temp_array)
        return combinations

    # Generates all of the combinations between different parameters
    @staticmethod
    def generate_all_combos(trackers_dict):
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
