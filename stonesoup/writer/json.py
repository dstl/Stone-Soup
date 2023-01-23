import json


class JSONParameterWriter:
    def __init__(self):
        self.json_parameters = dict()

    def add_configuration(self, proc_num=1, runs_num=10, **kwargs):
        configuration = dict()
        configuration['proc_num'] = proc_num
        configuration['runs_num'] = runs_num
        self.json_parameters['configuration'] = configuration
        if kwargs:
            for k in kwargs.keys():
                configuration[k] = kwargs[k]

    def add_parameter(self, number_of_samples, object_path, sample_type, value_min, value_max,
                      **kwargs):
        if "parameters" not in self.json_parameters:
            self.json_parameters['parameters'] = []

        parameter = dict()
        parameter['path'] = object_path
        parameter['type'] = sample_type
        parameter['value_min'] = value_min
        parameter['value_max'] = value_max
        if kwargs:
            for k in kwargs.keys():
                parameter[k] = kwargs[k]

        if sample_type == "StateVector":
            if len(value_min) != len(value_max):
                raise ValueError("Minimum and Maximum arrays must be the same dimension")
            if type(number_of_samples) is int:
                parameter['n_samples'] = [number_of_samples - 2] * len(value_min)
            else:
                parameter['n_samples'] = number_of_samples
        else:
            parameter['n_samples'] = number_of_samples - 2

        self.json_parameters['parameters'].append(parameter)

    def write(self, path, indent=4):
        with open(path, "w") as outfile:
            json.dump(self.json_parameters, outfile, indent=indent)
