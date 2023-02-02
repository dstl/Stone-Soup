import json
from pathlib import Path


class JSONParameterWriter:
    def __init__(self, path):
        if not isinstance(path, Path):
            path = Path(path)
        self._file = path.open('w')
        self.json_parameters = dict()

    def add_configuration(self, proc_num=1, runs_num=10, **kwargs):
        configuration = dict()
        configuration['proc_num'] = proc_num
        configuration['runs_num'] = runs_num
        self.json_parameters['configuration'] = configuration

    def add_parameter(self, number_of_samples, object_path, sample_type, value_min, value_max,
                      **kwargs):
        if "parameters" not in self.json_parameters:
            self.json_parameters['parameters'] = []

        parameter = dict()
        parameter['path'] = object_path
        parameter['type'] = sample_type
        parameter['value_min'] = value_min
        parameter['value_max'] = value_max

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

    def write(self, indent=4):
        json.dump(self.json_parameters, self._file, indent=indent)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if getattr(self, '_file', None):
            self._file.close()

    def __del__(self):
        self.__exit__()
