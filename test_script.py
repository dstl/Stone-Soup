from stonesoup.runmanager.inputmanager import InputManager
import json

def read_json(json_input):
        """Read json file from directory

        Parameters
        ----------
        json_input : str
            path to json file

        Returns
        -------
        dict
            returns a dictionary of json data
        """
        with open(json_input) as json_file:
            json_data = json.load(json_file)
            return json_data

parameter_path = "C:\\Users\\Davidb1\\Documents\\Python\\data\\configs\\v1.04b\\metrics_config_parameters.json"
im = InputManager(montecarlo=0, seed=6)
parameter_data = read_json(parameter_path)
trackers_combination_dict = im.generate_parameters_combinations(
            parameter_data["parameters"]) 

print(trackers_combination_dict)



