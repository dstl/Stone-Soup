from ruamel.yaml import YAML


def _get_yaml_scenario(
    scenario_config="ReinforcementLearning/configs/scenario_config.yaml",
):
    '''
    Function takes the path of a yaml file as an arguement and returns its
    constents as a scenario dictionary.
    '''

    stonesoup_yaml = YAML(typ=["rt", "stonesoup"], plug_ins=["stonesoup.serialise"])

    with open(scenario_config, "r") as file:
        data = stonesoup_yaml.load_all(file)
        scenario = data.__next__()

    return scenario
