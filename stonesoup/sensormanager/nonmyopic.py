import copy
import itertools as it
import numpy as np

from stonesoup.sensor.sensor import Sensor
from stonesoup.sensormanager import SensorManager


# Tree Structure
from anytree import NodeMixin

# Create a wrapper around the configs that allow it to be used in the tree structure
class ConfigWrapper(NodeMixin):
    def __init__(self, config, parent = None, children = None):
        self.config = config
        self.parent = parent
        if children:
            self.children = children

class BruteForceNonMyopicSensorManager(SensorManager):
    '''
    Non-myopic Brute Force Sensor Manager - more detail coming!
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # defaults for horizons are myopic sensor management
        self.set_horizons()


    def set_horizons(self, evaluation_horizon=1, schedule_horizon=1):
        self.evaluation_horizon = evaluation_horizon # how many steps do we look into the future
        if schedule_horizon > evaluation_horizon:
            raise ValueError('Schedule horizon must be less than or equal to the evaluation horizon.')
        else: self.schedule_horizon = schedule_horizon # how many steps do we schedule into the future (must be leq evaluation_horizon)


    def get_future_timesteps(self, current_timestep, all_timesteps):
        idx = all_timesteps.index(current_timestep)
        try:
            future_timesteps = [all_timesteps[i] for i in range(idx, idx + self.evaluation_horizon)]
        except IndexError:
            future_timesteps = all_timesteps[idx:]
        return future_timesteps


    def choose_actions(self, tracks, timestamps, nchoose=1, **kwargs):

        def get_action_choices(sensors, timestamp):
            all_action_choices = dict()
            for sensor in sensors:
                # get action 'generator(s)'
                action_generators = sensor.actions(timestamp)
                # list possible action combinations for the sensor
                action_choices = list(it.product(*action_generators))
                # dictionary of sensors: list(action combinations)
                all_action_choices[sensor] = action_choices
            return all_action_choices

        def evaluate_reward_function(self, configs, tracks, timestamp, nchoose):
            best_rewards = np.zeros(nchoose) - np.inf
            selected_configs = [None] * nchoose
            for config in configs:
                # calculate reward for dictionary of sensors: actions
                reward = self.reward_function(config.config, tracks, timestamp)
                if reward > min(best_rewards):
                    selected_configs[np.argmin(best_rewards)] = config
                    best_rewards[np.argmin(best_rewards)] = reward
            return selected_configs, best_rewards # selected_configs only multiple if nchoose > 1

        def calculate_path(self, configs : list[ConfigWrapper], depth : int):
            '''
            Recursively calculate optimal actions. First, we update the sensor with the first action chosen.
            We then recalculate the action choices from this new state. If we are at the required depth, 
            then evaluate the reward function at this depth and return the path leading to the final action.
            '''

            if depth == len(timestamps):
                # evaluate reward function of current actions
                # here configs is the final set of configs returned at the final step we're looking at
                # print(f'At terminating depth, evaluating reward function with {len(list(configs))} configs')
                selected_configs, reward_value = evaluate_reward_function(self, configs, tracks, timestamps[depth-1], nchoose)
                print(f'Selected config {selected_configs} with reward {reward_value}')
                return selected_configs
            else:
                # act on previous actions, then generate new actions and recall function
                total_configs = []
                for i, config_wrapper in enumerate(configs):
                    predicted_sensors = list() # make new list to store predicted sensors
                    for sensor, actions in config_wrapper.config.items(): # each config is a different set of actions to evaluate
                        memo = {}
                        predicted_sensor = copy.deepcopy(sensor, memo) 
                        predicted_sensor.add_actions(actions) # add the sequence of actions to the sensor
                        predicted_sensor.act(timestamps[depth]) # sensor has now acted on the previous actions. do we use 0 or depth here? 
                        if isinstance(sensor, Sensor):
                            predicted_sensors.append(predicted_sensor)
                    
                    all_action_choices = get_action_choices(predicted_sensors, timestamps[depth]) # get new set of actions for each sensor
                    
                    # this gets the next set of possible actions for the current config
                    new_configs = (     {sensor: action for sensor, action in zip(all_action_choices.keys(), actionconfig)}     for     actionconfig in it.product(*all_action_choices.values())    )

                    for conf in new_configs:
                        new_config = ConfigWrapper(conf, parent = config_wrapper)
                        total_configs.append(new_config) # make big list of all configs for next level
                print(f'Increasing depth from {depth} to {depth + 1}')
                depth += 1 # we now want to move down the tree so we increase depth 
                
                return calculate_path(self, total_configs, depth)

        def generate_actions(self, final_config):
            # return a list of actions to take in ascending order. ignoring the root
            # only works for nchoose = 1 currently
            actions = []
            for act in final_config.iter_path_reverse():
                actions.append(act.config)
            actions.reverse()
            try:
                final_actions = actions[1:self.schedule_horizon + 1]
            except IndexError:
                final_actions = actions[1:]
            return final_actions

        # get actions at first timestamp
        sensors = self.sensors
        all_action_choices = get_action_choices(sensors, timestamps[0])

        # get tuple of dictionaries of sensors: actions
        # configs is a tuple that contains dictionaries
        # each dictionary contains keys = sensors, values = actions
        # each config has a different set of actions to try
        configs = ({sensor: action
                    for sensor, action in zip(all_action_choices.keys(), actionconfig)}
                for actionconfig in it.product(*all_action_choices.values()))

        # Make a root for the tree
        root = ConfigWrapper(config = None)
        # Initialise the tree with the first layer, i.e. the initial configs
        for conf in configs:
            # We simply want to make each config a child of the root, we don't care about saving this value as the root will contain all of these
            _ = ConfigWrapper(conf, parent = root)

        final_config = calculate_path(self, root.children, 1)
   
        final_actions = generate_actions(self, final_config[0])
        
        return final_actions

