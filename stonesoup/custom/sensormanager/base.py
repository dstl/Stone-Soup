import numpy as np
import itertools as it

from ...base import Property
from ...sensormanager import SensorManager, BruteForceSensorManager


class UniqueBruteForceSensorManager(SensorManager):
    """A sensor manager which returns a choice of action from those available. The sensor manager
    iterates through every possible configuration of sensors and actions and
    selects the configuration which returns the maximum reward as calculated by a reward function.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def choose_actions(self, tracks, timestamp, nchoose=1, **kwargs):
        """Returns a chosen [list of] action(s) from the action set for each sensor.
        Chosen action(s) is selected by finding the configuration of sensors: actions which returns
        the maximum reward, as calculated by a reward function.

        Parameters
        ----------
        tracks: set of :class:`~Track`
            Set of tracks at given time. Used in reward function.
        timestamp: :class:`datetime.datetime`
            Time at which the actions are carried out until
        nchoose : int
            Number of actions from the set to choose (default is 1)

        Returns
        -------
        : dict
            The pairs of :class:`~.Sensor`: [:class:`~.Action`] selected
        """

        all_action_choices = dict()

        for sensor in self.sensors:
            # get action 'generator(s)'
            action_generators = sensor.actions(timestamp)
            # list possible action combinations for the sensor
            action_choices = list(it.product(*action_generators))
            # dictionary of sensors: list(action combinations)
            all_action_choices[sensor] = action_choices

        configs = []
        poss = []
        for actionconfig in it.product(*all_action_choices.values()):
            cfg = dict()
            pos = set()
            for sensor, actions in zip(all_action_choices.keys(), actionconfig):
                action_x = next(
                    action for action in actions if action.generator.attribute == 'location_x')
                action_y = next(
                    action for action in actions if action.generator.attribute == 'location_y')
                cfg[sensor] = actions
                pos.add((action_x.target_value, action_y.target_value))
            if pos not in poss:
                configs.append(cfg)
                poss.append(pos)

        best_rewards = np.zeros(nchoose) - np.inf
        selected_configs = [None] * nchoose
        rewards = []

        for i, config in enumerate(configs):
            reward, var = self.reward_function(config, tracks, timestamp)
            rewards.append(reward)
            # vars.append(var)
            if reward > min(best_rewards):
                selected_configs[np.argmin(best_rewards)] = config
                best_rewards[np.argmin(best_rewards)] = reward

        # Return mapping of sensors and chosen actions for sensors
        return selected_configs


class SampleBruteForceSensorManager(BruteForceSensorManager):
    """A sensor manager which returns a choice of action from those available. The sensor manager
    iterates through every possible configuration of sensors and actions and
    selects the configuration which returns the maximum reward as calculated by a reward function.

    """

    num_samples: int = Property(doc="Number of samples to take for each timestep", default=10)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def choose_actions(self, tracks, timestamp, nchoose=1, **kwargs):
        """Returns a chosen [list of] action(s) from the action set for each sensor.
        Chosen action(s) is selected by finding the configuration of sensors: actions which returns
        the maximum reward, as calculated by a reward function.

        Parameters
        ----------
        tracks: set of :class:`~Track`
            Set of tracks at given time. Used in reward function.
        timestamp: :class:`datetime.datetime`
            Time at which the actions are carried out until
        nchoose : int
            Number of actions from the set to choose (default is 1)

        Returns
        -------
        : dict
            The pairs of :class:`~.Sensor`: [:class:`~.Action`] selected
        """

        all_action_choices = dict()

        for sensor in self.sensors:
            # get action 'generator(s)'
            action_generators = sensor.actions(timestamp)
            # list possible action combinations for the sensor
            action_choices = list(it.product(*action_generators))
            # dictionary of sensors: list(action combinations)
            all_action_choices[sensor] = action_choices

        # get tuple of dictionaries of sensors: actions
        configs = list({sensor: action
                    for sensor, action in zip(all_action_choices.keys(), actionconfig)}
                   for actionconfig in it.product(*all_action_choices.values()))
        cfgs = []
        poss = []
        for actionconfig in it.product(*all_action_choices.values()):
            cfg = dict()
            pos = set()
            for sensor, actions in zip(all_action_choices.keys(), actionconfig):
                action_x = next(
                    action for action in actions if action.generator.attribute == 'location_x')
                action_y = next(
                    action for action in actions if action.generator.attribute == 'location_y')
                cfg[sensor] = actions
                pos.add((action_x.target_value, action_y.target_value))
            if pos not in poss:
                cfgs.append(cfg)
                poss.append(pos)

        idx = np.random.choice(len(configs), self.num_samples)

        configs = np.array([configs[i] for i in idx])

        best_rewards = np.zeros(nchoose) - np.inf
        selected_configs = [None] * nchoose
        rewards = []
        for config in configs:
            # calculate reward for dictionary of sensors: actions
            reward = self.reward_function(config, tracks, timestamp)
            rewards.append(reward)
            # if reward > min(best_rewards):
            #     selected_configs[np.argmin(best_rewards)] = config
            #     best_rewards[np.argmin(best_rewards)] = reward
        max_idx = np.argwhere(rewards == np.amax(rewards)).flatten()
        best_configs = configs[max_idx]
        if best_configs.size == 1:
            best_config = best_configs[0]
        else:
            best_config = None
            min_dist = np.inf
            for config in best_configs:
                dist = 0
                for sensor, actions in config.items():
                    action_x = next(
                        action for action in actions if action.generator.attribute == 'location_x')
                    action_y = next(
                        action for action in actions if action.generator.attribute == 'location_y')
                    sensor_loc = sensor.position[0:2].flatten()
                    action_loc = np.array([action_x.target_value, action_y.target_value])
                    dist += np.linalg.norm(sensor_loc - action_loc)
                if dist < min_dist:
                    min_dist = dist
                    best_config = config
        # Return mapping of sensors and chosen actions for sensors
        return [best_config]


def is_valid_config(config, **kwargs):
    num_sensors = int(len(kwargs)/2)
    actions_sets = list(config.values())
    for i in range(num_sensors):
        x = kwargs[f'x{i+1}']
        y = kwargs[f'y{i+1}']
        actions = actions_sets[i]
        action_x = next(
            action for action in actions if action.generator.attribute == 'location_x')
        action_y = next(
            action for action in actions if action.generator.attribute == 'location_y')
        if action_x.target_value != x or action_y.target_value != y:
            return False
    return True