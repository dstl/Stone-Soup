from abc import abstractmethod, ABC
from typing import Callable, Set
import random
import numpy as np
import itertools as it
from typing import TYPE_CHECKING

from ..base import Base, Property
from ..platform.base import Platform

if TYPE_CHECKING:
    from ..sensor.sensor import Sensor


class SensorManager(Base, ABC):
    """The sensor manager base class.

    The purpose of a sensor manager is to return a mapping of sensors and sensor actions
    appropriate to a specific
    scenario and with a particular objective, or objectives, in mind. This involves using
    estimates of the situation and knowledge of the sensor system to calculate metrics associated
    with actions, and then determine optimal, or near optimal, actions to take.

    There is considerable freedom in both the theory and practice of sensor management and these
    classes do not enforce a particular solution. A sensor manager may be 'centralised' in that
    it controls the actions of multiple sensors, or individual sensors may have their own managers
    which communicate with other sensor managers in a networked fashion.

    """
    sensors: Set['Sensor'] = Property(doc="The sensor(s) which the sensor manager is managing.")

    platforms: Set[Platform] = Property(doc="The platform(s) which the sensor manager is "
                                            "managing.")

    reward_function: Callable = Property(
        default=None, doc="A function or class designed to work out the reward associated with an "
                          "action or set of actions. For an example see :class:`~.RewardFunction`."
                          " This may also incorporate a notion of the "
                          "cost of making a measurement. The values returned may be scalar or "
                          "vector in the case of multi-objective optimisation. Metrics may be of "
                          "any type and in any units.")

    take_sensors_from_platforms: bool = Property(
        default=True, doc="Whether to update the sensor set with any sensors that are on the "
                          "platform(s) but not already in the sensor set. Any sensors not added "
                          "to the sensor set will not be considered by the sensor manager or "
                          "reward function.")

    @property
    def actionables(self):
        actionables = set()
        if self.take_sensors_from_platforms:
            for platform in self.platforms:
                self.sensors.update(platform.sensors)
        actionables.update(self.sensors, self.platforms)

        return actionables

    @abstractmethod
    def choose_actions(self, timestamp, nchoose, **kwargs):
        """A method which returns a set of actions, designed to be enacted by a sensor, or
        sensors, chosen by some means. This will likely make use of optimisation algorithms.

        Returns
        -------
        : dict {:class:`~.Sensor`: [:class:`~.Action`]}
            Key-value pairs of the form 'sensor: actions'. In the general case a sensor may be
            given a single action, or a list. The actions themselves are objects which must be
            interpretable by the sensor to which they are assigned.
        """
        raise NotImplementedError


class RandomSensorManager(SensorManager):
    """As the name suggests, a sensor manager which returns a random choice of action or actions
    from the list available. Its practical purpose is to serve as a baseline to test against.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def choose_actions(self, tracks, timestamp, nchoose=1, **kwargs):
        """Returns a randomly chosen [list of] action(s) from the action set for each sensor.

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

        configs = [dict() for _ in range(nchoose)]
        for config in configs:
            for actionable in self.actionables:
                action_generators = actionable.actions(timestamp)
                chosen_actions = []
                for action_gen in action_generators:
                    chosen_actions.append(random.choice(list(action_gen)))
                config[actionable] = chosen_actions

        return configs


class BruteForceSensorManager(SensorManager):
    """A sensor manager which returns a choice of action from those available. The sensor manager
    iterates through every possible configuration of sensors and actions and
    selects the configuration which returns the maximum reward as calculated by a reward function.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def choose_actions(self, tracks, timestamp, nchoose=1, return_reward=False, **kwargs):
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
        return_reward: bool
            Whether to return the reward for chosen actions (default is False)
            When True, returns a tuple of 1d arrays: (dictionaries of chosen actions, rewards)
        Returns
        -------
        : list(dict) or (list(dict), :class:`numpy.ndarray`)
            The pairs of :class:`~.Sensor`: [:class:`~.Action`] selected and the array contains
            the corresponding reward.
        """

        all_action_choices = dict()

        for actionable in self.actionables:
            # get action 'generator(s)'
            action_generators = actionable.actions(timestamp)
            # list possible action combinations for the sensor
            action_choices = list(it.product(*action_generators))
            # dictionary of sensors: list(action combinations)
            all_action_choices[actionable] = action_choices

        # get tuple of dictionaries of sensors: actions
        configs = ({sensor: action
                    for sensor, action in zip(all_action_choices.keys(), actionconfig)}
                   for actionconfig in it.product(*all_action_choices.values()))

        best_rewards = np.zeros(nchoose) - np.inf
        selected_configs = [None] * nchoose
        for config in configs:
            # calculate reward for dictionary of sensors: actions
            reward = self.reward_function(config, tracks, timestamp)
            if reward > min(best_rewards):
                selected_configs[np.argmin(best_rewards)] = config
                best_rewards[np.argmin(best_rewards)] = reward
        if return_reward:
            # Return mapping of sensors and chosen actions for sensors
            # Also returns rewards
            return selected_configs, best_rewards
        else:
            return selected_configs


class GreedySensorManager(SensorManager):
    """A sensor manager that returns a choice of actions from those available. Calculates
    a reward function for each sensor in isolation. Selects the action that maximises reward
    for each sensor.

    """

    def choose_actions(self, tracks, timestamp, nchoose=1, **kwargs):
        """Returns a chosen [list of] action(s) from the action set for each sensor.
        Chosen action(s) is selected by finding the configuration of sensors: actions which returns
        the maximum reward, as calculated by a reward function.

        Parameters
        ----------
        tracks: set of :class:`~.Track`
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

        chosen_actions = dict()

        for actionable in self.actionables:
            # get action 'generator(s)'
            action_generators = actionable.actions(timestamp)
            # list possible action combinations for the sensor/platform
            action_choices = list(it.product(*action_generators))

            best_rewards = np.zeros(nchoose) - np.inf
            selected_actions = [None] * nchoose
            for action in action_choices:
                # calculate reward for each action
                reward = self.reward_function({actionable: action}, tracks, timestamp)
                if reward > min(best_rewards):
                    selected_actions[np.argmin(best_rewards)] = action
                    best_rewards[np.argmin(best_rewards)] = reward

            # save nchoose best actions for the sensor/platform
            chosen_actions[actionable] = selected_actions

        # convert from single dict of actionable: list(actions) to list of dicts of
        # actionables: actions
        selected_configs = [{actionable: chosen_actions[actionable][i]
                             for actionable in chosen_actions}
                            for i in range(nchoose)]

        # Return mapping of sensors and chosen actions for sensors
        return selected_configs
