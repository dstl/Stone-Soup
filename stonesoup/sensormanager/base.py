# -*- coding: utf-8 -*-

from abc import abstractmethod, ABC
from typing import Callable, Set
import random
import numpy as np
import itertools as it

from ..base import Base, Property
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
    sensors: Set[Sensor] = Property(doc="The sensor(s) which the sensor manager is managing. "
                                        "These must be capable of returning available actions.")

    reward_function: Callable = Property(
        default=None, doc="A function or class designed to work out the reward associated with an "
                          "action or set of actions. For an example see :class:`~.RewardFunction`."
                          " This may also incorporate a notion of the "
                          "cost of making a measurement. The values returned may be scalar or "
                          "vector in the case of multi-objective optimisation. Metrics may be of "
                          "any type and in any units.")

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
        for sensor_action_assignment in configs:
            for sensor in self.sensors:
                action_generators = sensor.actions(timestamp)
                chosen_actions = []
                for action_gen in action_generators:
                    chosen_actions.append(random.choice(list(action_gen)))
                sensor_action_assignment[sensor] = chosen_actions

        return configs


class BruteForceSensorManager(SensorManager):
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

        # Return mapping of sensors and chosen actions for sensors
        return selected_configs
