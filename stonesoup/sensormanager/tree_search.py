from typing import Callable
import random
import numpy as np
import itertools as it
from datetime import timedelta
import copy

from ..base import Property
from .base import SensorManager
from ..types.track import Track


class MonteCarloTreeSearchSensorManager(SensorManager):
    r"""A Monte Carlo Tree Search bases sensor management algorithm implementing
    a simple value estimation."""

    reward_function: Callable = Property(
        default=None, doc="A function or class designed to work out the reward associated with an "
                          "action or set of actions. This will be implemented to evaluate each "
                          "action within the rollout with the discounted sum being stored at "
                          "the node representing the first action.")

    niterations: int = Property(
        default=100, doc="The number of iterations of the tree search process to be carried out.")

    time_step: timedelta = Property(
        default=timedelta(seconds=1), doc="The sample time between steps in the horizon.")

    exploration_factor: float = Property(
        default=1.0, doc="The exploration factor used in the upper confidence bound for trees.")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def choose_actions(self, tracks, timestamp, nchoose=1, **kwargs):
        """Returns a list of actions that reflect the most visited child nodes to the
        root node in the tree. randomly chosen [list of] action(s) from the action
        set for each sensor.

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

        nodes = [{'Child_IDs': [],
                  'sensors': self.sensors,
                  'config': dict(),  # the config that has resulted in this node
                  'configs': [],  # the configs that have not been simulated
                  'action_count': 0,
                  'visits': 0,
                  'reward': 0,
                  'tracks': {Track([track.state]) for track in tracks},
                  'timestamp': timestamp,
                  'level': 0}]

        loop_count = 0
        while loop_count <= self.niterations:
            loop_count += 1
            # select best child node
            node_indx = 0
            selected_branch = [0]
            while (len(nodes[node_indx]['Child_IDs']) > 0 and
                   len(nodes[node_indx]['Child_IDs']) == nodes[node_indx]['action_count']):
                node_indx = self.tree_policy(nodes, node_indx)
                selected_branch.insert(0, node_indx)

            next_timestamp = nodes[node_indx]['timestamp'] + self.time_step
            if not nodes[node_indx]['Child_IDs']:
                action_count = 0
                all_action_choices = dict()

                for sensor in nodes[node_indx]['sensors']:
                    # get action 'generator(s)'
                    action_generators = sensor.actions(next_timestamp)
                    # list possible action combinations for the sensor
                    action_choices = list(it.product(*action_generators))
                    if not len(action_choices) == 1 and not len(action_choices[0]) == 0:
                        action_count += len(action_choices)
                    # dictionary of sensors: list(action combinations)
                    all_action_choices[sensor] = action_choices

                nodes[node_indx]['action_count'] = action_count
                configs = [{sensor: action
                            for sensor, action in zip(all_action_choices.keys(), actionconfig)}
                           for actionconfig in it.product(*all_action_choices.values())]

                nodes[node_indx]['configs'] = configs

            # select one of the unsimulated configs
            config_indx = random.randint(0, len(nodes[node_indx]['configs'])-1)
            nodes[node_indx]['Child_IDs'].append(len(nodes))
            selected_branch.insert(0, len(nodes))
            nodes.append({'Child_IDs': [],
                          'sensors': set(),
                          'config': nodes[node_indx]['configs'][config_indx],
                          'configs': [],
                          'action_count': 0,
                          'visits': 0,
                          'reward': 0,
                          'timestamp': next_timestamp,
                          'tracks': set(),
                          'level': 0})

            selected_config = copy.deepcopy(nodes[node_indx]['configs'].pop(config_indx))

            reward, updates = self.simulate_action(nodes[-1], nodes[node_indx])

            for n, track in enumerate(nodes[node_indx]['tracks']):
                for update in updates[n]:
                    nodes[-1]['tracks'].add(Track([update]))

            for sensor, actions in selected_config.items():
                sensor.add_actions(actions)
                sensor.act(nodes[-1]['timestamp'])
                nodes[-1]['sensors'].add(sensor)

            for node_id in selected_branch:
                nodes[node_id]['visits'] += 1
                nodes[node_id]['reward'] += reward

        best_children = self.select_best_child(nodes)
        selected_configs = []
        for best_child in best_children:
            selected_configs.append(nodes[best_child]['config'])

        return selected_configs

    def tree_policy(self, nodes, node_indx):
        """Implements the upper confidence bound for trees, which balances exploitation
        of highly rewarding actiond and exploring actions that have been visited a fewer times"""

        uct = []
        for Child_ID in nodes[node_indx]['Child_IDs']:
            uct.append(nodes[Child_ID]['reward']/nodes[Child_ID]['visits'] +
                       self.exploration_factor*np.sqrt(np.log(nodes[node_indx]['visits'])
                                                       /nodes[Child_ID]['visits']))

        max_uct_indx = np.argmax(uct)
        return nodes[node_indx]['Child_IDs'][max_uct_indx]

    @staticmethod
    def select_best_child(nodes):
        """Selects the best child node to the root node in the tree according to
        maximum number of visits."""

        visit_list = []
        for Child_ID in nodes[0]['Child_IDs']:
            visit_list.append(nodes[Child_ID]['visits'])

        max_visit_indx = np.argmax(visit_list)
        return [nodes[0]['Child_IDs'][max_visit_indx]]

    def simulate_action(self, node, parent_node):
        """Simulates the expected reward that would be received by executing
        the candidate action."""

        reward, updates = self.reward_function(node['config'],
                                               parent_node['tracks'],
                                               node['timestamp'])

        return reward, updates


class MCTSRolloutSensorManager(MonteCarloTreeSearchSensorManager):
    r"""A Monte Carlo Tree Search bases sensor management algorithm implementing
    a Monte Carlo rollout policy for action value estimation."""

    rollout_depth: int = Property(
        default=1, doc="The depth of rollout to conduct for each node.")

    discount_factor: float = Property(
        default=0.9, doc="The discount factor is applied to each action evaluated in the "
                         "tree to assign an incrementally lower multiplier to future actions "
                         "in the tree.")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def simulate_action(self, node, parent_node):
        """Simulates the expected reward that would be received by executing
        the candidate action."""

        reward_list = []
        # calculate reward of new node
        reward, updates = self.reward_function(node['config'],
                                               parent_node['tracks'],
                                               node['timestamp'])
        reward_list.append(reward)
        # tmp_tracks = copy.deepcopy(parent_node['tracks'])
        tmp_tracks = {Track([track.state]) for track in parent_node['tracks']}
        for n, track in enumerate(tmp_tracks):
            for update in updates[n]:
                track.append(update)

        tmp_sensors = set()
        for sensor, actions in copy.deepcopy(node['config']).items():
            sensor.add_actions(actions)
            sensor.act(node['timestamp'])
            tmp_sensors.add(sensor)

        # execute Monte Carlo Rollout from the new node
        for d in range(self.rollout_depth):
            all_action_choices = dict()
            timestamp = node['timestamp'] + ((d + 1) * self.time_step)

            action_count = 0
            for sensor in tmp_sensors:
                # get action 'generator(s)'
                action_generators = sensor.actions(timestamp)
                # list possible action combinations for the sensor
                action_choices = list(it.product(*action_generators))
                if not len(action_choices) == 1 and not len(action_choices[0]) == 0:
                    action_count += len(action_choices)
                # dictionary of sensors: list(action combinations)
                all_action_choices[sensor] = action_choices

            configs = [{sensor: action
                        for sensor, action in zip(all_action_choices.keys(), actionconfig)}
                       for actionconfig in it.product(*all_action_choices.values())]

            random_config_indx = random.randint(0, action_count-1)
            random_config = configs[random_config_indx]

            reward, updates_ = self.reward_function(random_config, tmp_tracks, timestamp)

            reward *= self.discount_factor**(d+1)
            reward_list.append(reward)
            for n, track in enumerate(tmp_tracks):
                for update in updates_[n]:
                    track.append(update)

            for sensor, actions in random_config.items():
                sensor.add_actions(actions)
                sensor.act(timestamp)

        final_reward = sum(reward_list)
        return final_reward, updates
