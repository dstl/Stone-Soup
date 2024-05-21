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
    r"""A Monte Carlo Tree Search based sensor management algorithm implementing
    a simple value estimation.

    Monte Carlo Tree Search works by simultaneously constructing and evaluating a
    search tree of states and actions through an iterative process. The process
    consists of 4 stages: Selection, Expansion, Simulation and Backpropagation.
    The purpose of the algorithm is to arrive at the optimal action policy by
    sequentially estimating the action value function, :math:`Q`, and returning
    the maximum argument to this at the end of the process.

    Starting from the root node (current state or estimated state) the best child
    node is selected. The most common way, and the way implemented here, is to
    select this node is according to the upper confidence bound (UCB) for trees.
    This is given by

    .. math::
        \text{argmath}_{a} \frac{Q(h, a)}{N(h, a)}+c\sqrt{\frac{\log N(h)}{N(h,a)}},

    where :math:`a` is the action, :math:`h` is the history (for POMDP problems a
    history or belief is commonly used but in MDP problems h would be replaced
    with a state), :math:`Q(h, a)` is the current cumulative action value estimate,
    :math:`N(h, a)` is the number of visits or simulations of this node, :math:`N(h)`
    is the number of visits to the parent node and :math:`c` is the exploration factor,
    defined with :attr:`exploration_factor`. The purpose of the UCB is to trade off
    between exploitation of the most rewarding nodes in the tree and those that have
    been visited fewer times, as the second term in the above expression will
    accumulate as the ratio of number of parent visits and child visits increases.

    Once the best child node has been selected, this becomes a parent node and a
    new child node added according to the available set of unvisited actions. This
    selection happens at random. This node is then simulated by predicting the
    current state estimate in the parent node and updating this estimate with a
    generated detection after applying the candidate action. This provides a
    predicted future state which is used to calculate the action value of this node.
    This is done by providing a :attr:`reward_function`. Finally, this reward is
    added to the current action value estimated in each node on the search tree
    branch that was descended during selection. This creates a tradeoff between future
    and immediate rewards during the next iteration of the search process.

    Once a predefined computational budget has been reached, which in this implementation
    is the :attr:`niterations` attribute, the best child to the root node in the tree
    is determined and returned from the :meth:`choose_actions`. The user can select which
    criteria used to select this best action by defining the :attr:`best_child_policy`.
    Further detail on this particular implementation can be seen in work by Glover
    et al [1]_. Further detail on MCTS and its variations can also be seen in [2]_.

    References
    ----------
    .. [1] Glover, Timothy & Nanavati, Rohit V. & Coombes, Matthew & Liu, Cunjia &
           Chen, Wen-Hua & Perree, Nicola & Hiscocks, Steven. "A Monte Carlo Tree Search
           Framework for Autonomous Source Term Estimation in Stone Soup, 2024 27th
           International Conference on Information Fusion (FUSION), 1-8, 2024"
    .. [2] Kochenderfer, Mykel J. & Wheeler, Tim A. & Wray, Kyle H. "Algorithms for
           decision making", MIT Press, 2022 (https://algorithmsbook.com/)
    """

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

    best_child_policy: int = Property(
        default=0, doc="An integer controlling which policy to use when determining the best "
                       "child at the end of the MCTS process. The choices are 0: maximum reward, "
                       "1: maximum reward per visit or 2: maximum visit count"
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def choose_actions(self, tracks, timestamp, nchoose=1, **kwargs):
        """Returns a list of actions that reflect the best child nodes to the
        root node in the tree.

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
                  'sensors': self.actionables,
                  'config': dict(),  # the config that has resulted in this node
                  'configs': [],  # the configs that have not been simulated
                  'action_count': 0,
                  'visits': 0,
                  'reward': 0,
                  'tracks': {Track([track.state]) for track in tracks},
                  'timestamp': timestamp-self.time_step,
                  'level': 0}]

        loop_count = 0
        while loop_count <= self.niterations:
            loop_count += 1
            # select best child node
            node_indx = 0
            selected_branch = [0]
            level = 1
            while (len(nodes[node_indx]['Child_IDs']) > 0 and
                   len(nodes[node_indx]['Child_IDs']) == nodes[node_indx]['action_count']):
                node_indx = self.tree_policy(nodes, node_indx)
                selected_branch.insert(0, node_indx)
                level += 1

            next_timestamp = nodes[node_indx]['timestamp'] + self.time_step
            if not nodes[node_indx]['Child_IDs']:
                action_count = 1
                all_action_choices = dict()

                for sensor in nodes[node_indx]['sensors']:
                    # get action 'generator(s)'
                    action_generators = sensor.actions(next_timestamp)
                    # list possible action combinations for the sensor
                    action_choices = list(it.product(*action_generators))
                    if not len(action_choices) == 1 and not len(action_choices[0]) == 0:
                        action_count *= len(action_choices)
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
                          'level': level})

            selected_config = copy.deepcopy(nodes[node_indx]['configs'].pop(config_indx))

            reward, updates = self.simulate_action(nodes[-1], nodes[node_indx])

            for track in updates:
                nodes[-1]['tracks'].add(Track(track[-1]))

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
        of highly rewarding actioned and exploring actions that have been visited a fewer times"""

        uct = []
        for Child_ID in nodes[node_indx]['Child_IDs']:
            uct.append(nodes[Child_ID]['reward']/nodes[Child_ID]['visits'] +
                       self.exploration_factor*np.sqrt(np.log(nodes[node_indx]['visits'])
                                                       / nodes[Child_ID]['visits']))

        max_uct_indx = np.argmax(uct)
        return nodes[node_indx]['Child_IDs'][max_uct_indx]

    def select_best_child(self, nodes):
        """Selects the best child node to the root node in the tree according to
        maximum number of visits."""

        visit_list = []
        reward_list = []
        for Child_ID in nodes[0]['Child_IDs']:
            visit_list.append(nodes[Child_ID]['visits'])
            reward_list.append(nodes[Child_ID]['reward'])

        if self.best_child_policy == 0:
            max_reward_indx = np.argmax(reward_list)
            return [nodes[0]['Child_IDs'][max_reward_indx]]
        elif self.best_child_policy == 1:
            max_creward_indx = np.argmax(np.asarray(reward_list) / np.asarray(visit_list))
            return [nodes[0]['Child_IDs'][max_creward_indx]]
        elif self.best_child_policy == 2:
            max_visit_indx = np.argmax(visit_list)
            return [nodes[0]['Child_IDs'][max_visit_indx]]
        else:
            raise NotImplementedError('Selected best child policy is not a valid option')

    def simulate_action(self, node, parent_node):
        """Simulates the expected reward that would be received by executing
        the candidate action."""

        reward, updates = self.reward_function(node['config'],
                                               parent_node['tracks'],
                                               node['timestamp'])

        return reward, updates


class MCTSRolloutSensorManager(MonteCarloTreeSearchSensorManager):
    r"""A Monte Carlo Tree Search bases sensor management algorithm that implements Monte
    Carlo rollout for more robust action simulation. All other details are consistent
    with :class:`~.MonteCarloTreeSearchSensorManager`"""

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
        updates_ = updates

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

            reward, updates_ = self.reward_function(random_config, updates_, timestamp)

            reward *= self.discount_factor**(d+1)
            reward_list.append(reward)

            for sensor, actions in random_config.items():
                sensor.add_actions(actions)
                sensor.act(timestamp)

        final_reward = sum(reward_list)
        return final_reward, updates
