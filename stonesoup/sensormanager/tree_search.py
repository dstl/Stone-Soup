import copy
import itertools as it
from datetime import timedelta
from typing import Callable
from enum import Enum
import warnings

import numpy as np

from .base import SensorManager
from ..base import Property


class MCTSBestChildPolicyEnum(Enum):
    r"""Best child policy Enum class for specifying which policy to use when selecting
    the best child at the end of the MCTS process."""
    MAXAREWARD = 'max_average_reward'
    MAXCREWARD = 'max_cumulative_reward'
    MAXVISITS = 'max_visits'


class MonteCarloTreeSearchSensorManager(SensorManager):
    r"""A Monte Carlo tree search based sensor management algorithm implementing
    simple value estimation.

    Monte Carlo tree search works by simultaneously constructing and evaluating a
    search tree of states and actions through an iterative process. The process
    consists of 4 stages: Selection, Expansion, Simulation and Backpropagation.
    The purpose of the algorithm is to arrive at the optimal action policy by
    sequentially estimating the action value function, :math:`Q`, and returning
    the maximum argument to this at the end of the process.

    Starting from the root node (current state or estimated state) the best child
    node is selected. The most common way, and the way implemented here, is to
    select this node according to the upper confidence bound (UCB) for trees.
    This is given by

    .. math::
        \text{argmax}_{a} \frac{Q(h, a)}{N(h, a)}+c\sqrt{\frac{\log N(h)}{N(h,a)}},

    where :math:`a` is the action, :math:`h` is the history (for POMDP problems a
    history or belief is commonly used but in MDP problems :math:`h` would be replaced
    with a state), :math:`Q(h, a)` is the current cumulative action value estimate,
    :math:`N(h, a)` is the number of visits or simulations of this node, :math:`N(h)`
    is the number of visits to the parent node and :math:`c` is the exploration factor,
    defined with :attr:`exploration_factor`. The purpose of the UCB is to trade off
    between exploitation of the most rewarding nodes in the tree and exploration of
    those that have been visited fewer times, as the second term in the above
    expression will accumulate as the ratio of number of parent visits and child
    visits increases.

    Once the best child node has been selected, this becomes a parent node and a
    new child node added according to the available set of unvisited actions. This
    selection happens at random. This node is then simulated by predicting the
    current state estimate in the parent node and updating this estimate with a
    generated detection after applying the candidate action. This provides a
    predicted future state which is used to calculate the action value of this node.
    This is done by providing a :attr:`reward_function`. Finally, this reward is
    added to the node action value, discounted appropriately according to the
    depth into the future, and combined with action values of parent nodes (that
    were descended during selection) when completing the backpropagation process.
    This creates a tradeoff between future and immediate rewards during the next
    iteration of the search process.

    Once a predefined computational budget has been reached, which in this implementation
    is the :attr:`niterations` attribute, the best child to the root node in the tree
    is determined and returned from the :meth:`choose_actions`. The user can select which
    criteria used to select this best action by defining the :attr:`best_child_policy`.
    The initial implementation of MCTS with rollout (:class:`~.MCTSRolloutSensorManager`)
    can be seen in work by Glover et al [1]_ and further detail on MCTS and its
    variations can also be seen in [2]_.

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

    best_child_policy: MCTSBestChildPolicyEnum = Property(
        default=MCTSBestChildPolicyEnum.MAXCREWARD,
        doc="The policy for selecting the best child. Options are ``'max_average_reward'`` for "
            "the maximum reward per visit to a node, ``'max_cumulative_reward'`` for the maximum "
            "total reward after all simulations and ``'max_visits'`` for the node with the "
            "maximum number of visits. Default is ``'max_cumulative_reward'``.")

    discount_factor: float = Property(
        default=0.9,
        doc="The discount factor is applied to rewards beyond the immidiate future timestep "
            "to reduce the reward of future nodes to reflect the increasing level of uncertainty "
            "the further into the horizon the search progresses. It is applied multiplicatively "
            "such that the factor will be raised by power of the number of timesteps "
            "beyond the immidiate future timestep.")

    search_depth: int = Property(
        default=None,
        doc="The maximum depth to apply to the search tree, specifying the maximum number of "
            "future timesteps to expand to.")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Ensure birth scheme is a valid BirthSchemeEnum
        self.best_child_policy = MCTSBestChildPolicyEnum(self.best_child_policy)
        # if search depth none, replace with inf to allow logical operations
        if not self.search_depth:
            self.search_depth = np.inf

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
                  'action_value': 0,
                  'tracks': copy.deepcopy(tracks),
                  'timestamp': timestamp-self.time_step,
                  'level': 0}]

        loop_count = 0
        while loop_count <= self.niterations:
            loop_count += 1
            # select best child node
            node_indx = 0
            selected_branch = [0]
            level = 0
            while (len(nodes[node_indx]['Child_IDs']) > 0 and
                   len(nodes[node_indx]['Child_IDs']) == nodes[node_indx]['action_count']):
                node_indx = self.tree_policy(nodes, node_indx)
                selected_branch.insert(0, node_indx)
                level += 1

            if level <= self.search_depth:

                next_timestamp = nodes[node_indx]['timestamp'] + self.time_step
                if not nodes[node_indx]['Child_IDs']:
                    action_count = 1
                    all_action_choices = dict()

                    for sensor in nodes[node_indx]['sensors']:
                        # get action 'generator(s)'
                        action_generators = sensor.actions(next_timestamp)
                        # list possible action combinations for the sensor
                        action_choices = list(it.product(*action_generators))
                        if len(action_choices) != 1 and len(action_choices[0]) != 0:
                            action_count *= len(action_choices)
                        # dictionary of sensors: list(action combinations)
                        all_action_choices[sensor] = action_choices

                    nodes[node_indx]['action_count'] = action_count
                    configs = [{sensor: action
                                for sensor, action in zip(all_action_choices.keys(), actionconfig)}
                               for actionconfig in it.product(*all_action_choices.values())]

                    nodes[node_indx]['configs'] = configs

                # select one of the unsimulated configs
                config_indx = np.random.randint(0, len(nodes[node_indx]['configs']))
                nodes[node_indx]['Child_IDs'].append(len(nodes))
                selected_branch.insert(0, len(nodes))
                nodes.append({'Child_IDs': [],
                              'sensors': set(),
                              'config': nodes[node_indx]['configs'][config_indx],
                              'configs': [],
                              'action_count': 0,
                              'visits': 0,
                              'reward': 0,
                              'action_value': 0,
                              'timestamp': next_timestamp,
                              'tracks': set(),
                              'level': level})

                selected_config = copy.deepcopy(nodes[node_indx]['configs'].pop(config_indx))

                reward, future_reward, updates = self.simulate_action(nodes[-1], nodes[node_indx])

                nodes[-1]['tracks'] = updates
                nodes[-1]['reward'] = reward  # store immidiate reward as 'reward'

                for sensor, actions in selected_config.items():
                    sensor.add_actions(actions)
                    sensor.act(nodes[-1]['timestamp'])
                    nodes[-1]['sensors'].add(sensor)

            else:
                # search depth reached
                future_reward = 0

            # if future_reward is None, assign it as 0
            sim_action_value = future_reward if future_reward else 0
            for node_id in selected_branch:
                nodes[node_id]['visits'] += 1
                sim_action_value += nodes[node_id]['reward']
                nodes[node_id]['action_value'] += sim_action_value
                sim_action_value *= self.discount_factor

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
            uct.append(nodes[Child_ID]['action_value']/nodes[Child_ID]['visits'] +
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
            reward_list.append(nodes[Child_ID]['action_value'])

        if self.best_child_policy == MCTSBestChildPolicyEnum.MAXCREWARD:
            max_indx = np.argmax(reward_list)
        elif self.best_child_policy == MCTSBestChildPolicyEnum.MAXAREWARD:
            max_indx = np.argmax(np.asarray(reward_list) / np.asarray(visit_list))
        elif self.best_child_policy == MCTSBestChildPolicyEnum.MAXVISITS:
            max_indx = np.argmax(visit_list)

        return [nodes[0]['Child_IDs'][max_indx]]

    def simulate_action(self, node, parent_node):
        """Simulates the expected reward that would be received by executing
        the candidate action."""

        reward, updates = self.reward_function(node['config'],
                                               parent_node['tracks'],
                                               node['timestamp'])
        future_reward = None

        return reward, future_reward, updates


class MCTSRolloutSensorManager(MonteCarloTreeSearchSensorManager):
    r"""A Monte Carlo Tree Search based sensor management algorithm that implements Monte
    Carlo rollout for more robust action simulation. All other details are consistent
    with :class:`~.MonteCarloTreeSearchSensorManager`"""

    rollout_depth: int = Property(
        default=None,
        doc="The depth of rollout to conduct for each node. This is only used when "
            ":attr:`search_depth` is not set or set to `None`.")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # warn about rollout_depth not being considered when search depth in specified
        if self.search_depth < np.inf and self.rollout_depth:
            warnings.warn('`search_depth` and `rollout_depth` have been defined. '
                          '`search_depth` overrides rollout depth and forces rollout '
                          'to end at `search_depth`!')

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
        n_steps = self.rollout_depth if np.isinf(self.search_depth) \
            else self.search_depth - node['level']
        for d in range(n_steps):
            all_action_choices = dict()
            timestamp = node['timestamp'] + ((d + 1) * self.time_step)

            action_count = 0
            for sensor in tmp_sensors:
                # get action 'generator(s)'
                action_generators = sensor.actions(timestamp)
                # list possible action combinations for the sensor
                action_choices = list(it.product(*action_generators))
                if len(action_choices) != 1 and len(action_choices[0]) != 0:
                    action_count += len(action_choices)
                # dictionary of sensors: list(action combinations)
                all_action_choices[sensor] = action_choices

            configs = [{sensor: action
                        for sensor, action in zip(all_action_choices.keys(), actionconfig)}
                       for actionconfig in it.product(*all_action_choices.values())]

            random_config_indx = np.random.randint(0, action_count)
            random_config = configs[random_config_indx]

            reward, updates_ = self.reward_function(random_config, updates_, timestamp)

            reward *= self.discount_factor**(d+1)
            reward_list.append(reward)

            for sensor, actions in random_config.items():
                sensor.add_actions(actions)
                sensor.act(timestamp)

        rollout_reward = sum(reward_list[1:])
        return reward_list[0], rollout_reward, updates
