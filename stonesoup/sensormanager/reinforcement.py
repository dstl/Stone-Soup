import abc
from abc import abstractmethod, ABC
from typing import Callable, Set
import random
import numpy as np
import itertools as it

from .base import SensorManager
from ..base import Base, Property
from ..sensor.sensor import Sensor

try:
    import tensorflow as tf
    import reverb
    from tf_agents.environments import py_environment
    from tf_agents.environments import utils
    from tf_agents.specs import array_spec
    from tf_agents.trajectories import time_step as ts
    from tf_agents.agents.dqn import dqn_agent
    from tf_agents.drivers import py_driver
    from tf_agents.environments import tf_py_environment
    from tf_agents.networks import sequential
    from tf_agents.policies import py_tf_eager_policy
    from tf_agents.policies import random_tf_policy
    from tf_agents.policies import greedy_policy
    from tf_agents.replay_buffers import reverb_replay_buffer
    from tf_agents.replay_buffers import reverb_utils
    from tf_agents.specs import tensor_spec
    from tf_agents.utils import common
except ImportError as error:
    raise ImportError(
        "Usage of reinforcement learning classes requires that the optional"
        "package dependency tf-agents[reverb] is installed. "
        "This can be achieved by running "
        "'python -m pip install stonesoup[reinforcement]'") \
        from error


class PyEnvironment(object):

    def __init__(self):
        self._current_time_step = None

    def reset(self):
        """Return initial_time_step."""
        self._current_time_step = self._reset()
        return self._current_time_step

    def step(self, a):
        """Apply action and return new time_step."""
        if self._current_time_step is None:
            return self.reset()
        self._current_time_step = self._step(a)
        return self._current_time_step

    def current_time_step(self):
        return self._current_time_step

    def time_step_spec(self):
        """Return time_step_spec."""

    @abc.abstractmethod
    def observation_spec(self):
        """Return observation_spec."""

    @abc.abstractmethod
    def action_spec(self):
        """Return action_spec."""

    @abc.abstractmethod
    def _reset(self):
        """Return initial_time_step."""

    @abc.abstractmethod
    def _step(self, a):
        """Apply action and return new time_step."""


class BaseEnvironment(py_environment.PyEnvironment):
    def __init__(self):
        super().__init__()
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=self.num_actions, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(self.observation_size,), dtype=np.float32, minimum=0, name='observation')  # distance
        self._episode_ended = False
        self._current_episode = 0
        self._max_episode_length = self.episode_length

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        # global satAERMesT, satECIMesT
        self._episode_ended = False
        self._current_episode = 0
        return ts.restart(np.zeros(self.observation_size), dtype=np.float32)

    def _step(self, action):

        reward = 0
        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self.reset()

        self._current_episode += 1
        observation = list(self.observation_size)

        if self._current_episode >= self._max_episode_length:
            self._episode_ended = True
            return ts.termination(observation, reward)
        else:
            # print('here1')
            return ts.transition(observation, reward=reward, discount=1.0)


class ReinforcementLearningSensorManager(SensorManager):
    """A sensor manager that employs reinforcement learning algorithms from tensorflow-agents
    """

    """
    Things I need:
        Do I need import the environment? (env validation)
        If not, need action_spec, observation_spec and reward_function
        action_spec -> sensors.action(timestep) ? can be multi-D
        observation_spec -> user defined?
        reward_function -> user defined, passed into sensor manager (e.g. brute force)
        need to set hyper-parameters
        need to set NN size/shape
        need to choose which RL agent/network used
        set up reverb table
        return policy/training performance?
        or just implement policy with choose_actions
        what to do for multiple sensors? MARL?
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_avg_return(self, environment, policy, num_episodes=10):
        time_step = None
        episode_return = None
        total_return = 0.0
        for _ in range(num_episodes):
            time_step = environment.reset()
            episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return

        avg_return = total_return / num_episodes
        return avg_return.numpy()[0]

    def dense_layer(self, num_units):
        # Define a helper function to create Dense layers configured with the right
        # activation and kernel initializer.
        return tf.keras.layers.Dense(
            num_units,
            activation=tf.keras.activations.relu,
            kernel_initializer=tf.keras.initializers.VarianceScaling(
                scale=2.0, mode='fan_in', distribution='truncated_normal'))

    def train(self, env, hyperparameters, **kwargs):
        env.reset()

        train_py_env = env
        eval_py_env = env
        train_env = tf_py_environment.TFPyEnvironment(train_py_env)
        eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

        fc_layer_params = (100, 50)
        action_tensor_spec = tensor_spec.from_spec(env.action_spec())
        num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1

        # QNetwork consists of a sequence of Dense layers followed by a dense layer
        # with `num_actions` units to generate one q_value per available action as
        # its output.

        dense_layers = [self.dense_layer(num_units) for num_units in fc_layer_params]
        q_values_layer = tf.keras.layers.Dense(
            num_actions,
            activation=None,
            kernel_initializer=tf.keras.initializers.RandomUniform(
                minval=-0.03, maxval=0.03),
            bias_initializer=tf.keras.initializers.Constant(-0.2))
        q_net = sequential.Sequential(dense_layers + [q_values_layer])

        optimizer = tf.keras.optimizers.Adam(hyperparameters.learning_rate)

        train_step_counter = tf.Variable(0)

        self.agent = dqn_agent.DdqnAgent(
            train_env.time_step_spec(),
            train_env.action_spec(),
            q_network=q_net,
            optimizer=optimizer,
            td_errors_loss_fn=common.element_wise_squared_loss,
            train_step_counter=train_step_counter)

        self.agent.initialize()

        eval_policy = self.agent.policy
        collect_policy = self.agent.collect_policy
        random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(), train_env.action_spec())

        # @test {"skip": true}
        # See also the metrics module for standard implementations of different metrics.
        # https://github.com/tensorflow/agents/tree/master/tf_agents/metrics

        self.compute_avg_return(eval_env, random_policy, hyperparameters.num_eval_episodes)

        table_name = 'uniform_table'
        replay_buffer_signature = tensor_spec.from_spec(
            self.agent.collect_data_spec)
        replay_buffer_signature = tensor_spec.add_outer_dim(
            replay_buffer_signature)

        table = reverb.Table(
            table_name,
            max_size=hyperparameters.replay_buffer_max_length,
            sampler=reverb.selectors.Uniform(),
            remover=reverb.selectors.Fifo(),
            rate_limiter=reverb.rate_limiters.MinSize(1),
            signature=replay_buffer_signature)

        reverb_server = reverb.Server([table])

        replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
            self.agent.collect_data_spec,
            table_name=table_name,
            sequence_length=2,
            local_server=reverb_server)

        rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
            replay_buffer.py_client,
            table_name,
            sequence_length=2)

        # agent.collect_data_spec
        # agent.collect_data_spec._fields

        # @test {"skip": true}
        py_driver.PyDriver(
            env,
            py_tf_eager_policy.PyTFEagerPolicy(
                random_policy, use_tf_function=True),
            [rb_observer],
            max_steps=hyperparameters.initial_collect_steps).run(train_py_env.reset())

        # Dataset generates trajectories with shape [Bx2x...]
        dataset = replay_buffer.as_dataset(
            num_parallel_calls=3,
            sample_batch_size=hyperparameters.batch_size,
            num_steps=2).prefetch(3)

        iterator = iter(dataset)

        # #@test {"skip": true}
        # try:
        #   % % time
        # except:
        #   pass

        # (Optional) Optimize by wrapping some code in a graph using TF function.
        self.agent.train = common.function(self.agent.train)

        # Reset the train step.
        self.agent.train_step_counter.assign(0)

        # Evaluate the agent's policy once before training.
        avg_return = self.compute_avg_return(eval_env, self.agent.policy, hyperparameters.num_eval_episodes)
        returns = [avg_return]

        # Reset the environment.
        time_step = train_py_env.reset()

        # Create a driver to collect experience.
        collect_driver = py_driver.PyDriver(
            env,
            py_tf_eager_policy.PyTFEagerPolicy(
                self.agent.collect_policy, use_tf_function=True),
            [rb_observer],
            max_steps=hyperparameters.collect_steps_per_iteration)

        for _ in range(hyperparameters.num_iterations):
            # print('Iteration: {i}'.format(i=_))
            # Collect a few steps and save to the replay buffer.
            time_step, _ = collect_driver.run(time_step)

            # Sample a batch of data from the buffer and update the agent's network.
            experience, unused_info = next(iterator)
            # import pdb; pdb.set_trace()
            train_loss = self.agent.train(experience).loss

            step = self.agent.train_step_counter.numpy()

            if step % hyperparameters.log_interval == 0:
                print('step = {0}: loss = {1}'.format(step, train_loss))

            if step % hyperparameters.eval_interval == 0:
                # Agent Policy Output
                avg_return = self.compute_avg_return(eval_env, self.agent.policy, hyperparameters.num_eval_episodes)
                returns.append(avg_return)
                print('step = {0}: Average Return = {1}'.format(step, avg_return))

    def choose_actions(self, tracks, timestamp, env, nchoose=1, **kwargs):
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
            env : :class:`BaseEnvironment`
                Environment used during training

            Returns
            -------
            : dict
                The pairs of :class:`~.Sensor`: [:class:`~.Action`] selected
            """

            # Need to convert timestamp -> time_step somehow?
            # Maybe by comparing to start timestamp?
            # time_step is observation at timestamp?
            # This is a bit different as we need to use policy to select actions.
            # Want to select actions from possible actions
            action_spec = self.agent.policy.action(timestamp)

            configs = [dict() for _ in range(nchoose)]
            for sensor_action_assignment in configs:
                for sensor in self.sensors:
                    chosen_actions = []
                    action_step = self.agent.policy.action(timestamp)
                    action = self.agent.policy.action(action_step)
                    chosen_actions.append(action)
                    sensor_action_assignment[sensor] = chosen_actions

            return configs
