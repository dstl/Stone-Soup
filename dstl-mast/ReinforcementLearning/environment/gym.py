import math
import os
import random
from datetime import datetime, timedelta
from os import path as osp
from typing import Any, TypeVar

import gymnasium as gym
import imageio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import PIL
from ordered_set import OrderedSet
from ruamel.yaml import YAML
from stonesoup.predictor.kalman import ExtendedKalmanPredictor
from stonesoup.types.detection import Detection
from stonesoup.types.groundtruth import *
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.state import *
from stonesoup.types.track import Track
from stonesoup.updater.kalman import ExtendedKalmanUpdater

from ReinforcementLearning.utils.scenario_utils import _get_yaml_scenario

stonesoup_yaml = YAML(typ=["rt", "stonesoup"], plug_ins=["stonesoup.serialise"])

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class StoneSoupEnv(gym.Env):
    """OpenAI gymnasium v0.26 Environment convention:
    https://gymnasium.farama.org/content/migration-guide/"""

    """Declare initial set up for env"""

    def __init__(
        self,
        scenario_config="ReinforcementLearning/configs/scenario_config.yaml",
        render_episodes=False,
        log_dir="ReinforcementLearning/logs/renders/",
    ):
        self.render_episodes = render_episodes
        self.log_dir = log_dir

        if self.render_episodes:
            self._plot = plt.imshow(np.zeros((2000, 2000)))
            plt.show(block=False)
        self.step_count = 0
        self.step_time = datetime.datetime.now()

        """Declare initial set up for env"""
        self._set_action_space()
        self._scenario_creator(scenario_config)
        self.scenario_bounds = (
            -self.scenario_items["scenario_dimensions"][0] / 2,
            self.scenario_items["scenario_dimensions"][0] / 2,
        )

        self._set_observation_space()
        self.agent = self.scenario_items["sensor_manager"]["platform"]
        self.agent.states[0].timestamp = self.step_time
        self.frames = []
        self.previous_distance = 0

    def step(
        self, action: ActType
    ) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
        """Override for `step` in gymnasium API.

        Takes action from agent and progresses the environment to the next
        discrete time step, returning agent states, rewards, etc.

        Args:
            action: Agent's current action to be fed to environment.
        Returns:
            observation: Agent's new state after effecting action.
            reward: Scalar value representing the reward.
            terminated: Whether episode has ended.
            truncated: True if episode truncates due to time limit or
                a reason that is not defined as part of the task MDP
            info: Debug/diagnostic information.
        """

        self.step_count += 1
        self.step_time = self.step_time + timedelta(seconds=1)
        self._apply_action(action)
        self._step_targets()

        tracks = self._compute_obs()
        obs = self._normalise_obs(tracks)
        reward = self._compute_rewards(obs)

        info = self._compute_info()

        terminated = self._compute_done()
        truncated = False

        if self.render_episodes:
            plt.clf()
            img = self._set_data(tracks)
            self._plot.set_data(img)
            plt.gcf().canvas.draw_idle()
            plt.gcf().canvas.start_event_loop(0.01)

            self.frames.append(img)

        return obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None) -> ObsType:
        """Reset the environment and return the initial agent state & info

        Args:
            seed: Used to set random seed state for reproducibility.
        Returns:
            observation: Initial agent state.
        """
        self.step_count = 0
        self._scenario_creator(reset=True)

        self.agent = self.scenario_items["sensor_manager"]["platform"]
        self.agent.states[0].timestamp = self.step_time

        # Initial detection is made on reset. Can change if necessary
        tracks = self._compute_obs()
        obs = self._normalise_obs(tracks)

        info = {}
        self.previous_distance = 0
        return obs, info

    def render(self):
        super().render()

    def close(self):
        if self.render_episodes:

            dtime = datetime.datetime.now().strftime("%d.%m.%Y_%H%s")
            if not osp.exists(self.log_dir):
                os.makedirs(self.log_dir, exist_ok=True)

            imageio.mimsave(
                self.log_dir + "render_" + str(dtime) + ".gif",
                self.frames,
                fps=10,
                loop=100,
            )

            plt.close()

        super().close()

    def _set_observation_space(self) -> None:
        """Set observation space attribute, using gymnasium.Spaces API.

        The observaation vector consists of 3 parts, the track observations,
        the platform observations, and the number of sensors.

        Obs vector - track obs + platform obs + n sensors

        Tracks consist of positional and directional data, with the values for all
        dimensions being represented.
        This means representing the position on the x axis, as well as the direction

        track obs - pos1_x   pos1_y  vel1_x   vel1_y  pos2_x   pos2_y   vel2_x   vel2_y
        pos3_x   pos3_y  vel3_x   vel3_y   pos4_x   pos4_y  vel4_x   vel4_y   pos5_x   pos5_y
        vel5_x  vel5_y

        platform obs - pos1_x   pos1_y   vel1_x   vel1_y

        n sensors obs - n_sensors

        n_targets [x, y * n_known_targets]

        """

        """
        We get the know and unknown target count to determine the observation vector size.
        We get the number of sensors for the agent to know how many are available.
        """
        n_known_targets = len(self.scenario_items["targets"])
        n_unknown_targets = len(self.scenario_items["unknown_targets"])

        """
        The x and y range of possible positions can be represented in normalised
        values between -1 and 1.
        The x and y directional values can be represented in normalised values
        between -1 and 1. Up and right are represented 1, down and left are
        represented by -1. We repeat this pattern of position and direction pairs 5
        times as this is the length of our track. We then concatenate this by the
        number of targets in the scenario (known and unknown).
        Structure: [x_pos, y_pos, x_vel, y_vel]
        """

        track_data_low = [0, 0, -1, -1] * 5 * (n_known_targets + n_unknown_targets)
        track_data_high = [2, 2, 2, 2] * 5 * (n_known_targets + n_unknown_targets)

        """
        The x and y range of possible positions can be represented in normalised
        values between 0 and 1.
        The x and y directional values can be represented in normalised values
        between -1 and 1. Up and right are represented 1, down and left are
        represented by -1.
        The x and y velocitiy values can be represented in normalised
        values between 0 and 1. Representing stationary to full velocity.
        Structure: [x_pos, y_pos, x_vel, y_vel]
        """
        platform_data_low = [0, 0, -1, -1]
        platform_data_high = [1, 1, 1, 1]

        """
        The number of sensors per platform can be represents in normalised
        values ranging from 0 to 1.
        Example: If the maximum number of sensors is 5, then 4 sensors is
        represented as 0.75.
        """
        sensor_num_low = [0]
        sensor_num_high = [1]

        """
        The x and y range of possible positions can be represented in normalised
        values between 0 and 1.
        This applies to all known targets.
        """
        n_targets_low = [0, 0] * n_known_targets
        n_targets_high = [1, 1] * n_known_targets

        self.observation_space = gym.spaces.Box(
            low=np.array(
                track_data_low + platform_data_low + sensor_num_low + n_targets_low
            ),
            high=np.array(
                track_data_high + platform_data_high + sensor_num_high + n_targets_high
            ),
            dtype=np.float64,
        )

    def _set_action_space(self) -> None:
        """
        Set action space attribute, using gymnasium.Spaces API.
        We have movement of the platform as possible actions such that

        0 : stay where you are
        1 : go left
        2 : go right
        3 : go backwards
        4 : go forwards
        5 : go diagonal forward left
        6 : go diagonal forward right
        7 : go diagonal backward left
        8 : go diagonal backward right

        """

        self.actions_meta = [
            "stay where you are",
            "go left",
            "go right",
            "go backwards",
            "go forwards",
            "go diagonal forward left",
            "go diagonal forward right",
            "go diagonal backward left",
            "go diagonal backward right",
        ]
        self.state_vectors = [
            [0.00001, 0.00001],
            [-2, 0],
            [2, 0],
            [0, -2],
            [0, 2],
            [-2, 2],
            [2, 2],
            [-2, -2],
            [2, -2],
        ]

        self.action_space = gym.spaces.Discrete(9)

    def _scenario_creator(
        self,
        config_path="ReinforcementLearning/configs/scenario_config.yaml",
        reset=False,
    ):
        """
        Function to import scenario config file,
        and generate scenario items accordingly.
        Creates known targets from 'targets' in
        scenario config and adds metadata 'known'
        Creates unknown targets at random positions
        and adds metadata 'unknown'
        Adds known target current positions to scenario items

        ALL targets are initialised with the step count in the env,
        i.e. all are initialised together
        """
        # Put back into scenario items
        self.step_time = datetime.datetime.now()

        self.scenario_items = _get_yaml_scenario(config_path)
        unknown_targets = []

        for target_number in range(self.scenario_items["unknown_targets"]):

            # generate ground truth state + path
            rand_state = np.random.rand(4)
            rand_state[1] = 2 * rand_state[1] - 1
            rand_state[3] = 2 * rand_state[3] - 1
            ground_truth = GroundTruthState(
                rand_state,
                metadata={"type": "unknown"},
                timestamp=self.step_time,
            )
            target_path = GroundTruthPath([ground_truth])

            target_path.transition_model = random.choice(
                [
                    self.scenario_items["transition_model"],
                    self.scenario_items["turn_model"],
                ]
            )

            unknown_targets.append(target_path)

        self.scenario_items["unknown_targets"] = unknown_targets

        for target in self.scenario_items["targets"]:
            target.state.timestamp = self.step_time
            target.transition_model = random.choice(
                [
                    self.scenario_items["transition_model"],
                    self.scenario_items["turn_model"],
                ]
            )

        self.scenario_items["known_target_positions"] = [
            [target[-1].state_vector[0], target[-1].state_vector[2]]
            for target in self.scenario_items["targets"]
        ]

        # Randomise target position and velocity on reset
        if reset:
            for target in self.scenario_items["targets"]:
                rand_state = np.random.rand(4)
                rand_state[1] = 2 * rand_state[1] - 1
                rand_state[3] = 2 * rand_state[3] - 1
                target.states[0] = GroundTruthState(
                    rand_state, metadata={"type": "known"}, timestamp=self.step_time
                )

                target.transition_model = random.choice(
                    [
                        self.scenario_items["transition_model"],
                        self.scenario_items["turn_model"],
                    ]
                )

            rand_state = np.random.rand(4)
            rand_state[1] = 2 * rand_state[1] - 1
            rand_state[3] = 2 * rand_state[3] - 1
            self.scenario_items["sensor_manager"][
                "platform"
            ].movement_controller.states = [
                State(state_vector=StateVector(rand_state), timestamp=self.step_time)
            ]

        self.scenario_items["discovered_targets"] = []

        targets = (
            self.scenario_items["targets"] + self.scenario_items["unknown_targets"]
        )

        self.measurements = {}
        for i in range(len(targets)):
            self.measurements[f"measurements_{i+1}"] = []

        self.past_observations = {}
        # StateVector([np.inf, np.inf, np.inf, np.inf]) means no detection for
        # this timestep
        for k in range(len(targets)):
            self.past_observations[f"target_{k+1}"] = [
                StateVector([np.inf, np.inf, np.inf, np.inf])
            ] * 5

        self.tracks = {}
        # All initial priors are initial GroundTruthStates
        for idx, target in enumerate(targets):
            self.tracks[f"track_{idx+1}"] = Track(
                GaussianState(
                    target.state_vector,
                    np.diag([1.5, 0.5, 1.5, 0.5]),
                    timestamp=self.step_time,
                )
            )

    def _apply_action(self, action):

        desired_agent_state = State(
            [
                [self.agent.position[0]],
                [self.state_vectors[action][0]],
                [self.agent.position[1]],
                [self.state_vectors[action][1]],
            ]
        )

        state_vector = self.agent.transition_model.function(
            state=desired_agent_state,
            noise=False,
            timestamp=self.step_time,
            time_interval=self.step_time - self.agent.state.timestamp,
        )
        new_state = State.from_state(
            self.agent.state, state_vector=state_vector, timestamp=self.step_time
        )
        self.agent.states.append(new_state)

    def _compute_obs(self):
        """
        Return StateVector of last 5 tracks for each target.
        StateVector([np.inf,np.inf,np.inf,np.inf]) means mo detection
        Most recent position at end of list
        """
        # Only for one sensor, can generalise
        targets = (
            self.scenario_items["targets"] + self.scenario_items["unknown_targets"]
        )

        # Currently end of list is most recent
        # TODO: Detectable if detectable by any sensor
        sensor = self.scenario_items["sensor_manager"]["platform"].sensors[0]
        detectable_targets = [sensor.is_detectable(target[-1]) for target in targets]

        self.discovered_this_step = 0
        for idx, target in enumerate(targets):
            if detectable_targets[idx]:

                # TODO: loop through sensors
                measurement_model = (
                    self.scenario_items["sensor_manager"]["platform"]
                    .sensors[0]
                    .measurement_model
                )

                measurement = measurement_model.function(target[-1], noise=False)

                self.measurements[f"measurements_{idx+1}"].append(
                    Detection(
                        measurement,
                        timestamp=self.step_time,
                        measurement_model=measurement_model,
                    )
                )

                # TODO: Needs to be put in yaml generator
                predictor = ExtendedKalmanPredictor(target.transition_model)
                updater = ExtendedKalmanUpdater(measurement_model)

                # If detected for first time prior will be last time it was detected
                # TODO: Add data association for multi-target
                prediction = predictor.predict(
                    self.tracks[f"track_{idx+1}"][-1], timestamp=self.step_time
                )
                hypothesis = SingleHypothesis(
                    prediction, self.measurements[f"measurements_{idx+1}"][-1]
                )
                post = updater.update(hypothesis)

                self.tracks[f"track_{idx+1}"].append(post)

                self.past_observations[f"target_{idx+1}"].append(post.state_vector)
                self.past_observations[f"target_{idx+1}"].pop(0)

                if idx > len(self.scenario_items["targets"]) & (
                    target.id not in self.scenario_items["discovered_targets"]
                ):
                    self.discovered_this_step += 1
                    self.scenario_items["discovered_targets"].append(target.id)

            else:
                self.past_observations[f"target_{idx+1}"].append(
                    StateVector([np.inf, np.inf, np.inf, np.inf])
                )
                self.past_observations[f"target_{idx+1}"].pop(0)

        tracks = []
        for lst in self.past_observations.values():
            tracks += [lst]

        return tracks

    def _compute_done(self):

        if self.step_count >= self.scenario_items["episode_threshold"]:
            return True

        is_outside = self._outside_scenario(scenario_bounds=self.scenario_bounds, platform=self.agent)

        if is_outside:
            return True

        return False

    def _step_targets(self):
        """
        Function to move targets in scenario
        Takes transition model and applies it to
        generate next states for all targets.

        Upon generating new target states,
        known target locations are updated in scneario items
        """

        targets = (
            self.scenario_items["targets"] + self.scenario_items["unknown_targets"]
        )
        transition_model = self.scenario_items["transition_model"]
        self.truths = OrderedSet()

        for idx, target in enumerate(targets):
            target.transition_model = transition_model

            state_vector = target.transition_model.function(
                state=target.state,
                noise=False,
                timestamp=self.step_time,
                time_interval=self.step_time - target.state.timestamp,
            )

            new_state = State.from_state(
                target.state, state_vector=state_vector, timestamp=self.step_time
            )

            new_state = self._outside_scenario(scenario_bounds=self.scenario_bounds, target=target, new_state=new_state)

            target.states.append(new_state)

            self.truths.add(target)

            # update target positions of known targets
            if target.state.metadata["type"] == "known":
                self.scenario_items["known_target_positions"][idx] = [
                    target[-1].state_vector[0],
                    target[-1].state_vector[2],
                ]

    def _normalise_obs(self, tracks: list) -> list:

        platform_state = self.agent.state

        bounds = ((0, 1), (-1, 1))

        norm_platform_state = self._normalise_values(
            state=platform_state.state_vector, bounds=bounds, clip=True
        )

        norm_tracks = (
            np.array(
                [
                    [
                        self._normalise_values(
                            state=track_state, bounds=bounds, clip=True
                        )
                        for track_state in track
                    ]
                    for track in tracks
                ]
            )
            .astype(np.float64)
            .flatten()
            .tolist()
        )

        max_sensors = self.scenario_items["sensor_manager"]["max_sensors_per_platform"]
        n_sensors = len(self.scenario_items["sensor_manager"]["platform"].sensors)

        norm_sensors = [n_sensors / max_sensors]

        norm_target_positions = (
            np.array(
                [
                    self._normalise_values(position=position, bounds=bounds, clip=True)
                    for position in self.scenario_items["known_target_positions"]
                ]
            )
            .astype(np.float64)
            .flatten()
            .tolist()
        )

        normalised_obs = (
            norm_tracks + norm_platform_state + norm_sensors + norm_target_positions
        )

        return normalised_obs

    def _normalise_values(
        self, state=None, position=None, bounds=tuple, clip: bool = True
    ) -> list:

        if position:
            x, y = position
        else:
            if all(np.isinf(value) for value in state):
                return [2, 2, 2, 2]

            x, y = state[0], state[2]

        width_x, width_y = self.scenario_items["scenario_dimensions"]

        min_x, min_y = (
            -(width_x / 2),
            -(width_y / 2),
        )

        norm_x = (x - min_x) / width_x
        norm_y = (y - min_y) / width_y

        if position:
            norm_x, norm_y = np.clip(
                [norm_x, norm_y], a_min=bounds[0][0], a_max=bounds[0][1]
            )
            return [norm_x, norm_y]

        vel_limit = self.scenario_items["velocity_limit"]
        vx, vy = state[1], state[3]

        norm_vx = vx / vel_limit
        norm_vy = vy / vel_limit

        norm_x, norm_y = np.clip(
            [norm_x, norm_y], a_min=bounds[0][0], a_max=bounds[0][1]
        )
        norm_vx, norm_vy = np.clip(
            [norm_vx, norm_vy], a_min=bounds[1][0], a_max=bounds[1][1]
        )

        norm_state = [norm_x, norm_y, norm_vx, norm_vy]

        return norm_state

    def _compute_rewards(self, obs):
        """
        Function to calculate rewards for tracking objects.
        Reward structure:
            +5 for each known target tracked
            -10 for each known target lost
            +10 for discovering new target -- one time reward
            +5 for tracking discovered target
            -10 for unknown target lost
        """

        reward = 0

        total_known_targets = len(self.scenario_items["targets"])
        total_unknown_targets = len(self.scenario_items["unknown_targets"])

        # Get number of tracks per target to slice obs on those indices
        known_target_tracks = 4 * 5 * total_known_targets
        unknown_target_tracks = 4 * 5 * total_unknown_targets

        # Rewards for known targets
        known_target_obs = obs[:known_target_tracks]
        known_target_obs = [
            known_target_obs[idx : idx + 20]
            for idx in range(0, len(known_target_obs), 20)
        ]

        for track in known_target_obs:
            if all(value == 2 for value in track[-4:]):
                # Lost known target
                reward -= 10
            else:
                # Known target still being tracked
                reward += 5

        # Rewards for unknown targets
        unknown_target_obs = obs[
            known_target_tracks : known_target_tracks + unknown_target_tracks
        ]
        unknown_target_obs = [
            unknown_target_obs[idx : idx + 20]
            for idx in range(0, len(unknown_target_obs), 20)
        ]

        for track in unknown_target_obs:
            # TODO dont penalise if target isnt discovered and isnt being tracked
            if all(value == 2 for value in track[-4:]):
                # Lost unknown target
                reward -= 10
            else:
                # unknown target still being tracked
                reward += 5

        # Positive reward for discovering target THIS timestep
        reward += 10 * self.discovered_this_step

        # Calculate current cumulative distance
        target_states = []
        for target in self.scenario_items["targets"]:
            target_states.append(target.states[-1])
        for target in self.scenario_items["unknown_targets"]:
            target_states.append(target.states[-1])

        platform_state = self.scenario_items["sensor_manager"][
            "platform"
        ].movement_controller.states[-1]
        p_coords = [platform_state.state_vector[0], platform_state.state_vector[2]]

        current_cumulative_distance = 0
        for state in target_states:
            t_coords = [state.state_vector[0], state.state_vector[2]]
            current_cumulative_distance += math.dist(t_coords, p_coords)

        # Range normalising constant for calculating distance based reward
        range_normalising_constant = 0.5

        # Higher reward if closer to target, lower if farther from previous state
        reward += range_normalising_constant * (self.previous_distance - current_cumulative_distance)

        return reward

    def _compute_info(self):

        info = {}
        target_states = []
        for target in self.scenario_items["targets"]:
            target_states.append(target.states[-1])
        for target in self.scenario_items["unknown_targets"]:
            target_states.append(target.states[-1])

        platform_state = self.scenario_items["sensor_manager"][
            "platform"
        ].movement_controller.states[-1]
        p_coords = [platform_state.state_vector[0], platform_state.state_vector[2]]

        cumulative_distance = 0
        for state in target_states:
            t_coords = [state.state_vector[0], state.state_vector[2]]
            cumulative_distance += math.dist(t_coords, p_coords)

        info["cumulative_distance"] = cumulative_distance
        self.previous_distance = cumulative_distance

        sensor = self.scenario_items["sensor_manager"]["platform"].sensors[0]
        targets = (
            self.scenario_items["targets"] + self.scenario_items["unknown_targets"]
        )
        detectable_targets = [1 if sensor.is_detectable(target[-1]) else 0 for target in targets]
        info['detected_this_step'] = sum(detectable_targets)
        info['agent_outside'] = self._outside_scenario(scenario_bounds=self.scenario_bounds, platform=self.agent)

        return info

    def _set_data(self, tracks):

        fig = plt.gcf()
        ax = plt.gca()
        plt.grid()

        colors = ["g", "c", "r", "m", "b"]
        labels = ["Known Target", "Unknown Target", "Platfrom", "Track", "Detection"]

        """Establish Legend"""
        test_list = []
        for j in range(len(colors)):
            test_list.append(matplotlib.patches.Patch(color=colors[j], label=labels[j]))
        legend = ax.legend(handles=test_list, loc="upper right")
        ax.add_artist(legend)

        x_range = self.scenario_items["scenario_dimensions"][0]
        y_range = self.scenario_items["scenario_dimensions"][1]

        """Set env dimensions"""
        ax.set_xlabel("$East$")
        ax.set_ylabel("$North$")
        ax.set_ylim(-(y_range / 2), y_range / 2)
        ax.set_xlim(-(x_range / 2), x_range / 2)

        """Plot the ground truth path of known targets"""
        for gtruth in self.scenario_items["targets"]:
            X = [state.state_vector[0] for state in gtruth.states[:-1]][-10:]
            Y = [state.state_vector[2] for state in gtruth.states[:-1]][-10:]
            ax.plot(X, Y, color="g", linestyle="dashed")

        """Plot the ground truth path of unknown targets"""
        for hgtruth in self.scenario_items["unknown_targets"]:
            X = [state.state_vector[0] for state in hgtruth.states[:-1]][-10:]
            Y = [state.state_vector[2] for state in hgtruth.states[:-1]][-10:]
            ax.plot(X, Y, color="c", linestyle="dashed")

        """Plot the platform (either as a single point or a path)"""
        # Uncomment for trail
        # X = [state.state_vector[0] for state in self.logs["platform_states"][:i]]
        # Y = [state.state_vector[2] for state in self.logs["platform_states"][:i]]

        # Uncomment for marker
        X = [self.agent.states[-1].state_vector[0]]
        Y = [self.agent.states[-1].state_vector[2]]

        ax.plot(X, Y, marker="x", color="r")

        """Plot tracks"""
        track_list = []
        state_list = []

        """Code for generating structured tracks"""
        for track_inst in tracks:
            """Use to provide accurate timestamp for each state"""
            time_track = 0
            for state_inst in track_inst:
                """Generate states for each track"""
                state_list.append(
                    State(state_inst, self.step_time - timedelta(seconds=time_track))
                )
                time_track += 1

            """Make track out of list of states"""
            track = Track(state_list)
            state_list = []
            """Cumulative list of tracks over all steps"""
            track_list.append(track)
            time_track = 0

        for track in track_list:
            X = []
            Y = []
            for state in track.states:
                if state.state_vector[0] != np.inf:
                    X.append(state.state_vector[0])
                    Y.append(state.state_vector[2])
            ax.plot(X, Y, color="m")

        """Plot detections"""
        X = []
        Y = []

        targets = (
            self.scenario_items["targets"] + self.scenario_items["unknown_targets"]
        )

        measure_frame = []
        for idx, target in enumerate(targets):
            measure_frame.append(self.measurements[f"measurements_{idx+1}"])

        for det in measure_frame:
            if len(det) >= 1:
                detection = det[-1]
                if detection.timestamp == self.step_time:
                    detection.measurement_model.mapping = [0, 1]
                    X.append(detection.measurement_model.inverse_function(detection)[0])
                    Y.append(detection.measurement_model.inverse_function(detection)[1])
                    ax.plot(X, Y, marker=".", color="b", linestyle="dashed")
                    X = []
                    Y = []

        fig.canvas.draw()
        img = PIL.Image.frombytes(
            "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
        )

        return img

    def _outside_scenario(self, target=None, new_state=None, platform=None, scenario_bounds=list):

        if platform:
            position = platform.position
        else:
            position = new_state.state_vector[0], new_state.state_vector[2]

        is_outside_x = (position[0] < scenario_bounds[0]) | (position[0] > scenario_bounds[1])
        is_outside_y = (position[1] < scenario_bounds[0]) | (position[1] > scenario_bounds[1])
        is_outside = is_outside_x | is_outside_y

        if is_outside:

            if platform:
                return is_outside

            target.transition_model = self.scenario_items['turn_model']

            state_vector = target.transition_model.function(
                state=target.state,
                noise=False,
                timestamp=self.step_time,
                time_interval=self.step_time - target.state.timestamp,
            )

            new_state = State.from_state(
                target.state, state_vector=state_vector, timestamp=self.step_time
            )

            return new_state

        return new_state
