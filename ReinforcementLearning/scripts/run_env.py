import argparse
import os
import sys

from tqdm import trange

sys.path.insert(0, os.getcwd())
from ReinforcementLearning.environment.gym import StoneSoupEnv

# Environment default parameters
DEFAULT_RENDERING = False

# Serve default parameters
DEFAULT_ITERS_PER_EPISODE = 50
DEFAULT_SERVE_PORT = 7999
DEFAULT_USE_SERVE = False
DEFAULT_NO_BAR = True
DEAFULT_SCENARIO_CONFIG = "ReinforcementLearning/configs/scenario_config.yaml"
DEFAULT_LOG_DIR = "ReinforcementLearning/logs/renders/"


def run(
    render=DEFAULT_RENDERING,
    iters=DEFAULT_ITERS_PER_EPISODE,
    no_bar=DEFAULT_NO_BAR,
):
    # Create the environment with or without video capture
    run.env = StoneSoupEnv(
        scenario_config=DEAFULT_SCENARIO_CONFIG,
        render_episodes=render,
        log_dir=DEFAULT_LOG_DIR,
    )

    # Run the simulation ###################################
    for ep in range(5):

        print("New episode starting")
        print("FULL ACTION SPACE:::::::", run.env.action_space)
        print("FULL OBSERVATION SPACE::::::", run.env.observation_space)

        obs, info = run.env.reset()

        with trange(iters, disable=no_bar) as t:

            for j in t:
                # Sampling a random action
                print("Position ", run.env.agent.position)
                act = run.env.action_space.sample()

                # If sensor actions are included
                if run.env.scenario_items["actionable_sensors"] is True:
                    print("Random platform action:", run.env.actions_meta[act[0]])

                    # For each actionable
                    num_actionable_sensors = len(run.env.sensor_level_actions)

                    # For each sensor with actionables
                    for x in range(0, num_actionable_sensors):

                        # For each actionable in this particular sensor, add to multi discrete space with the number of options in the dictionary value.
                        for i, j in enumerate(run.env.sensor_level_actions[x]):
                            print(
                                "Random sensor "
                                + str(x)
                                + ", action "
                                + str(j)
                                + ": "
                                + str(act[x + i + 1])
                            )

                    # Action summary of each discrete action
                    print("Action summary: " + str(act))
                else:
                    # Just platform actions
                    print("Random platform action:", run.env.actions_meta[act])

                obs, reward, terminated, truncated, info = run.env.step(act)

                if terminated:
                    print("Done")
                    break

    run.env.close()


if __name__ == "__main__":
    # Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--no_bar",
        default=DEFAULT_NO_BAR,
        type=bool,
        help="Render progress bar or not",
    )
    parser.add_argument("-i", "--iters", type=int, default=DEFAULT_ITERS_PER_EPISODE)
    parser.add_argument(
        "-r", "--render", action="store_true", default=DEFAULT_RENDERING
    )

    ARGS = parser.parse_args()

    run(**vars(ARGS))
