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
DEFAULT_LOG_DIR = 'ReinforcementLearning/logs/renders/'


def run(
    render=DEFAULT_RENDERING,
    iters=DEFAULT_ITERS_PER_EPISODE,
    no_bar=DEFAULT_NO_BAR,
):
    # Create the environment with or without video capture
    run.env = StoneSoupEnv(scenario_config=DEAFULT_SCENARIO_CONFIG,
                           render_episodes=render,
                           log_dir=DEFAULT_LOG_DIR)

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
                print("Random action:", run.env.actions_meta[act])

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
    parser.add_argument("-r", "--render", action="store_true", default=DEFAULT_RENDERING)

    ARGS = parser.parse_args()

    run(**vars(ARGS))
