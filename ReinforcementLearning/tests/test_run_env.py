import os
import sys

sys.path.insert(0, os.getcwd())
from ReinforcementLearning.environment.gym import StoneSoupEnv
from ReinforcementLearning.scripts.run_env import run

DEFAULT_RENDERING = False
DEFAULT_ITERS_PER_EPISODE = 30
DEFAULT_NO_BAR = True


def test_run(
    render=DEFAULT_RENDERING, iters=DEFAULT_ITERS_PER_EPISODE, no_bar=DEFAULT_NO_BAR
):
    """
    Test run function

    Tests run function input
    and attribute by type checking.
    """

    # Run function
    run(render, iters, no_bar)

    assert isinstance(render, bool)
    assert isinstance(iters, int)
    assert isinstance(no_bar, bool)
    assert isinstance(run.env, StoneSoupEnv)

    # Successful run
    assert True
