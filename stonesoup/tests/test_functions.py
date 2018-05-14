import numpy as np

from stonesoup.functions import jacobian


def test_jacobian():
    """ jacobian function test """

    # State related variables
    state_mean = np.array([[3.0], [1.0]])

    def f(x):
        return np.array([[1, 1], [0, 1]])@x

    jac = jacobian(f, state_mean)
    jac = jac  # Stop flake8 unused warning
