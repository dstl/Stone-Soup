import numpy as np
import pytest

from ..linear import LinearControlModel
from ....types.array import StateVector
from ....types.state import State


@pytest.mark.parametrize("control_input, control_model", [
        (State(StateVector([1])),
            LinearControlModel(np.array([[0.25], [1]]))),
        (State(StateVector([1, 2, 5])),
            LinearControlModel(np.array([[0.25, 0, 0],
                                         [1, 0, 0],
                                         [0, 0.25, 0],
                                         [0, 1, 0],
                                         [0, 0, 0.25],
                                         [0, 0, 1]]), control_noise=np.eye(3)))
    ],
    ids=["Control1D", "Control3D"]
)
def test_linear_model(control_input, control_model):

    assert np.all(np.isclose(control_model.function(control_input, noise=False),
           control_model.matrix() @ control_input.state_vector))
    assert control_model.ndim == np.shape(control_model.control_noise)[1]
