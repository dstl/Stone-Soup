from ..base_driver import GaussianDriver
from ..driven import Process
import numpy as np
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from datetime import datetime, timedelta


def test_gaussian_process_model():
    seed = 1991
    mu_W = [2]
    sigma_W2 = np.eye(1)

    gaussian_driver = GaussianDriver(mu_W=mu_W, sigma_W2=sigma_W2, seed=seed)
    transition_model = Process(g_driver=gaussian_driver)

    jtimes = np.array([0.1, 0.5])
    dt = 1.

    assert(np.allclose(transition_model.matrix(), np.eye(1)))
    assert(np.allclose(transition_model.ext_input(), np.zeros((1, 1))))
    assert(np.allclose(transition_model.ft(dt=dt, jtimes=jtimes), np.ones_like(jtimes)[:, np.newaxis, np.newaxis]))
    # assert(np.allclose(transition_model.ft2(dt=dt, jtimes=jtimes), np.ones_like(jtimes)[:, np.newaxis, np.newaxis]))
    assert(np.allclose(transition_model.e_ft(dt=dt), dt * np.ones((1, 1))))
    # assert(np.allclose(transition_model.e_ft2(dt=dt), dt * np.eye(1)))
    assert(np.allclose(transition_model.e_gt(dt=dt), dt * np.ones((1, 1))))
    # assert(np.allclose(transition_model.e_gt2(dt=dt), dt * np.eye(1)))

    start_time = datetime.now().replace(microsecond=0)
    timesteps = [start_time]
    truth = GroundTruthPath([GroundTruthState([0], timestamp=timesteps[0])])

    expected_shape = (1, 1)
    num_steps = 10
    
    for k in range(1, num_steps + 1):
        timesteps.append(start_time+timedelta(seconds=k))  # add next timestep to list of timesteps
        state = transition_model.function(truth[k-1], noise=False, time_interval=timedelta(seconds=1))
        assert(state.shape == expected_shape)
        assert(np.allclose(state, np.zeros(1)))
        truth.append(GroundTruthState(
            state,
            timestamp=timesteps[k]))
    assert(len(truth) == num_steps + 1)
        
    for k in range(1, num_steps + 1):
        timesteps.append(start_time+timedelta(seconds=k))  # add next timestep to list of timesteps
        state = transition_model.function(truth[k-1], noise=True, time_interval=timedelta(seconds=1))
        assert(state.shape == expected_shape)
        assert(not np.allclose(state, np.zeros(1)))
        truth.append(GroundTruthState(
            state,
            timestamp=timesteps[k]))
    
    assert(len(truth) == 2 * num_steps + 1)

