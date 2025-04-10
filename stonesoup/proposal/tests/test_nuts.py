# nuts test, we should compare the NUTS bit to a gaussian distribution
import itertools
import datetime
import numpy as np

from stonesoup.proposal.nuts import NUTSProposal
from stonesoup.models.transition.linear import ConstantVelocity
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.types.state import ParticleState, State
from stonesoup.types.particle import Particle
from stonesoup.types.prediction import Prediction
from stonesoup.types.detection import Detection


def test_nuts():

    # Define time related variables
    timestamp = datetime.datetime.now()
    timediff = 2  # 2 sec
    new_timestamp = timestamp + datetime.timedelta(seconds=timediff)
    time_interval = new_timestamp - timestamp

    # parameters
    mapping = (0,)
    transition_model = ConstantVelocity([0])
    number_particles = 9
    measurement_model = LinearGaussian(
        ndim_state=2,
        mapping=mapping,
        noise_covar=np.diag([0]))

    # track states
    start_state = State(state_vector=[0, 1], timestamp=timestamp)
    next_state = State(transition_model.function(start_state, noise=False,
                                                 time_interval=time_interval),
                       timestamp=new_timestamp)

    # Define prior state
    prior_particles = [Particle(np.array([[i], [j]]), 1/number_particles)
                       for i, j in itertools.product([1, 2, 3], [1, 2, 3])]
    start = ParticleState(None, particle_list=prior_particles, timestamp=timestamp)

    # find a detection
    detection = Detection(measurement_model.function(next_state, noise=False),
                          timestamp=new_timestamp,
                          measurement_model=measurement_model)

    # load the NUTS proposal
    proposal = NUTSProposal(transition_model=transition_model,
                            measurement_model=measurement_model,
                            step_size=1.0,
                            mass_matrix=np.eye(len(start.state_vector)),
                            mapping=mapping,
                            v_mapping=[1],
                            target_proposal_input=None,
                            grad_target=None,
                            num_dims=2,
                            num_samples=number_particles)

    # new state
    nuts_state = proposal.rvs(start, time_interval=time_interval)
    tx_state = transition_model.function(start, noise=False, time_interval=time_interval)

    # transform the prediction
    eval_prediction = Prediction.from_state(start,
                                            parent=start,
                                            state_vector=tx_state,
                                            timestamp=new_timestamp,
                                            transition_model=transition_model,
                                            prior=start)

    eval_mean = np.mean(np.hstack([i.state_vector for i in eval_prediction]),
                        axis=1).reshape(2, 1)

    nuts_mean = np.mean(np.hstack([i.state_vector for i in nuts_state]),
                        axis=1).reshape(2, 1)

    # case without a detection
    assert np.all([eval_prediction.state_vector[:, i] ==
                   nuts_state.state_vector[:, i] for i in range(9)])
    assert np.allclose(nuts_mean, eval_mean)
