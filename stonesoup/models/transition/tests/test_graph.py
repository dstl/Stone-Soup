import datetime

import numpy as np
import pytest
from scipy.stats import multivariate_normal as mvn

try:
    from stonesoup.models.transition.graph import OptimalPathToDestinationTransitionModel
    from stonesoup.types.graph import RoadNetwork
except ImportError:
    # Catch optional dependencies import error
    pytest.skip(
        "Skipping due to missing optional dependencies. Usage of the road network classes requires"
        "that the optional package dependencies 'geopandas' and 'networkx' are installed.",
        allow_module_level=True
    )
from stonesoup.models.transition.linear import ConstantVelocity
from stonesoup.types.array import StateVectors
from stonesoup.types.numeric import Probability
from stonesoup.types.state import ParticleState


# The next fixture defines a simple network with 4 nodes and 4 edges that looks like:
#        (.25)
#       2-->--3
#       |     |
#  (.5) ^     ^ (1.)     where the numbers in brackets are the edge weights and the arrows
#       |     |          indicate the direction of the edge
#       0-->--1
#        (1.)
@pytest.fixture()
def graph():
    # Nodes are defined as a tuple of (node, {'pos': (x, y)})
    nodes = [(1, {'pos': (0, 0)}),
             (2, {'pos': (1, 0)}),
             (3, {'pos': (0, 1)}),
             (4, {'pos': (1, 1)})]
    # Edges are defined as a tuple of (node1, node2, {'weight': weight})
    edges = [(1, 2, {'weight': 1.}),
             (1, 3, {'weight': .5}),
             (2, 4, {'weight': 1.}),
             (3, 4, {'weight': .25})]
    # Create the network
    network = RoadNetwork()
    network.add_nodes_from(nodes)
    network.add_edges_from(edges)
    return network


@pytest.fixture(scope='function')
def eval_fixture(request):
    """Fixture for evaluating the OptimalPathToDestinationTransitionModel model."""
    dest_resample_prob, possible_destinations = request.param
    if not dest_resample_prob and not possible_destinations:
        eval_state_vectors = StateVectors([[0.03944934, 0., 1., 0.,
                                            0.77751235, 0., 0., 0.,
                                            0.59963901, -1.05445506],
                                           [0.09586569, -0.50258062, 2.50743876, -2.1222713,
                                            1.61676025, -0.50404886, 0.85937104, -1.64991066,
                                            -0.30328854, -1.00153735],
                                           [1., 2., 2., 1., 2., 2., 2., 1., 1., 1.],
                                           [3., 2., 3., 2., 4., 4., 2., 4., 2., 1.],
                                           [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]])
        eval_lik = np.array([Probability(0.5424396224818488),
                             Probability(0.33984742421943714),
                             Probability(0.013239952300537344),
                             Probability(0.00011525003806155926),
                             Probability(0.1574757882786008),
                             Probability(0.3687520172565367),
                             Probability(0.10141649505352639),
                             Probability(0.0028273529968739467),
                             Probability(0.12782144965036057),
                             Probability(0.04390143816991418)])
    elif not dest_resample_prob:
        eval_state_vectors = StateVectors([[0.03944934, 0., 1., 0.,
                                            0.77751235, 0., 0., 0.,
                                            0.59963901, 0.],
                                           [0.09586569, -0.50258062, 2.50743876, -2.1222713,
                                            1.61676025, -0.50404886, 0.85937104, -1.64991066,
                                            -0.30328854, -1.00153735],
                                           [1., 2., 2., 1., 2., 2., 2., 1., 1., 1.],
                                           [2., 4., 2., 4., 4., 4., 4., 4., 4., 2.],
                                           [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]])
        eval_lik = np.array([Probability(0.5424396224818488),
                             Probability(0.33984742421943714),
                             Probability(0.013239952300537344),
                             Probability(0.00011525003806155926),
                             Probability(0.1574757882786008),
                             Probability(0.3687520172565367),
                             Probability(0.10141649505352639),
                             Probability(0.0028273529968739467),
                             Probability(0.12782144965036057),
                             Probability(0.08052260260315869)])
    elif not possible_destinations:
        eval_state_vectors = StateVectors([[0.03944934, 0., 1., 0.,
                                            0.77751235, 0., 0., 0.,
                                            0.59963901, 0.],
                                           [0.09586569, -0.50258062, 2.50743876, -2.1222713,
                                            1.61676025, -0.50404886, 0.85937104, -1.64991066,
                                            -0.30328854, -1.00153735],
                                           [1., 2., 2., 1., 2., 2., 2., 1., 1., 1.],
                                           [3., 2., 3., 2., 4., 4., 2., 4., 2., 3.],
                                           [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]])
        eval_lik = np.array([Probability(0.5424396224818488),
                             Probability(0.33984742421943714),
                             Probability(0.013239952300537344),
                             Probability(0.0),
                             Probability(0.1574757882786008),
                             Probability(0.3687520172565367),
                             Probability(0.10141649505352639),
                             Probability(0.0),
                             Probability(0.0),
                             Probability(0.0)])
    else:
        eval_state_vectors = StateVectors([[0.03944934, 0., 1., 0.,
                                            0.77751235, 0., 0., 0.,
                                            0.59963901, 0.],
                                           [0.09586569, -0.50258062, 2.50743876,
                                            -2.1222713, 1.61676025, -0.50404886,
                                            0.85937104, -1.64991066, -0.30328854, -1.00153735],
                                           [1., 2., 2., 1., 2., 2., 2., 1., 1., 1.],
                                           [2., 4., 2., 4., 4., 4., 4., 4., 4., 4.],
                                           [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]])
        eval_lik = np.array([Probability(0.0),
                             Probability(0.33984742421943714),
                             Probability(0.013239952300537344),
                             Probability(0.00011525003806155926),
                             Probability(0.1574757882786008),
                             Probability(0.3687520172565367),
                             Probability(0.10141649505352639),
                             Probability(0.0028273529968739467),
                             Probability(0.12782144965036057),
                             Probability(0.08052260260315869)])
    return dest_resample_prob, possible_destinations, eval_state_vectors, eval_lik


@pytest.mark.parametrize(
    "eval_fixture",
    [
        (0, None),
        (0, [2, 4]),
        (0.5, None),
        (0.5, [2, 4])
    ],
    indirect=True
)
def test_shortest_path_model(eval_fixture, graph):
    # Extract pytest parameters
    dest_resample_prob, possible_destinations, eval_state_vectors, eval_lik = eval_fixture

    # Seed for reproducibility
    seed = 1994
    np.random.seed(seed)

    # State related variables
    num_particles = 10
    speed = 0.1
    init_destinations = possible_destinations if possible_destinations \
        else [i + 1 for i in range(graph.number_of_nodes())]
    old_timestamp = datetime.datetime.now()
    timediff = 1  # 1sec
    new_timestamp = old_timestamp + datetime.timedelta(seconds=timediff)
    time_interval = new_timestamp - old_timestamp
    prior_sv = StateVectors([
        np.zeros((num_particles,)),  # r
        mvn.rvs(0, speed, (num_particles,)),  # r_dot
        np.random.choice([1, 2], num_particles),  # edge
        np.random.choice(init_destinations, (num_particles,)),  # destination node
        np.full((num_particles,), 1)  # source node
    ])
    state = ParticleState(prior_sv, timestamp=old_timestamp)

    # Model-related components
    cv_model = ConstantVelocity(1)
    transition_model = OptimalPathToDestinationTransitionModel(
        transition_model=cv_model,
        graph=graph,
        destination_resample_probability=dest_resample_prob,
        possible_destinations=possible_destinations,
        seed=seed
    )

    # Ensure ndim_state operates as expected
    # --------------------------------------------------
    assert transition_model.ndim_state == 5

    # Ensure propagation operates as expected
    # --------------------------------------------------
    # Propagate a state vector through the model
    new_state_vectors = transition_model.function(state, time_interval=time_interval, noise=True)
    # Reproduce the propagation manually
    assert np.allclose(new_state_vectors, eval_state_vectors)

    # Ensure pdf operates as expected
    # --------------------------------------------------
    new_state = ParticleState(new_state_vectors, timestamp=new_timestamp)
    lik = transition_model.pdf(new_state, state, time_interval=time_interval)
    assert np.allclose(lik.astype(float), eval_lik.astype(float))
