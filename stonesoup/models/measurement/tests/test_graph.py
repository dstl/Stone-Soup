import pytest
import numpy as np
from scipy.stats import multivariate_normal as mvn

from stonesoup.models.measurement.graph import OptimalPathToDestinationMeasurementModel
from stonesoup.types.array import StateVector
from stonesoup.types.detection import Detection
from stonesoup.types.graph import RoadNetwork
from stonesoup.types.state import State


# The next fixture defines a simple network with 4 nodes and 4 edges that looks like:
#        (.25)
#       3-->--4
#       |     |
#  (.5) ^     ^ (1.)     where the numbers in brackets are the edge weights and the arrows
#       |     |          indicate the direction of the edge
#       1-->--2
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


@pytest.mark.parametrize(
    'use_indicator',
    [True, False],
)
def test_shortest_path_model(graph, use_indicator):
    # Measurement model
    mapping = [0, 1]
    R = np.eye(2) * 0.0002
    measurement_model = OptimalPathToDestinationMeasurementModel(ndim_state=5, mapping=mapping,
                                                                 noise_covar=R, graph=graph,
                                                                 use_indicator=use_indicator)

    # Test the function method (without noise)
    state_vector = StateVector([0, 1, 1, 3, 1])
    state = State(state_vector)
    meas_state_vector = measurement_model.function(state, noise=False)
    # The state vector should be equal to the position of node 1
    eval_state_vector = StateVector(graph.nodes[1]['pos'])
    assert np.array_equal(meas_state_vector, eval_state_vector)

    # Test the function method (with noise)
    seed = 1994
    np.random.seed(seed)    # Set the seed for reproducibility
    meas_state_vector = measurement_model.function(state, noise=True)
    # The state vector should be equal to the position of node 1 plus the noise
    np.random.seed(seed)    # Reset the seed to the same value
    noise = measurement_model.rvs()
    eval_state_vector = StateVector(graph.nodes[1]['pos']) + noise
    assert np.array_equal(meas_state_vector, eval_state_vector)

    # Test the logpdf method (no path to destination)
    # Assume the state is at edge 2 (2->4), the source node is 2 and the destination is node 3
    state_vector = StateVector([0, 1, 2, 3, 2])
    state = State(state_vector)
    detection = Detection(StateVector(graph.nodes[2]['pos']) + noise)  # Create a detection
    likelihood = measurement_model.logpdf(detection, state)
    if use_indicator:
        # The likelihood should be equal to -np.inf
        assert likelihood == -np.inf
    else:
        # The likelihood should be equal to the likelihood of the noise
        assert np.isclose(likelihood, mvn.logpdf(noise.T, mean=np.zeros(2), cov=R))

    # Test the logpdf method (current edge is not in the shortest path to destination)
    # Assume the state is at edge 0 (1->2), the source node is 1 and the destination is node 4
    # (i.e. the shortest path is 1->3->4)
    state_vector = StateVector([0, 1, 0, 4, 1])
    state = State(state_vector)
    detection = Detection(StateVector(graph.nodes[1]['pos']) + noise)  # Create a detection
    likelihood = measurement_model.logpdf(detection, state)
    if use_indicator:
        # The likelihood should be equal to -np.inf
        assert likelihood == -np.inf
    else:
        # The likelihood should be equal to the likelihood of the noise
        assert np.isclose(likelihood, mvn.logpdf(noise.T, mean=np.zeros(2), cov=R))
