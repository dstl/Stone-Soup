import pytest

import numpy as np


from stonesoup.architecture import InformationArchitecture
from ..edge import Edge, Edges
from ..node import RepeaterNode, SensorNode
from ...sensor.categorical import HMMSensor
from ...models.measurement.categorical import MarkovianMeasurementModel


@pytest.fixture
def fixtures():
    E = np.array([[0.8, 0.1],  # P(small | bike), P(small | car)
                  [0.19, 0.3],  # P(medium | bike), P(medium | car)
                  [0.01, 0.6]])  # P(large | bike), P(large | car)

    model = MarkovianMeasurementModel(emission_matrix=E,
                                      measurement_categories=['small', 'medium', 'large'])

    hmm_sensor = HMMSensor(measurement_model=model)

    node1 = SensorNode(sensor=hmm_sensor, label='1')
    node2 = SensorNode(sensor=hmm_sensor, label='2')
    node3 = SensorNode(sensor=hmm_sensor, label='3')
    node4 = SensorNode(sensor=hmm_sensor, label='4')
    node5 = SensorNode(sensor=hmm_sensor, label='5')
    node6 = SensorNode(sensor=hmm_sensor, label='6')
    node7 = SensorNode(sensor=hmm_sensor, label='7')

    nodes = [node1, node2, node3, node4, node5, node6, node7]

    hierarchical_edges = Edges(
        [Edge((node2, node1)), Edge((node3, node1)), Edge((node4, node2)), Edge((node5, node2)),
         Edge((node6, node3)), Edge((node7, node6))])

    centralised_edges = Edges(
        [Edge((node2, node1)), Edge((node3, node1)), Edge((node4, node2)), Edge((node5, node2)),
         Edge((node6, node3)), Edge((node7, node6)), Edge((node7, node5)), Edge((node5, node3))])

    simple_edges = Edges([Edge((node2, node1)), Edge((node3, node1))])

    linear_edges = Edges([Edge((node1, node2)), Edge((node2, node3)), Edge((node3, node4)),
                          Edge((node4, node5))])

    decentralised_edges = Edges(
        [Edge((node2, node1)), Edge((node3, node1)), Edge((node3, node4)), Edge((node3, node5)),
         Edge((node5, node4))])

    disconnected_edges = Edges([Edge((node2, node1)), Edge((node4, node3))])

    fixtures = dict()
    fixtures["hierarchical_edges"] = hierarchical_edges
    fixtures["centralised_edges"] = centralised_edges
    fixtures["decentralised_edges"] = decentralised_edges
    fixtures["nodes"] = nodes
    fixtures["simple_edges"] = simple_edges
    fixtures["linear_edges"] = linear_edges
    fixtures["disconnected_edges"] = disconnected_edges
    return fixtures


def test_hierarchical_plot(tmpdir, fixtures):

    nodes = fixtures["nodes"]
    edges = fixtures["hierarchical_edges"]

    arch = InformationArchitecture(edges=edges)

    arch.plot(dir_path=tmpdir.join('test.pdf'), produce_plot=False)

    assert nodes[0].position[1] == 0
    assert nodes[1].position[1] == -1
    assert nodes[2].position[1] == -1
    assert nodes[3].position[1] == -2
    assert nodes[4].position[1] == -2
    assert nodes[5].position[1] == -2
    assert nodes[6].position[1] == -3


def test_density(fixtures):
    edges = fixtures["Edges2"]
    arch = InformationArchitecture(edges=edges)

    assert arch.density == 2/3


def test_is_hierarchical(fixtures):

    simple_edges = fixtures["simple_edges"]
    hierarchical_edges = fixtures["hierarchical_edges"]
    centralised_edges = fixtures["centralised_edges"]
    linear_edges = fixtures["linear_edges"]
    decentralised_edges = fixtures["decentralised_edges"]
    disconnected_edges = fixtures["disconnected_edges"]

    # Simple architecture should be hierarchical
    simple_architecture = InformationArchitecture(edges=simple_edges)
    assert simple_architecture.is_hierarchical

    # Hierarchical architecture should be hierarchical
    hierarchical_architecture = InformationArchitecture(edges=hierarchical_edges)
    assert hierarchical_architecture.is_hierarchical

    # Centralised architecture should not be hierarchical
    centralised_architecture = InformationArchitecture(edges=centralised_edges)
    assert centralised_architecture.is_hierarchical is False

    # Linear architecture should be hierarchical
    linear_architecture = InformationArchitecture(edges=linear_edges)
    assert linear_architecture.is_hierarchical

    # Decentralised architecture should not be hierarchical
    decentralised_architecture = InformationArchitecture(edges=decentralised_edges)
    assert decentralised_architecture.is_hierarchical is False

    # Disconnected architecture should not be connected
    disconnected_architecture = InformationArchitecture(edges=disconnected_edges,
                                                        force_connected=False)
    assert disconnected_architecture.is_hierarchical is False


def test_is_centralised(fixtures):

    simple_edges = fixtures["simple_edges"]
    hierarchical_edges = fixtures["hierarchical_edges"]
    centralised_edges = fixtures["centralised_edges"]
    linear_edges = fixtures["linear_edges"]
    decentralised_edges = fixtures["decentralised_edges"]
    disconnected_edges = fixtures["disconnected_edges"]

    # Simple architecture should be centralised
    simple_architecture = InformationArchitecture(edges=simple_edges)
    assert simple_architecture.is_centralised

    # Hierarchical architecture should be centralised
    hierarchical_architecture = InformationArchitecture(edges=hierarchical_edges)
    assert hierarchical_architecture.is_centralised

    # Centralised architecture should be centralised
    centralised_architecture = InformationArchitecture(edges=centralised_edges)
    assert centralised_architecture.is_centralised

    # Decentralised architecture should not be centralised
    decentralised_architecture = InformationArchitecture(edges=decentralised_edges)
    assert decentralised_architecture.is_centralised is False

    # Linear architecture should be centralised
    linear_architecture = InformationArchitecture(edges=linear_edges)
    assert linear_architecture.is_centralised

    # Disconnected architecture should not be centralised
    disconnected_architecture = InformationArchitecture(edges=disconnected_edges,
                                                        force_connected=False)
    assert disconnected_architecture.is_centralised is False


def test_is_connected(fixtures):
    simple_edges = fixtures["simple_edges"]
    hierarchical_edges = fixtures["hierarchical_edges"]
    centralised_edges = fixtures["centralised_edges"]
    linear_edges = fixtures["linear_edges"]
    decentralised_edges = fixtures["decentralised_edges"]
    disconnected_edges = fixtures["disconnected_edges"]

    # Simple architecture should be connected
    simple_architecture = InformationArchitecture(edges=simple_edges)
    assert simple_architecture.is_connected

    # Hierarchical architecture should be connected
    hierarchical_architecture = InformationArchitecture(edges=hierarchical_edges)
    assert hierarchical_architecture.is_connected

    # Centralised architecture should be connected
    centralised_architecture = InformationArchitecture(edges=centralised_edges)
    assert centralised_architecture.is_connected

    # Decentralised architecture should be connected
    decentralised_architecture = InformationArchitecture(edges=decentralised_edges)
    assert decentralised_architecture.is_connected

    # Linear architecture should be connected
    linear_architecture = InformationArchitecture(edges=linear_edges)
    assert linear_architecture.is_connected

    # Disconnected architecture should not be connected
    disconnected_architecture = InformationArchitecture(edges=disconnected_edges,
                                                        force_connected=False)
    assert disconnected_architecture.is_connected is False


def test_leaf_nodes(fixtures):
    nodes = fixtures["nodes"]
    simple_edges = fixtures["simple_edges"]
    hierarchical_edges = fixtures["hierarchical_edges"]
    centralised_edges = fixtures["centralised_edges"]
    linear_edges = fixtures["linear_edges"]
    decentralised_edges = fixtures["decentralised_edges"]
    disconnected_edges = fixtures["disconnected_edges"]

    # Simple architecture should be connected
    simple_architecture = InformationArchitecture(edges=simple_edges)
    assert simple_architecture.leaf_nodes == set([nodes[1], nodes[2]])



