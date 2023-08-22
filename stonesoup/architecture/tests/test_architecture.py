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

    non_hierarchical_edges = Edges(
        [Edge((node2, node1)), Edge((node3, node1)), Edge((node4, node2)), Edge((node5, node2)),
         Edge((node6, node3)), Edge((node7, node6)), Edge((node7, node1))])

    edges2 = Edges([Edge((node2, node1)), Edge((node3, node1))])

    fixtures = dict()
    fixtures["hierarchical_edges"] = hierarchical_edges
    fixtures["non_hierarchical_edges"] = non_hierarchical_edges
    fixtures["Nodes"] = nodes
    fixtures["Edges2"] = edges2

    return fixtures


def test_hierarchical_plot(tmpdir, fixtures):

    nodes = fixtures["Nodes"]
    edges = fixtures["hierarchical_edges"]

    arch = InformationArchitecture(edges=edges)

    arch.plot(dir_path=tmpdir.join('test.pdf'), show_plot=False)

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

    h_edges = fixtures["hierarchical_edges"]
    n_h_edges = fixtures["non_hierarchical_edges"]

    h_architecture = InformationArchitecture(edges=h_edges)
    n_h_architecture = InformationArchitecture(edges=n_h_edges)

    assert h_architecture.is_hierarchical
    assert n_h_architecture.is_hierarchical is False


