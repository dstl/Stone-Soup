import pytest

import numpy as np


from ..architecture import InformationArchitecture
from ..edge import Edge, Edges
from ..node import RepeaterNode, SensorNode
from ...sensor.categorical import HMMSensor
from ...models.measurement.categorical import MarkovianMeasurementModel

# @pytest.fixture
# def dependencies():
#     E = np.array([[0.8, 0.1],  # P(small | bike), P(small | car)
#                   [0.19, 0.3],  # P(medium | bike), P(medium | car)
#                   [0.01, 0.6]])  # P(large | bike), P(large | car)
#
#     model = MarkovianMeasurementModel(emission_matrix=E,
#                                       measurement_categories=['small', 'medium', 'large'])
#
#     hmm_sensor = HMMSensor(measurement_model=model)
#
#
#     node1 = SensorNode(sensor=hmm_sensor, label='1')
#     node2 = SensorNode(sensor=hmm_sensor, label='2')
#     node3 = SensorNode(sensor=hmm_sensor, label='3')
#     node4 = SensorNode(sensor=hmm_sensor, label='4')
#     node5 = SensorNode(sensor=hmm_sensor, label='5')
#     node6 = SensorNode(sensor=hmm_sensor, label='6')
#     node7 = SensorNode(sensor=hmm_sensor, label='7')
#
#     nodes = [node1, node2, node3, node4, node5, node6, node7]
#
#     edges = Edges(
#             [Edge((node2, node1)), Edge((node3, node1)), Edge((node4, node2)), Edge((node5, node2)),
#              Edge((node6, node3)), Edge((node7, node6))])
#
#     fixtures = dict()
#     fixtures["Edges"] = edges
#     fixtures["Nodes"] = nodes
#
#     return fixtures


def test_hierarchical_plot():
    #fixtures = fixtures()

    #
    # edges = fixtures["Edges"]
    # nodes = fixtures["Nodes"]

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

    edges = Edges(
            [Edge((node2, node1)), Edge((node3, node1)), Edge((node4, node2)), Edge((node5, node2)),
             Edge((node6, node3)), Edge((node7, node6))])

    arch = InformationArchitecture(edges=edges)

    arch.plot()

    assert nodes[1].position == (0, 0)
    assert nodes[2].position == (-0.5, -1)
    assert nodes[3].position == (1.0, -1)
    assert nodes[4].position == (0.0, -2)
    assert nodes[5].position == (-1.0, -2)
    assert nodes[6].position == (1.0, -2)
    assert nodes[7].position == (1.0, -3)


def test_simple_information_architecture():
    arch = InformationArchitecture()
