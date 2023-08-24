import pytest

from datetime import datetime
import numpy as np

from ..edge import Edge, DataPiece, Edges
from ..node import Node, RepeaterNode, SensorNode, FusionNode, SensorFusionNode
from ...types.track import Track
from ...sensor.categorical import HMMSensor
from ...models.measurement.categorical import MarkovianMeasurementModel


@pytest.fixture
def edges():
    edge_a = Edge((Node(label="edge_a sender"), Node(label="edge_a recipient")))
    edge_b = Edge((Node(label="edge_b sender"), Node(label="edge_b recipient")))
    edge_c = Edge((Node(label="edge_c sender"), Node(label="edge_c recipient")))
    return {'a': edge_a, 'b': edge_b, 'c': edge_c}


@pytest.fixture
def nodes():
    E = np.array([[0.8, 0.1],  # P(small | bike), P(small | car)
                  [0.19, 0.3],  # P(medium | bike), P(medium | car)
                  [0.01, 0.6]])  # P(large | bike), P(large | car)

    model = MarkovianMeasurementModel(emission_matrix=E,
                                      measurement_categories=['small', 'medium', 'large'])

    hmm_sensor = HMMSensor(measurement_model=model)

    node_a = Node(label="node a")
    node_b = Node(label="node b")
    sensornode_1 = SensorNode(sensor=hmm_sensor, label='s1')
    sensornode_2 = SensorNode(sensor=hmm_sensor, label='s2')
    sensornode_3 = SensorNode(sensor=hmm_sensor, label='s3')
    sensornode_4 = SensorNode(sensor=hmm_sensor, label='s4')
    sensornode_5 = SensorNode(sensor=hmm_sensor, label='s5')
    sensornode_6 = SensorNode(sensor=hmm_sensor, label='s6')
    sensornode_7 = SensorNode(sensor=hmm_sensor, label='s7')
    sensornode_8 = SensorNode(sensor=hmm_sensor, label='s8')
    pnode_1 = SensorNode(sensor=hmm_sensor, label='p1', position=(0, 0))
    pnode_2 = SensorNode(sensor=hmm_sensor, label='p2', position=(-1, -1))
    pnode_3 = SensorNode(sensor=hmm_sensor, label='p3', position=(1, -1))

    return {"a": node_a, "b": node_b, "s1": sensornode_1, "s2": sensornode_2, "s3": sensornode_3,
            "s4": sensornode_4, "s5": sensornode_5, "s6": sensornode_6, "s7": sensornode_7,
            "s8": sensornode_8, "p1": pnode_1, "p2": pnode_2, "p3": pnode_3}


@pytest.fixture
def edge_lists(nodes):
    hierarchical_edges = Edges([Edge((nodes['s2'], nodes['s1'])), Edge((nodes['s3'], nodes['s1'])),
                                Edge((nodes['s4'], nodes['s2'])), Edge((nodes['s5'], nodes['s2'])),
                                Edge((nodes['s6'], nodes['s3'])), Edge((nodes['s7'], nodes['s6']))])

    centralised_edges = Edges(
        [Edge((nodes['s2'], nodes['s1'])), Edge((nodes['s3'], nodes['s1'])),
         Edge((nodes['s4'], nodes['s2'])), Edge((nodes['s5'], nodes['s2'])),
         Edge((nodes['s6'], nodes['s3'])), Edge((nodes['s7'], nodes['s6'])),
         Edge((nodes['s7'], nodes['s5'])), Edge((nodes['s5'], nodes['s3']))])

    simple_edges = Edges([Edge((nodes['s2'], nodes['s1'])), Edge((nodes['s3'], nodes['s1']))])

    linear_edges = Edges([Edge((nodes['s1'], nodes['s2'])), Edge((nodes['s2'], nodes['s3'])),
                          Edge((nodes['s3'], nodes['s4'])),
                          Edge((nodes['s4'], nodes['s5']))])

    decentralised_edges = Edges(
        [Edge((nodes['s2'], nodes['s1'])), Edge((nodes['s3'], nodes['s1'])),
         Edge((nodes['s3'], nodes['s4'])), Edge((nodes['s3'], nodes['s5'])),
         Edge((nodes['s5'], nodes['s4']))])

    disconnected_edges = Edges([Edge((nodes['s2'], nodes['s1'])), Edge((nodes['s4'], nodes['s3']))])

    k4_edges = Edges(
        [Edge((nodes['s1'], nodes['s2'])), Edge((nodes['s1'], nodes['s3'])),
         Edge((nodes['s1'], nodes['s4'])), Edge((nodes['s2'], nodes['s3'])),
         Edge((nodes['s2'], nodes['s4'])), Edge((nodes['s3'], nodes['s4']))])

    circular_edges = Edges(
        [Edge((nodes['s1'], nodes['s2'])), Edge((nodes['s2'], nodes['s3'])),
         Edge((nodes['s3'], nodes['s4'])), Edge((nodes['s4'], nodes['s5'])),
         Edge((nodes['s5'], nodes['s1']))])

    disconnected_loop_edges = Edges(
        [Edge((nodes['s2'], nodes['s1'])), Edge((nodes['s4'], nodes['s3'])),
         Edge((nodes['s3'], nodes['s4']))])

    return {"hierarchical_edges": hierarchical_edges, "centralised_edges": centralised_edges,
            "simple_edges": simple_edges, "linear_edges": linear_edges,
            "decentralised_edges": decentralised_edges, "disconnected_edges": disconnected_edges,
            "k4_edges": k4_edges, "circular_edges": circular_edges,
            "disconnected_loop_edges": disconnected_loop_edges}



@pytest.fixture
def data_pieces(times, nodes):
    data_piece_a = DataPiece(node=nodes['a'], originator=nodes['a'],
                             data=Track([]), time_arrived=times['a'])
    data_piece_b = DataPiece(node=nodes['a'], originator=nodes['b'],
                             data=Track([]), time_arrived=times['b'])
    return {'a': data_piece_a, 'b': data_piece_b}


@pytest.fixture
def times():
    time_a = datetime.strptime("23/08/2023 13:36:00", "%d/%m/%Y %H:%M:%S")
    time_b = datetime.strptime("23/08/2023 13:37:00", "%d/%m/%Y %H:%M:%S")
    return {'a': time_a, 'b': time_b}
