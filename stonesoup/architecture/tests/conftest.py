import pytest

from datetime import datetime

from ..edge import Edge, DataPiece
from ..node import Node
from ...types.track import Track


@pytest.fixture
def edges():
    edge_a = Edge(Node(label="edge_a sender"), Node("edge_a_recipient"))
    return {'a': edge_a}


@pytest.fixture
def nodes():
    node_a = Node(label="node a")
    node_b = Node(label="node b")
    return {"a": node_a, "b": node_b}


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
