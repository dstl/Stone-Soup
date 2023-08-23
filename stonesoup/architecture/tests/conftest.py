import pytest

from datetime import datetime

from ..edge import Edge
from ..node import Node


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
def times():
    time_a = datetime.strptime("23/08/2023 13:36:20", "%d/%m/%Y %H:%M:%S")
    return {'a': time_a}
