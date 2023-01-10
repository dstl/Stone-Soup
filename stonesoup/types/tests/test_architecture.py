from ..architecture import Architecture, NetworkArchitecture, InformationArchitecture, \
    CombinedArchitecture, ProcessingNode, RepeaterNode

import pytest


def test_information_architecture():
    with pytest.raises(TypeError):
        InformationArchitecture(node_set={RepeaterNode()})

def test_density():
    a, b, c, d = RepeaterNode(), RepeaterNode(), RepeaterNode(), RepeaterNode()
    edge_list = [(0, 1), (1, 2)]
    assert Architecture(node_set={a, b, c, d}, edge_list=edge_list).density == 1/3