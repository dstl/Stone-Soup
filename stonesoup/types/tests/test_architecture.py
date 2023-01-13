from ..architecture import Architecture, NetworkArchitecture, InformationArchitecture, \
    CombinedArchitecture, ProcessingNode, RepeaterNode
import networkx as nx

import pytest


def test_architecture():
    a, b, c, d, e = RepeaterNode(), RepeaterNode(), RepeaterNode(), RepeaterNode(), RepeaterNode()
    edge_list_fail = [(a, b), (b, d)]
    with pytest.raises(ValueError):
        # Should fail, as the node set does not match the nodes used in the edge list
        Architecture(edge_list=edge_list_fail, node_set={a, b, c, d}, force_connected=False)
    edge_list_unconnected = [(a, b), (c, d)]
    with pytest.raises(ValueError):
        Architecture(edge_list=edge_list_unconnected, node_set={a, b, c, d}, force_connected=True)
    a_test = Architecture(edge_list=edge_list_unconnected,
                          node_set={a, b, c, d},
                          force_connected=False)
    edge_list_connected = [(a, b), (b, c)]
    a_test_hier = Architecture(edge_list=edge_list_connected,
                               force_connected=False)
    edge_list_loop = [(a, b), (b, c), (c, a)]
    a_test_loop = Architecture(edge_list=edge_list_loop,
                               force_connected=False)

    assert a_test_loop.is_connected and a_test_hier.is_connected
    assert a_test_hier.is_hierarchical
    assert not a_test_loop.is_hierarchical


def test_information_architecture():
    with pytest.raises(TypeError):
        # Repeater nodes have no place in an information architecture
        InformationArchitecture(edge_list=[(RepeaterNode(), RepeaterNode())])
    ia_test = InformationArchitecture()


def test_density():
    a, b, c, d = RepeaterNode(), RepeaterNode(), RepeaterNode(), RepeaterNode()
    edge_list = [(a, b), (c, d), (d, a)]
    assert Architecture(edge_list=edge_list, node_set={a, b, c, d}).density == 1/2


