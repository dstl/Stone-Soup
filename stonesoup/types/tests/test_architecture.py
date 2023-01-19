from ..architecture import Architecture, NetworkArchitecture, InformationArchitecture, \
    CombinedArchitecture, ProcessingNode, RepeaterNode

import pytest


def test_architecture():
    a, b, c, d, e = RepeaterNode("a"), RepeaterNode("b"), RepeaterNode("c"), RepeaterNode("d"), \
                    RepeaterNode("e")

    edge_list_unconnected = [(a, b), (c, d)]
    with pytest.raises(ValueError):
        Architecture(edge_list=edge_list_unconnected, force_connected=True)
    edge_list_connected = [(a, b), (b, c), (b, d)]
    a_test_hier = Architecture(edge_list=edge_list_connected,
                               force_connected=False, name="bleh")
    edge_list_loop = [(a, b), (b, c), (c, a)]
    a_test_loop = Architecture(edge_list=edge_list_loop,
                               force_connected=False)

    assert a_test_loop.is_connected and a_test_hier.is_connected
    assert a_test_hier.is_hierarchical
    assert not a_test_loop.is_hierarchical

    a_test_hier.plot(dir_path='U:\\My Documents\\temp')


def test_information_architecture():
    with pytest.raises(TypeError):
        # Repeater nodes have no place in an information architecture
        InformationArchitecture(edge_list=[(RepeaterNode(), RepeaterNode())])
    ia_test = InformationArchitecture()


def test_density():
    a, b, c, d = RepeaterNode(), RepeaterNode(), RepeaterNode(), RepeaterNode()
    edge_list = [(a, b), (c, d), (d, a)]
    assert Architecture(edge_list=edge_list, node_set={a, b, c, d}).density == 1/2


