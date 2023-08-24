import pytest

from ..node import Node, SensorNode, FusionNode, SensorFusionNode
from ...types.hypothesis import Hypothesis
from ...types.track import Track
from ...types.state import State, StateVector


def test_node(data_pieces, times, nodes):
    node = Node()
    assert node.latency == 0.0
    assert node.font_size == 5
    assert len(node.data_held) == 3
    assert node.data_held == {"fused": {}, "created": {}, "unfused": {}}

    node.update(times['a'], times['b'], data_pieces['a'], "fused")
    new_data_piece = node.data_held['fused'][times['a']].pop()
    assert new_data_piece.originator == nodes['a']
    assert isinstance(new_data_piece.data, Track) and len(new_data_piece.data) == 0
    assert new_data_piece.time_arrived == times['b']

    with pytest.raises(TypeError):
        node.update(times['b'], times['a'], data_pieces['hyp'], "created")
    node.update(times['b'], times['a'], data_pieces['hyp'], "created",
                track=Track([State(state_vector=StateVector([1]))]))
    new_data_piece2 = node.data_held['created'][times['b']].pop()
    assert new_data_piece2.originator == nodes['b']
    assert isinstance(new_data_piece2.data, Hypothesis)
    assert new_data_piece2.time_arrived == times['a']
