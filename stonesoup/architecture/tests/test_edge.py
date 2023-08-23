import pytest

from ..edge import Edges, Edge, DataPiece
from ...types.track import Track


def test_data_piece(nodes, times):
    with pytest.raises(TypeError):
        data_piece_fail = DataPiece()

    data_piece = DataPiece(node=nodes['a'], originator=nodes['a'],
                           data=Track([]), time_arrived=times['a'])
    assert data_piece.sent_to == set()
    assert data_piece.track is None


def test_edge_init(nodes, times, data_pieces):
    with pytest.raises(TypeError):
        edge_fail = Edge()
    edge = Edge((nodes['a'], nodes['b']))
    assert edge.edge_latency == 0.0
    assert edge.sender == nodes['a']
    assert edge.recipient == nodes['b']
    assert edge.nodes == (nodes['a'], nodes['b'])

    assert edge.unsent_data == []
    nodes['a'].data_held['fused'][times['a']] = [data_pieces['a'], data_pieces['b']]
    assert (data_pieces['a'], times['a']) in edge.unsent_data
    assert (data_pieces['b'], times['a']) in edge.unsent_data
    assert len(edge.unsent_data) == 2

    assert edge.ovr_latency == 0.0
    nodes['a'].latency = 1.0
    nodes['b'].latency = 2.0
    assert edge.ovr_latency == 1.0

