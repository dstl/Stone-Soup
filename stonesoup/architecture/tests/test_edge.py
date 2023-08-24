import pytest

from ..edge import Edges, Edge, DataPiece, Message
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
    assert all(len(edge.messages_held[status]) == 0 for status in ['pending', 'received'])

    assert edge.unsent_data == []
    nodes['a'].data_held['fused'][times['a']] = [data_pieces['a'], data_pieces['b']]
    assert (data_pieces['a'], times['a']) in edge.unsent_data
    assert (data_pieces['b'], times['a']) in edge.unsent_data
    assert len(edge.unsent_data) == 2

    assert edge.ovr_latency == 0.0
    nodes['a'].latency = 1.0
    nodes['b'].latency = 2.0
    assert edge.ovr_latency == 1.0


def test_send_update_message(edges, times, data_pieces):
    edge = edges['a']
    assert len(edge.messages_held['pending']) == 0

    message = Message(edge, times['a'], times['a'], data_pieces['a'])
    edge.send_message(data_pieces['a'], times['a'], times['a'])

    assert len(edge.messages_held['pending']) == 1
    assert times['a'] in edge.messages_held['pending']
    assert len(edge.messages_held['pending'][times['a']]) == 1
    print("\n\n\n")
    print(message)
    print("\n\n\n")
    print(edge.messages_held['pending'][times['a']])
    print(message)
    assert message in edge.messages_held['pending'][times['a']]
    assert len(edge.messages_held['received']) == 0
    # times_b is 1 min later
    edge.update_messages(current_time=times['b'])

    assert len(edge.messages_held['received']) == 1
    assert len(edge.messages_held['pending']) == 0
    #assert message in edge.messages_held['received'][times['a']]


def test_failed():
    assert True