import pytest

from ..edge import Edges, Edge, DataPiece, Message, FusionQueue
from ...types.track import Track
from ...types.time import CompoundTimeRange, TimeRange

from datetime import timedelta


def test_data_piece(nodes, times):
    with pytest.raises(TypeError):
        _ = DataPiece()

    data_piece = DataPiece(node=nodes['a'], originator=nodes['a'],
                           data=Track([]), time_arrived=times['a'])
    assert data_piece.sent_to == set()
    assert data_piece.track is None


def test_edge_init(nodes, times, data_pieces):
    with pytest.raises(TypeError):
        Edge()
    with pytest.raises(TypeError):
        Edge(nodes['a'], nodes['b'])  # e.g. forgetting to put nodes inside a tuple
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
    assert message in edge.messages_held['pending'][times['a']]
    assert len(edge.messages_held['received']) == 0
    # times_b is 1 min later
    edge.update_messages(current_time=times['b'])

    assert len(edge.messages_held['received']) == 1
    assert len(edge.messages_held['pending']) == 0
    assert message in edge.messages_held['received'][times['a']]


def test_failed(edges, times):
    edge = edges['a']
    assert edge.time_range_failed == CompoundTimeRange()
    edge.failed(times['a'], timedelta(seconds=5))
    new_time_range = TimeRange(times['a'], times['a'] + timedelta(seconds=5))
    assert edge.time_range_failed == CompoundTimeRange([new_time_range])


def test_edges(edges, nodes):
    edges_list = Edges([edges['a'], edges['b']])
    assert edges_list.edges == [edges['a'], edges['b']]
    edges_list.add(edges['c'])
    assert edges['c'] in edges_list
    edges_list.add(Edge((nodes['a'], nodes['b'])))
    assert len(edges_list) == 4
    assert (nodes['a'], nodes['b']) in edges_list.edge_list
    assert (nodes['a'], nodes['b']) in edges_list.edge_list

    empty_edges = Edges()
    assert len(empty_edges) == 0
    assert empty_edges.edge_list == []
    assert empty_edges.edges == []
    assert [edge for edge in empty_edges] == []
    empty_edges.add(Edge((nodes['a'], nodes['b'])))
    assert [edge for edge in empty_edges] == [Edge((nodes['a'], nodes['b']))]


def test_message(edges, data_pieces, times):
    with pytest.raises(TypeError):
        Message()
    edge = edges['a']
    message = Message(edge=edge, time_pertaining=times['a'], time_sent=times['b'],
                      data_piece=data_pieces['a'])
    assert message.sender_node == edge.sender
    assert message.recipient_node == edge.recipient
    edge.edge_latency = 5.0
    edge.sender.latency = 1.0
    assert message.arrival_time == times['b'] + timedelta(seconds=6.0)
    assert message.status == 'sending'
    with pytest.raises(ValueError):
        message.update(times['a'])

    message.update(times['b'])
    assert message.status == 'sending'

    message.update(times['b'] + timedelta(seconds=3))
    assert message.status == 'transferring'

    message.update(times['b'] + timedelta(seconds=8))
    assert message.status == 'received'


def test_fusion_queue():
    q = FusionQueue()
    iter_q = iter(q)
    assert q._to_consume == 0
    assert not q.waiting_for_data
    assert not q._consuming
    q.put("item")
    q.put("another item")

    with pytest.raises(NotImplementedError):
        q.get("anything")

    assert q._to_consume == 2
    a = next(iter_q)
    assert a == "item"
    assert q._to_consume == 2
    b = next(iter_q)
    assert b == "another item"
    assert q._to_consume == 1
