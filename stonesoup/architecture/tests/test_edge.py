import datetime

import pytest

from .. import RepeaterNode
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

    assert (edge == 10) is False


def test_send_update_message(edges, times, data_pieces):
    edge = edges['a']
    assert len(edge.messages_held['pending']) == 0

    message = Message(edge, times['a'], times['a'], data_pieces['a'])
    edge.send_message(data_pieces['a'], times['a'], times['a'])

    with pytest.raises(TypeError):
        edge.send_message('not_a_data_piece', times['a'], times['a'])

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


def test_message_destinations(times, radar_nodes):
    start_time = times['start']
    node1 = RepeaterNode(label='n1')
    node2 = radar_nodes['a']
    node2.label = 'n2'
    node3 = RepeaterNode(label='n3')
    edge1 = Edge((node1, node2))
    edge2 = Edge((node1, node3))

    # Create a message without defining a destination
    message1 = Message(edge1, datetime.datetime(2016, 1, 2, 3, 4, 5), start_time,
                       DataPiece(node1, node1, Track([]),
                                 datetime.datetime(2016, 1, 2, 3, 4, 5)))

    # Create a message with node 2 as a destination
    message2 = Message(edge1, datetime.datetime(2016, 1, 2, 3, 4, 5), start_time,
                       DataPiece(node1, node1, Track([]),
                                 datetime.datetime(2016, 1, 2, 3, 4, 5)),
                       destinations={node2})

    # Create a message with as a defined destination that isn't node 2
    message3 = Message(edge1, datetime.datetime(2016, 1, 2, 3, 4, 5), start_time,
                       DataPiece(node1, node1, Track([]),
                                 datetime.datetime(2016, 1, 2, 3, 4, 5)),
                       destinations={node3})

    # Create message that has node2 and node3 as a destination
    message4 = Message(edge1, datetime.datetime(2016, 1, 2, 3, 4, 5), start_time,
                       DataPiece(node1, node1, Track([]),
                                 datetime.datetime(2016, 1, 2, 3, 4, 5)),
                       destinations={node2, node3})

    # Add messages to node1.messages_to_pass_on and check that unpassed_data() catches it
    node1.messages_to_pass_on = [message1, message2, message3, message4]
    assert edge1.unpassed_data == [message1, message2, message3, message4]
    assert edge2.unpassed_data == [message1, message2, message3, message4]

    # Pass data to edges
    for edge in [edge1, edge2]:
        for message in edge.unpassed_data:
            edge.pass_message(message)

    # Check that no 'unsent' data remains
    assert edge1.unsent_data == []
    assert edge2.unsent_data == []

    # Check that all messages are sent to both edges
    assert len(edge1.messages_held['pending'][start_time]) == 4
    assert len(edge2.messages_held['pending'][start_time]) == 4

    # Check node2 and node3 have no messages to pass on
    assert node2.messages_to_pass_on == []
    assert node3.messages_to_pass_on == []

    # Update both edges
    edge1.update_messages(start_time+datetime.timedelta(minutes=1), to_network_node=False)
    edge2.update_messages(start_time + datetime.timedelta(minutes=1), to_network_node=True)

    # Check node2.messages_to_pass_on contains message3 that does not have node 2 as a destination
    assert len(node2.messages_to_pass_on) == 2
    # Check node3.messages_to_pass_on contains all messages as it is not in information arch
    assert len(node3.messages_to_pass_on) == 4

    # Check that node2 has opened message1 and message3 that were intended to be processed by node3
    data_held = []
    for time in node2.data_held['unfused'].keys():
        data_held += node2.data_held['unfused'][time]
    assert len(data_held) == 3


def test_unpassed_data(times):
    start_time = times['start']
    node1 = RepeaterNode()
    node2 = RepeaterNode()
    edge = Edge((node1, node2))

    # Create a message without defining a destination (send to all)
    message = Message(edge, datetime.datetime(2016, 1, 2, 3, 4, 5), start_time,
                      DataPiece(node1, node1, 'test_data',
                                datetime.datetime(2016, 1, 2, 3, 4, 5)))

    # Add message to node.messages_to_pass_on and check that unpassed_data catches it
    node1.messages_to_pass_on.append(message)
    assert edge.unpassed_data == [message]

    # Pass message on and check that unpassed_data no longer flags it as unsent
    edge.pass_message(message)
    assert edge.unpassed_data == []


def test_add():
    node1 = RepeaterNode()
    node2 = RepeaterNode()
    node3 = RepeaterNode()

    edge1 = Edge((node1, node2))
    edge2 = Edge((node1, node2))
    edge3 = Edge((node2, node3))
    edge4 = Edge((node1, node3))

    edges = Edges([edge1, edge2, edge3])

    # Check edges.edges returns all edges
    assert edges.edges == [edge1, edge2, edge3]

    # Add an edge and check the change is reflected in edges.edges
    edges.add(edge4)
    assert edges.edges == [edge1, edge2, edge3, edge4]


def test_remove():
    node1 = RepeaterNode()
    node2 = RepeaterNode()
    node3 = RepeaterNode()

    edge1 = Edge((node1, node2))
    edge2 = Edge((node1, node2))
    edge3 = Edge((node2, node3))

    edges = Edges([edge1, edge2, edge3])

    # Check edges.edges returns all edges
    assert edges.edges == [edge1, edge2, edge3]

    # Remove an edge and check the change is reflected in edges.edges
    edges.remove(edge1)
    assert edges.edges == [edge2, edge3]


def test_get():
    node1 = RepeaterNode()
    node2 = RepeaterNode()
    node3 = RepeaterNode()

    edge1 = Edge((node1, node2))
    edge2 = Edge((node1, node2))
    edge3 = Edge((node2, node3))

    edges = Edges([edge1, edge2, edge3])

    assert edges.get((node1, node2)) == [edge1, edge2]
    assert edges.get((node2, node3)) == [edge3]
    assert edges.get((node3, node2)) == []
    assert edges.get((node1, node3)) == []

    with pytest.raises(ValueError):
        edges.get(node_pair=(node1, node2, node3))


def test_pass_message(times):
    start_time = times['start']
    node1 = RepeaterNode()
    node2 = RepeaterNode()
    edge = Edge((node1, node2))
    message = Message(edge, datetime.datetime(2016, 1, 2, 3, 4, 5), start_time,
                      DataPiece(node1, node1, 'test_data', datetime.datetime(2016, 1, 2, 3, 4, 5)))

    node1.messages_to_pass_on.append(message)

    assert node1.messages_to_pass_on == [message]

    edge.pass_message(message)
    assert node1.messages_to_pass_on == [message]
    assert node2.messages_to_pass_on == []
    assert message in edge.messages_held['pending'][start_time]
    assert edge.unpassed_data == []

    assert (message == 10) is False
