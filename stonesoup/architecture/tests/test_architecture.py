import pytest
import datetime

from stonesoup.architecture import InformationArchitecture
from ..edge import Edge, Edges
from ..node import RepeaterNode
from stonesoup.types.detection import TrueDetection


def test_hierarchical_plot(tmpdir, nodes, edge_lists):

    edges = edge_lists["hierarchical_edges"]

    arch = InformationArchitecture(edges=edges)

    arch.plot(dir_path=tmpdir.join('test.pdf'), save_plot=False)

    # Check that nodes are plotted on the correct layer. x position of each node is can change
    # depending on the order that they are iterated though, hence not entirely predictable so no
    # assertion on this is made.
    assert nodes['s1'].position[1] == 0
    assert nodes['s2'].position[1] == -1
    assert nodes['s3'].position[1] == -1
    assert nodes['s4'].position[1] == -2
    assert nodes['s5'].position[1] == -2
    assert nodes['s6'].position[1] == -2
    assert nodes['s7'].position[1] == -3

    # Check that type(position) for each node is a tuple.
    assert type(nodes['s1'].position) == tuple
    assert type(nodes['s2'].position) == tuple
    assert type(nodes['s3'].position) == tuple
    assert type(nodes['s4'].position) == tuple
    assert type(nodes['s5'].position) == tuple
    assert type(nodes['s6'].position) == tuple
    assert type(nodes['s7'].position) == tuple

    decentralised_edges = edge_lists["decentralised_edges"]
    arch = InformationArchitecture(edges=decentralised_edges)

    with pytest.raises(ValueError):
        arch.plot(dir_path=tmpdir.join('test.pdf'), save_plot=False, plot_style='hierarchical')


def test_plot_title(nodes, tmpdir, edge_lists):
    edges = edge_lists["decentralised_edges"]

    arch = InformationArchitecture(edges=edges)

    # Check that plot function runs when plot_title is given as a str.
    arch.plot(dir_path=tmpdir.join('test.pdf'), save_plot=False, plot_title="This is the title of "
                                                                            "my plot")

    # Check that plot function runs when plot_title is True.
    arch.plot(dir_path=tmpdir.join('test.pdf'), save_plot=False, plot_title=True)

    # Check that error is raised when plot_title is not a str or a bool.
    x = RepeaterNode()
    with pytest.raises(ValueError):
        arch.plot(dir_path=tmpdir.join('test.pdf'), save_plot=False, plot_title=x)


def test_plot_positions(nodes, tmpdir):
    edges1 = Edges([Edge((nodes['p2'], nodes['p1'])), Edge((nodes['p3'], nodes['p1']))])

    arch1 = InformationArchitecture(edges=edges1)

    arch1.plot(dir_path=tmpdir.join('test.pdf'), save_plot=False, use_positions=True)

    # Assert positions are correct after plot() has run
    assert nodes['p1'].position == (0, 0)
    assert nodes['p2'].position == (-1, -1)
    assert nodes['p3'].position == (1, -1)

    # Change plot positions to non tuple values
    nodes['p3'].position = RepeaterNode()
    nodes['p2'].position = 'Not a tuple'
    nodes['p3'].position = ['Definitely', 'not', 'a', 'tuple']

    edges2 = Edges([Edge((nodes['s2'], nodes['s1'])), Edge((nodes['s3'], nodes['s1']))])
    arch2 = InformationArchitecture(edges=edges2)

    with pytest.raises(TypeError):
        arch2.plot(dir_path=tmpdir.join('test.pdf'), save_plot=False, use_positions=True)


def test_density(edge_lists):

    simple_edges = edge_lists["simple_edges"]
    k4_edges = edge_lists["k4_edges"]

    # Graph k3 (complete graph with 3 nodes) has 3 edges
    # Simple architecture has 3 nodes and 2 edges: density should be 2/3
    simple_architecture = InformationArchitecture(edges=simple_edges)
    assert simple_architecture.density == 2/3

    # Graph k4 has 6 edges and density 1
    k4_architecture = InformationArchitecture(edges=k4_edges)
    assert k4_architecture.density == 1


def test_is_hierarchical(edge_lists):

    simple_edges = edge_lists["simple_edges"]
    hierarchical_edges = edge_lists["hierarchical_edges"]
    centralised_edges = edge_lists["centralised_edges"]
    linear_edges = edge_lists["linear_edges"]
    decentralised_edges = edge_lists["decentralised_edges"]
    disconnected_edges = edge_lists["disconnected_edges"]

    # Simple architecture should be hierarchical
    simple_architecture = InformationArchitecture(edges=simple_edges)
    assert simple_architecture.is_hierarchical

    # Hierarchical architecture should be hierarchical
    hierarchical_architecture = InformationArchitecture(edges=hierarchical_edges)
    assert hierarchical_architecture.is_hierarchical

    # Centralised architecture should not be hierarchical
    centralised_architecture = InformationArchitecture(edges=centralised_edges)
    assert centralised_architecture.is_hierarchical is False

    # Linear architecture should be hierarchical
    linear_architecture = InformationArchitecture(edges=linear_edges)
    assert linear_architecture.is_hierarchical

    # Decentralised architecture should not be hierarchical
    decentralised_architecture = InformationArchitecture(edges=decentralised_edges)
    assert decentralised_architecture.is_hierarchical is False

    # Disconnected architecture should not be connected
    disconnected_architecture = InformationArchitecture(edges=disconnected_edges,
                                                        force_connected=False)
    assert disconnected_architecture.is_hierarchical is False


def test_is_centralised(edge_lists):

    simple_edges = edge_lists["simple_edges"]
    hierarchical_edges = edge_lists["hierarchical_edges"]
    centralised_edges = edge_lists["centralised_edges"]
    linear_edges = edge_lists["linear_edges"]
    decentralised_edges = edge_lists["decentralised_edges"]
    disconnected_edges = edge_lists["disconnected_edges"]
    disconnected_loop_edges = edge_lists["disconnected_loop_edges"]

    # Simple architecture should be centralised
    simple_architecture = InformationArchitecture(edges=simple_edges)
    assert simple_architecture.is_centralised

    # Hierarchical architecture should be centralised
    hierarchical_architecture = InformationArchitecture(edges=hierarchical_edges)
    assert hierarchical_architecture.is_centralised

    # Centralised architecture should be centralised
    centralised_architecture = InformationArchitecture(edges=centralised_edges)
    assert centralised_architecture.is_centralised

    # Decentralised architecture should not be centralised
    decentralised_architecture = InformationArchitecture(edges=decentralised_edges)
    assert decentralised_architecture.is_centralised is False

    # Linear architecture should be centralised
    linear_architecture = InformationArchitecture(edges=linear_edges)
    assert linear_architecture.is_centralised

    # Disconnected architecture should not be centralised
    disconnected_architecture = InformationArchitecture(edges=disconnected_edges,
                                                        force_connected=False)
    assert disconnected_architecture.is_centralised is False

    disconnected_loop_architecture = InformationArchitecture(edges=disconnected_loop_edges,
                                                             force_connected=False)
    assert disconnected_loop_architecture.is_centralised is False


def test_is_connected(edge_lists):
    simple_edges = edge_lists["simple_edges"]
    hierarchical_edges = edge_lists["hierarchical_edges"]
    centralised_edges = edge_lists["centralised_edges"]
    linear_edges = edge_lists["linear_edges"]
    decentralised_edges = edge_lists["decentralised_edges"]
    disconnected_edges = edge_lists["disconnected_edges"]

    # Simple architecture should be connected
    simple_architecture = InformationArchitecture(edges=simple_edges)
    assert simple_architecture.is_connected

    # Hierarchical architecture should be connected
    hierarchical_architecture = InformationArchitecture(edges=hierarchical_edges)
    assert hierarchical_architecture.is_connected

    # Centralised architecture should be connected
    centralised_architecture = InformationArchitecture(edges=centralised_edges)
    assert centralised_architecture.is_connected

    # Decentralised architecture should be connected
    decentralised_architecture = InformationArchitecture(edges=decentralised_edges)
    assert decentralised_architecture.is_connected

    # Linear architecture should be connected
    linear_architecture = InformationArchitecture(edges=linear_edges)
    assert linear_architecture.is_connected

    # Disconnected architecture should not be connected
    disconnected_architecture = InformationArchitecture(edges=disconnected_edges,
                                                        force_connected=False)
    assert disconnected_architecture.is_connected is False

    # Raise error with force_connected=True on a disconnected graph
    with pytest.raises(ValueError):
        _ = InformationArchitecture(edges=disconnected_edges)


def test_recipients(nodes, edge_lists):
    centralised_edges = edge_lists["centralised_edges"]

    centralised_architecture = InformationArchitecture(edges=centralised_edges)
    assert centralised_architecture.recipients(nodes['s1']) == set()
    assert centralised_architecture.recipients(nodes['s2']) == {nodes['s1']}
    assert centralised_architecture.recipients(nodes['s3']) == {nodes['s1']}
    assert centralised_architecture.recipients(nodes['s4']) == {nodes['s2']}
    assert centralised_architecture.recipients(nodes['s5']) == {nodes['s2'], nodes['s3']}
    assert centralised_architecture.recipients(nodes['s6']) == {nodes['s3']}
    assert centralised_architecture.recipients(nodes['s7']) == {nodes['s6'], nodes['s5']}

    with pytest.raises(ValueError):
        centralised_architecture.recipients(nodes['s8'])


def test_senders(nodes, edge_lists):
    centralised_edges = edge_lists["centralised_edges"]

    centralised_architecture = InformationArchitecture(edges=centralised_edges)
    assert centralised_architecture.senders(nodes['s1']) == {nodes['s2'], nodes['s3']}
    assert centralised_architecture.senders(nodes['s2']) == {nodes['s4'], nodes['s5']}
    assert centralised_architecture.senders(nodes['s3']) == {nodes['s5'], nodes['s6']}
    assert centralised_architecture.senders(nodes['s4']) == set()
    assert centralised_architecture.senders(nodes['s5']) == {nodes['s7']}
    assert centralised_architecture.senders(nodes['s6']) == {nodes['s7']}
    assert centralised_architecture.senders(nodes['s7']) == set()

    with pytest.raises(ValueError):
        centralised_architecture.senders(nodes['s8'])


def test_shortest_path_dict(nodes, edge_lists):

    hierarchical_edges = edge_lists["hierarchical_edges"]
    disconnected_edges = edge_lists["disconnected_edges"]

    h_arch = InformationArchitecture(edges=hierarchical_edges)

    assert h_arch.shortest_path_dict[nodes['s7']][nodes['s6']] == 1
    assert h_arch.shortest_path_dict[nodes['s7']][nodes['s3']] == 2
    assert h_arch.shortest_path_dict[nodes['s7']][nodes['s1']] == 3
    assert h_arch.shortest_path_dict[nodes['s7']][nodes['s7']] == 0
    assert h_arch.shortest_path_dict[nodes['s5']][nodes['s2']] == 1
    assert h_arch.shortest_path_dict[nodes['s5']][nodes['s1']] == 2

    with pytest.raises(KeyError):
        _ = h_arch.shortest_path_dict[nodes['s2']][nodes['s3']]

    with pytest.raises(KeyError):
        _ = h_arch.shortest_path_dict[nodes['s3']][nodes['s6']]

    disconnected_arch = InformationArchitecture(edges=disconnected_edges, force_connected=False)

    assert disconnected_arch.shortest_path_dict[nodes['s2']][nodes['s1']] == 1
    assert disconnected_arch.shortest_path_dict[nodes['s4']][nodes['s3']] == 1

    with pytest.raises(KeyError):
        _ = disconnected_arch.shortest_path_dict[nodes['s1']][nodes['s4']]
        _ = disconnected_arch.shortest_path_dict[nodes['s3']][nodes['s4']]


def test_recipient_position(nodes, tmpdir, edge_lists):

    centralised_edges = edge_lists["centralised_edges"]

    centralised_arch = InformationArchitecture(edges=centralised_edges)
    centralised_arch.plot(dir_path=tmpdir.join('test.pdf'), save_plot=False,
                          plot_style='hierarchical')

    # Check types of _recipient_position output is correct
    assert type(centralised_arch._recipient_position(nodes['s4'])) == tuple
    assert type(centralised_arch._recipient_position(nodes['s4'])[0]) == float
    assert type(centralised_arch._recipient_position(nodes['s4'])[1]) == int

    # Check that finding the position of the recipient node of a node with no recipient raises an
    # error
    with pytest.raises(ValueError):
        centralised_arch._recipient_position(nodes['s1'])

    # Check that calling _recipient_position on a node with multiple recipients raises an error
    with pytest.raises(ValueError):
        centralised_arch._recipient_position(nodes['s7'])


def test_top_level_nodes(nodes, edge_lists):
    simple_edges = edge_lists["simple_edges"]
    hierarchical_edges = edge_lists["hierarchical_edges"]
    centralised_edges = edge_lists["centralised_edges"]
    linear_edges = edge_lists["linear_edges"]
    decentralised_edges = edge_lists["decentralised_edges"]
    disconnected_edges = edge_lists["disconnected_edges"]
    circular_edges = edge_lists["circular_edges"]
    disconnected_loop_edges = edge_lists["disconnected_loop_edges"]

    # Simple architecture 1 top node
    simple_architecture = InformationArchitecture(edges=simple_edges)
    assert simple_architecture.top_level_nodes == {nodes['s1']}

    # Hierarchical architecture should have 1 top node
    hierarchical_architecture = InformationArchitecture(edges=hierarchical_edges)
    assert hierarchical_architecture.top_level_nodes == {nodes['s1']}

    # Centralised architecture should have 1 top node
    centralised_architecture = InformationArchitecture(edges=centralised_edges)
    assert centralised_architecture.top_level_nodes == {nodes['s1']}

    # Decentralised architecture should have 2 top nodes
    decentralised_architecture = InformationArchitecture(edges=decentralised_edges)
    assert decentralised_architecture.top_level_nodes == {nodes['s1'], nodes['s4']}

    # Linear architecture should have 1 top node
    linear_architecture = InformationArchitecture(edges=linear_edges)
    assert linear_architecture.top_level_nodes == {nodes['s5']}

    # Disconnected architecture should have 2 top nodes
    disconnected_architecture = InformationArchitecture(edges=disconnected_edges,
                                                        force_connected=False)
    assert disconnected_architecture.top_level_nodes == {nodes['s1'], nodes['s3']}

    # Circular architecture should have no top node
    circular_architecture = InformationArchitecture(edges=circular_edges)
    assert circular_architecture.top_level_nodes == set()

    disconnected_loop_architecture = InformationArchitecture(edges=disconnected_loop_edges,
                                                             force_connected=False)
    assert disconnected_loop_architecture.top_level_nodes == {nodes['s1']}


def test_number_of_leaves(nodes, edge_lists):

    hierarchical_edges = edge_lists["hierarchical_edges"]
    circular_edges = edge_lists["circular_edges"]

    hierarchical_architecture = InformationArchitecture(edges=hierarchical_edges)

    # Check number of leaves for top node and senders of top node
    assert hierarchical_architecture.number_of_leaves(nodes['s1']) == 3
    assert hierarchical_architecture.number_of_leaves(nodes['s2']) == 2
    assert hierarchical_architecture.number_of_leaves(nodes['s3']) == 1

    # Check number of leafs of a leaf node is 1 despite having no senders
    assert hierarchical_architecture.number_of_leaves(nodes['s7']) == 1

    circular_architecture = InformationArchitecture(edges=circular_edges)

    # Check any node in a circular architecture has no leaves
    assert circular_architecture.number_of_leaves(nodes['s1']) == 0
    assert circular_architecture.number_of_leaves(nodes['s2']) == 0
    assert circular_architecture.number_of_leaves(nodes['s3']) == 0
    assert circular_architecture.number_of_leaves(nodes['s4']) == 0
    assert circular_architecture.number_of_leaves(nodes['s5']) == 0


def test_leaf_nodes(nodes, edge_lists):
    simple_edges = edge_lists["simple_edges"]
    hierarchical_edges = edge_lists["hierarchical_edges"]
    centralised_edges = edge_lists["centralised_edges"]
    linear_edges = edge_lists["linear_edges"]
    decentralised_edges = edge_lists["decentralised_edges"]
    disconnected_edges = edge_lists["disconnected_edges"]
    circular_edges = edge_lists["circular_edges"]

    # Simple architecture should have 2 leaf nodes
    simple_architecture = InformationArchitecture(edges=simple_edges)
    assert simple_architecture.leaf_nodes == {nodes['s2'], nodes['s3']}

    # Hierarchical architecture should have 3 leaf nodes
    hierarchical_architecture = InformationArchitecture(edges=hierarchical_edges)
    assert hierarchical_architecture.leaf_nodes == {nodes['s4'], nodes['s5'], nodes['s7']}

    # Centralised architecture should have 2 leaf nodes
    centralised_architecture = InformationArchitecture(edges=centralised_edges)
    assert centralised_architecture.leaf_nodes == {nodes['s4'], nodes['s7']}

    # Decentralised architecture should have 2 leaf nodes
    decentralised_architecture = InformationArchitecture(edges=decentralised_edges)
    assert decentralised_architecture.leaf_nodes == {nodes['s3'], nodes['s2']}

    # Linear architecture should have 1 leaf node
    linear_architecture = InformationArchitecture(edges=linear_edges)
    assert linear_architecture.leaf_nodes == {nodes['s1']}

    # Disconnected architecture should have 2 leaf nodes
    disconnected_architecture = InformationArchitecture(edges=disconnected_edges,
                                                        force_connected=False)
    assert disconnected_architecture.leaf_nodes == {nodes['s2'], nodes['s4']}

    # Circular architecture should have no leaf nodes
    circular_architecture = InformationArchitecture(edges=circular_edges)
    assert circular_architecture.top_level_nodes == set()


def test_all_nodes(nodes, edge_lists):
    simple_edges = edge_lists["simple_edges"]
    hierarchical_edges = edge_lists["hierarchical_edges"]
    centralised_edges = edge_lists["centralised_edges"]
    linear_edges = edge_lists["linear_edges"]
    decentralised_edges = edge_lists["decentralised_edges"]
    disconnected_edges = edge_lists["disconnected_edges"]
    circular_edges = edge_lists["circular_edges"]

    # Simple architecture should have 3 nodes
    simple_architecture = InformationArchitecture(edges=simple_edges)
    assert simple_architecture.all_nodes == {nodes['s1'], nodes['s2'], nodes['s3']}

    # Hierarchical architecture should have 7 nodes
    hierarchical_architecture = InformationArchitecture(edges=hierarchical_edges)
    assert hierarchical_architecture.all_nodes == {nodes['s1'], nodes['s2'], nodes['s3'],
                                                   nodes['s4'], nodes['s5'], nodes['s6'],
                                                   nodes['s7']}

    # Centralised architecture should have 7 nodes
    centralised_architecture = InformationArchitecture(edges=centralised_edges)
    assert centralised_architecture.all_nodes == {nodes['s1'], nodes['s2'], nodes['s3'],
                                                  nodes['s4'], nodes['s5'], nodes['s6'],
                                                  nodes['s7']}

    # Decentralised architecture should have 5 nodes
    decentralised_architecture = InformationArchitecture(edges=decentralised_edges)
    assert decentralised_architecture.all_nodes == {nodes['s1'], nodes['s2'], nodes['s3'],
                                                    nodes['s4'], nodes['s5']}

    # Linear architecture should have 5 nodes
    linear_architecture = InformationArchitecture(edges=linear_edges)
    assert linear_architecture.all_nodes == {nodes['s1'], nodes['s2'], nodes['s3'], nodes['s4'],
                                             nodes['s5']}

    # Disconnected architecture should have 4 nodes
    disconnected_architecture = InformationArchitecture(edges=disconnected_edges,
                                                        force_connected=False)
    assert disconnected_architecture.all_nodes == {nodes['s1'], nodes['s2'], nodes['s3'],
                                                   nodes['s4']}

    # Circular architecture should have 4 nodes
    circular_architecture = InformationArchitecture(edges=circular_edges)
    assert circular_architecture.all_nodes == {nodes['s1'], nodes['s2'], nodes['s3'], nodes['s4'],
                                               nodes['s5']}


def test_sensor_nodes(edge_lists, ground_truths, radar_nodes):
    radar_edges = edge_lists["radar_edges"]
    hierarchical_edges = edge_lists["hierarchical_edges"]

    network = InformationArchitecture(edges=radar_edges)

    assert network.sensor_nodes == {radar_nodes['a'], radar_nodes['b'], radar_nodes['d'],
                                    radar_nodes['e'], radar_nodes['h']}

    h_arch = InformationArchitecture(edges=hierarchical_edges)

    assert h_arch.sensor_nodes == h_arch.all_nodes
    assert len(h_arch.sensor_nodes) == 7


def test_fusion_nodes(edge_lists, ground_truths, radar_nodes):
    radar_edges = edge_lists["radar_edges"]
    hierarchical_edges = edge_lists["hierarchical_edges"]

    network = InformationArchitecture(edges=radar_edges)

    assert network.fusion_nodes == {radar_nodes['c'], radar_nodes['f'], radar_nodes['g']}

    h_arch = InformationArchitecture(edges=hierarchical_edges)

    assert h_arch.fusion_nodes == set()


def test_len(edge_lists):
    simple_edges = edge_lists["simple_edges"]
    hierarchical_edges = edge_lists["hierarchical_edges"]
    centralised_edges = edge_lists["centralised_edges"]
    linear_edges = edge_lists["linear_edges"]
    decentralised_edges = edge_lists["decentralised_edges"]
    disconnected_edges = edge_lists["disconnected_edges"]

    # Simple architecture should be connected
    simple_architecture = InformationArchitecture(edges=simple_edges)
    assert len(simple_architecture) == len(simple_architecture.all_nodes)

    # Hierarchical architecture should be connected
    hierarchical_architecture = InformationArchitecture(edges=hierarchical_edges)
    assert len(hierarchical_architecture) == len(hierarchical_architecture.all_nodes)

    # Centralised architecture should be connected
    centralised_architecture = InformationArchitecture(edges=centralised_edges)
    assert len(centralised_architecture) == len(centralised_architecture.all_nodes)

    # Decentralised architecture should be connected
    decentralised_architecture = InformationArchitecture(edges=decentralised_edges)
    assert len(decentralised_architecture) == len(decentralised_architecture.all_nodes)

    # Linear architecture should be connected
    linear_architecture = InformationArchitecture(edges=linear_edges)
    assert len(linear_architecture) == len(linear_architecture.all_nodes)

    # Disconnected architecture should not be connected
    disconnected_architecture = InformationArchitecture(edges=disconnected_edges,
                                                        force_connected=False)
    assert len(disconnected_architecture) == len(disconnected_architecture.all_nodes)


def test_information_arch_measure(edge_lists, ground_truths, times):
    edges = edge_lists["radar_edges"]
    start_time = times['start']

    network = InformationArchitecture(edges=edges)
    all_detections = network.measure(ground_truths=ground_truths, current_time=start_time)

    # Check all_detections is a dictionary
    assert type(all_detections) == dict

    # Check that number all_detections contains data for all sensor nodes
    assert all_detections.keys() == network.sensor_nodes

    # Check that correct number of detections recorded for each sensor node is equal to the number
    # of targets
    for sensornode in network.sensor_nodes:
        # Check that a detection is made for all 3 targets
        assert(len(all_detections[sensornode])) == 3
        assert type(all_detections[sensornode]) == set
        for detection in all_detections[sensornode]:
            assert type(detection) == TrueDetection

    for node in network.sensor_nodes:
        # Check that each sensor node has data held for the detection of all 3 targets
        assert len(node.data_held['created'][datetime.datetime(1306, 12, 25, 23, 47, 59)]) == 3


def test_information_arch_measure_no_noise(edge_lists, ground_truths, times):
    edges = edge_lists["radar_edges"]
    start_time = times['start']
    network = InformationArchitecture(edges=edges)
    all_detections = network.measure(ground_truths=ground_truths, current_time=start_time,
                                     noise=False)

    assert type(all_detections) == dict
    assert all_detections.keys() == network.sensor_nodes
    for sensornode in network.sensor_nodes:
        assert(len(all_detections[sensornode])) == 3
        assert type(all_detections[sensornode]) == set
        for detection in all_detections[sensornode]:
            assert type(detection) == TrueDetection


def test_information_arch_measure_no_detections(edge_lists, ground_truths, times):
    edges = edge_lists["radar_edges"]
    start_time = times['start']
    network = InformationArchitecture(edges=edges, current_time=None)
    all_detections = network.measure(ground_truths=[], current_time=start_time)

    assert type(all_detections) == dict
    assert all_detections.keys() == network.sensor_nodes

    # There should exist a key for each sensor node containing an empty list
    for sensornode in network.sensor_nodes:
        assert(len(all_detections[sensornode])) == 0
        assert type(all_detections[sensornode]) == set


def test_information_arch_measure_no_time(edge_lists, ground_truths):
    edges = edge_lists["radar_edges"]
    network = InformationArchitecture(edges=edges)
    all_detections = network.measure(ground_truths=ground_truths)

    assert type(all_detections) == dict
    assert all_detections.keys() == network.sensor_nodes
    for sensornode in network.sensor_nodes:
        assert(len(all_detections[sensornode])) == 3
        assert type(all_detections[sensornode]) == set
        for detection in all_detections[sensornode]:
            assert type(detection) == TrueDetection


def test_fully_propagated(edge_lists, times, ground_truths):
    edges = edge_lists["radar_edges"]
    start_time = times['start']

    network = InformationArchitecture(edges=edges, current_time=start_time)
    network.measure(ground_truths=ground_truths, noise=True)

    for node in network.sensor_nodes:
        # Check that each sensor node has data held for the detection of all 3 targets
        for key in node.data_held['created'].keys():
            assert len(node.data_held['created'][key]) == 3

    # Network should not be fully propagated
    assert network.fully_propagated is False

    network.propagate(time_increment=1)

    # Network should now be fully propagated
    assert network.fully_propagated


def test_information_arch_propagate(edge_lists, ground_truths, times):
    edges = edge_lists["radar_edges"]
    start_time = times['start']
    network = InformationArchitecture(edges=edges, current_time=start_time)

    network.measure(ground_truths=ground_truths, noise=True)
    network.propagate(time_increment=1)

    assert network.fully_propagated


def test_architecture_init(edge_lists, times):
    time = times['start']
    edges = edge_lists["decentralised_edges"]
    arch = InformationArchitecture(edges=edges, name='Name of Architecture', current_time=time)

    assert arch.name == 'Name of Architecture'
    assert arch.current_time == time


def test_information_arch_init(edge_lists):
    edges = edge_lists["repeater_edges"]

    # Network contains a repeater node, InformationArchitecture should raise a type error.
    with pytest.raises(TypeError):
        _ = InformationArchitecture(edges=edges)
