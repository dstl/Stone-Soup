import pytest

import numpy as np


from stonesoup.architecture import InformationArchitecture
from ..edge import Edge, Edges
from ..node import RepeaterNode, SensorNode
from ...sensor.categorical import HMMSensor
from ...models.measurement.categorical import MarkovianMeasurementModel


@pytest.fixture
def fixtures():
    E = np.array([[0.8, 0.1],  # P(small | bike), P(small | car)
                  [0.19, 0.3],  # P(medium | bike), P(medium | car)
                  [0.01, 0.6]])  # P(large | bike), P(large | car)

    model = MarkovianMeasurementModel(emission_matrix=E,
                                      measurement_categories=['small', 'medium', 'large'])

    hmm_sensor = HMMSensor(measurement_model=model)

    node1 = SensorNode(sensor=hmm_sensor, label='1')
    node2 = SensorNode(sensor=hmm_sensor, label='2')
    node3 = SensorNode(sensor=hmm_sensor, label='3')
    node4 = SensorNode(sensor=hmm_sensor, label='4')
    node5 = SensorNode(sensor=hmm_sensor, label='5')
    node6 = SensorNode(sensor=hmm_sensor, label='6')
    node7 = SensorNode(sensor=hmm_sensor, label='7')
    node8 = SensorNode(sensor=hmm_sensor, label='8')

    nodes = [node1, node2, node3, node4, node5, node6, node7, node8]

    nodep1 = SensorNode(sensor=hmm_sensor, label='p1', position=(0, 0))
    nodep2 = SensorNode(sensor=hmm_sensor, label='p1', position=(-1, -1))
    nodep3 = SensorNode(sensor=hmm_sensor, label='p1', position=(1, -1))

    pnodes = [nodep1, nodep2, nodep3]

    hierarchical_edges = Edges(
        [Edge((node2, node1)), Edge((node3, node1)), Edge((node4, node2)), Edge((node5, node2)),
         Edge((node6, node3)), Edge((node7, node6))])

    centralised_edges = Edges(
        [Edge((node2, node1)), Edge((node3, node1)), Edge((node4, node2)), Edge((node5, node2)),
         Edge((node6, node3)), Edge((node7, node6)), Edge((node7, node5)), Edge((node5, node3))])

    simple_edges = Edges([Edge((node2, node1)), Edge((node3, node1))])

    linear_edges = Edges([Edge((node1, node2)), Edge((node2, node3)), Edge((node3, node4)),
                          Edge((node4, node5))])

    decentralised_edges = Edges(
        [Edge((node2, node1)), Edge((node3, node1)), Edge((node3, node4)), Edge((node3, node5)),
         Edge((node5, node4))])

    disconnected_edges = Edges([Edge((node2, node1)), Edge((node4, node3))])

    k4_edges = Edges(
        [Edge((node1, node2)), Edge((node1, node3)), Edge((node1, node4)), Edge((node2, node3)),
         Edge((node2, node4)), Edge((node3, node4))])

    circular_edges = Edges(
        [Edge((node1, node2)), Edge((node2, node3)), Edge((node3, node4)), Edge((node4, node5)),
         Edge((node5, node1))])

    disconnected_loop_edges = Edges([Edge((node2, node1)), Edge((node4, node3)),
                                     Edge((node3, node4))])

    fixtures = dict()
    fixtures["hierarchical_edges"] = hierarchical_edges
    fixtures["centralised_edges"] = centralised_edges
    fixtures["decentralised_edges"] = decentralised_edges
    fixtures["nodes"] = nodes
    fixtures["pnodes"] = pnodes
    fixtures["simple_edges"] = simple_edges
    fixtures["linear_edges"] = linear_edges
    fixtures["disconnected_edges"] = disconnected_edges
    fixtures["k4_edges"] = k4_edges
    fixtures["circular_edges"] = circular_edges
    fixtures["disconnected_loop_edges"] = disconnected_loop_edges
    return fixtures


def test_hierarchical_plot(tmpdir, fixtures):

    nodes = fixtures["nodes"]
    edges = fixtures["hierarchical_edges"]

    arch = InformationArchitecture(edges=edges)

    arch.plot(dir_path=tmpdir.join('test.pdf'), save_plot=False)

    # Check that nodes are plotted on the correct layer. x position of each node is can change
    # depending on the order that they are iterated though, hence not entirely predictable so no
    # assertion on this is made.
    assert nodes[0].position[1] == 0
    assert nodes[1].position[1] == -1
    assert nodes[2].position[1] == -1
    assert nodes[3].position[1] == -2
    assert nodes[4].position[1] == -2
    assert nodes[5].position[1] == -2
    assert nodes[6].position[1] == -3

    # Check that type(position) for each node is a tuple.
    assert type(nodes[0].position) == tuple
    assert type(nodes[1].position) == tuple
    assert type(nodes[2].position) == tuple
    assert type(nodes[3].position) == tuple
    assert type(nodes[4].position) == tuple
    assert type(nodes[5].position) == tuple
    assert type(nodes[6].position) == tuple

    decentralised_edges = fixtures["decentralised_edges"]
    arch = InformationArchitecture(edges=decentralised_edges)

    with pytest.raises(ValueError):
        arch.plot(dir_path=tmpdir.join('test.pdf'), save_plot=False, plot_style='hierarchical')


def test_plot_title(fixtures, tmpdir):
    edges = fixtures["decentralised_edges"]

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


def test_plot_positions(fixtures, tmpdir):
    pnodes = fixtures["pnodes"]
    edges = Edges([Edge((pnodes[1], pnodes[0])), Edge((pnodes[2], pnodes[0]))])

    arch = InformationArchitecture(edges=edges)

    arch.plot(dir_path=tmpdir.join('test.pdf'), save_plot=False, use_positions=True)

    # Assert positions are correct after plot() has run
    assert pnodes[0].position == (0, 0)
    assert pnodes[1].position == (-1, -1)
    assert pnodes[2].position == (1, -1)

    # Change plot positions to non tuple values
    pnodes[0].position = RepeaterNode()
    pnodes[1].position = 'Not a tuple'
    pnodes[2].position = ['Definitely', 'not', 'a', 'tuple']

    edges = Edges([Edge((pnodes[1], pnodes[0])), Edge((pnodes[2], pnodes[0]))])

    with pytest.raises(TypeError):
        arch.plot(dir_path=tmpdir.join('test.pdf'), save_plot=False, use_positions=True)


def test_density(fixtures):

    simple_edges = fixtures["simple_edges"]
    k4_edges = fixtures["k4_edges"]

    # Graph k3 (complete graph with 3 nodes) has 3 edges
    # Simple architecture has 3 nodes and 2 edges: density should be 2/3
    simple_architecture = InformationArchitecture(edges=simple_edges)
    assert simple_architecture.density == 2/3

    # Graph k4 has 6 edges and density 1
    k4_architecture = InformationArchitecture(edges=k4_edges)
    assert k4_architecture.density == 1


def test_is_hierarchical(fixtures):

    simple_edges = fixtures["simple_edges"]
    hierarchical_edges = fixtures["hierarchical_edges"]
    centralised_edges = fixtures["centralised_edges"]
    linear_edges = fixtures["linear_edges"]
    decentralised_edges = fixtures["decentralised_edges"]
    disconnected_edges = fixtures["disconnected_edges"]

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


def test_is_centralised(fixtures):

    simple_edges = fixtures["simple_edges"]
    hierarchical_edges = fixtures["hierarchical_edges"]
    centralised_edges = fixtures["centralised_edges"]
    linear_edges = fixtures["linear_edges"]
    decentralised_edges = fixtures["decentralised_edges"]
    disconnected_edges = fixtures["disconnected_edges"]
    disconnected_loop_edges = fixtures["disconnected_loop_edges"]

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


def test_is_connected(fixtures):
    simple_edges = fixtures["simple_edges"]
    hierarchical_edges = fixtures["hierarchical_edges"]
    centralised_edges = fixtures["centralised_edges"]
    linear_edges = fixtures["linear_edges"]
    decentralised_edges = fixtures["decentralised_edges"]
    disconnected_edges = fixtures["disconnected_edges"]

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


def test_recipients(fixtures):
    nodes = fixtures["nodes"]
    centralised_edges = fixtures["centralised_edges"]

    centralised_architecture = InformationArchitecture(edges=centralised_edges)
    assert centralised_architecture.recipients(nodes[0]) == set()
    assert centralised_architecture.recipients(nodes[1]) == {nodes[0]}
    assert centralised_architecture.recipients(nodes[2]) == {nodes[0]}
    assert centralised_architecture.recipients(nodes[3]) == {nodes[1]}
    assert centralised_architecture.recipients(nodes[4]) == {nodes[1], nodes[2]}
    assert centralised_architecture.recipients(nodes[5]) == {nodes[2]}
    assert centralised_architecture.recipients(nodes[6]) == {nodes[5], nodes[4]}

    with pytest.raises(ValueError):
        centralised_architecture.recipients(nodes[7])


def test_senders(fixtures):
    nodes = fixtures["nodes"]
    centralised_edges = fixtures["centralised_edges"]

    centralised_architecture = InformationArchitecture(edges=centralised_edges)
    assert centralised_architecture.senders(nodes[0]) == {nodes[1], nodes[2]}
    assert centralised_architecture.senders(nodes[1]) == {nodes[3], nodes[4]}
    assert centralised_architecture.senders(nodes[2]) == {nodes[4], nodes[5]}
    assert centralised_architecture.senders(nodes[3]) == set()
    assert centralised_architecture.senders(nodes[4]) == {nodes[6]}
    assert centralised_architecture.senders(nodes[5]) == {nodes[6]}
    assert centralised_architecture.senders(nodes[6]) == set()

    with pytest.raises(ValueError):
        centralised_architecture.senders(nodes[7])


def test_shortest_path_dict(fixtures):

    hierarchical_edges = fixtures["hierarchical_edges"]
    disconnected_edges = fixtures["disconnected_edges"]
    nodes = fixtures["nodes"]

    h_arch = InformationArchitecture(edges=hierarchical_edges)

    assert h_arch.shortest_path_dict[nodes[6]][nodes[5]] == 1
    assert h_arch.shortest_path_dict[nodes[6]][nodes[2]] == 2
    assert h_arch.shortest_path_dict[nodes[6]][nodes[0]] == 3
    assert h_arch.shortest_path_dict[nodes[6]][nodes[6]] == 0
    assert h_arch.shortest_path_dict[nodes[4]][nodes[1]] == 1
    assert h_arch.shortest_path_dict[nodes[4]][nodes[0]] == 2

    with pytest.raises(KeyError):
        dist = h_arch.shortest_path_dict[nodes[1]][nodes[2]]

    with pytest.raises(KeyError):
        dist = h_arch.shortest_path_dict[nodes[2]][nodes[5]]

    disconnected_arch = InformationArchitecture(edges=disconnected_edges, force_connected=False)

    assert disconnected_arch.shortest_path_dict[nodes[1]][nodes[0]] == 1
    assert disconnected_arch.shortest_path_dict[nodes[3]][nodes[2]] == 1

    with pytest.raises(KeyError):
        _ = disconnected_arch.shortest_path_dict[nodes[0]][nodes[3]]
        _ = disconnected_arch.shortest_path_dict[nodes[2]][nodes[3]]


def test_recipient_position(fixtures, tmpdir):

    nodes = fixtures["nodes"]
    centralised_edges = fixtures["centralised_edges"]

    centralised_arch = InformationArchitecture(edges=centralised_edges)
    centralised_arch.plot(dir_path=tmpdir.join('test.pdf'), save_plot=False,
                          plot_style='hierarchical')

    # Check types of _recipient_position output is correct
    assert type(centralised_arch._recipient_position(nodes[3])) == tuple
    assert type(centralised_arch._recipient_position(nodes[3])[0]) == float
    assert type(centralised_arch._recipient_position(nodes[3])[1]) == int

    # Check that finding the position of the recipient node of a node with no recipient raises an
    # error
    with pytest.raises(ValueError):
        centralised_arch._recipient_position(nodes[0])

    # Check that calling _recipient_position on a node with multiple recipients raises an error
    with pytest.raises(ValueError):
        centralised_arch._recipient_position(nodes[6])


def test_top_level_nodes(fixtures):
    nodes = fixtures["nodes"]
    simple_edges = fixtures["simple_edges"]
    hierarchical_edges = fixtures["hierarchical_edges"]
    centralised_edges = fixtures["centralised_edges"]
    linear_edges = fixtures["linear_edges"]
    decentralised_edges = fixtures["decentralised_edges"]
    disconnected_edges = fixtures["disconnected_edges"]
    circular_edges = fixtures["circular_edges"]
    disconnected_loop_edges = fixtures["disconnected_loop_edges"]

    # Simple architecture 1 top node
    simple_architecture = InformationArchitecture(edges=simple_edges)
    assert simple_architecture.top_level_nodes == {nodes[0]}

    # Hierarchical architecture should have 1 top node
    hierarchical_architecture = InformationArchitecture(edges=hierarchical_edges)
    assert hierarchical_architecture.top_level_nodes == {nodes[0]}

    # Centralised architecture should have 1 top node
    centralised_architecture = InformationArchitecture(edges=centralised_edges)
    assert centralised_architecture.top_level_nodes == {nodes[0]}

    # Decentralised architecture should have 2 top nodes
    decentralised_architecture = InformationArchitecture(edges=decentralised_edges)
    assert decentralised_architecture.top_level_nodes == {nodes[0], nodes[3]}

    # Linear architecture should have 1 top node
    linear_architecture = InformationArchitecture(edges=linear_edges)
    assert linear_architecture.top_level_nodes == {nodes[4]}

    # Disconnected architecture should have 2 top nodes
    disconnected_architecture = InformationArchitecture(edges=disconnected_edges,
                                                        force_connected=False)
    assert disconnected_architecture.top_level_nodes == {nodes[0], nodes[2]}

    # Circular architecture should have no top node
    circular_architecture = InformationArchitecture(edges=circular_edges)
    assert circular_architecture.top_level_nodes == set()

    disconnected_loop_architecture = InformationArchitecture(edges=disconnected_loop_edges,
                                                             force_connected=False)
    assert disconnected_loop_architecture.top_level_nodes == {nodes[0]}


def test_number_of_leaves(fixtures):

    hierarchical_edges = fixtures["hierarchical_edges"]
    circular_edges = fixtures["circular_edges"]
    nodes = fixtures["nodes"]

    hierarchical_architecture = InformationArchitecture(edges=hierarchical_edges)

    # Check number of leaves for top node and senders of top node
    assert hierarchical_architecture.number_of_leaves(nodes[0]) == 3
    assert hierarchical_architecture.number_of_leaves(nodes[1]) == 2
    assert hierarchical_architecture.number_of_leaves(nodes[2]) == 1

    # Check number of leafs of a leaf node is 1 despite having no senders
    assert hierarchical_architecture.number_of_leaves(nodes[6]) == 1

    circular_architecture = InformationArchitecture(edges=circular_edges)

    # Check any node in a circular architecture has no leaves
    assert circular_architecture.number_of_leaves(nodes[0]) == 0
    assert circular_architecture.number_of_leaves(nodes[1]) == 0
    assert circular_architecture.number_of_leaves(nodes[2]) == 0
    assert circular_architecture.number_of_leaves(nodes[3]) == 0
    assert circular_architecture.number_of_leaves(nodes[4]) == 0


def test_leaf_nodes(fixtures):
    nodes = fixtures["nodes"]
    simple_edges = fixtures["simple_edges"]
    hierarchical_edges = fixtures["hierarchical_edges"]
    centralised_edges = fixtures["centralised_edges"]
    linear_edges = fixtures["linear_edges"]
    decentralised_edges = fixtures["decentralised_edges"]
    disconnected_edges = fixtures["disconnected_edges"]
    circular_edges = fixtures["circular_edges"]

    # Simple architecture should have 2 leaf nodes
    simple_architecture = InformationArchitecture(edges=simple_edges)
    assert simple_architecture.leaf_nodes == {nodes[1], nodes[2]}

    # Hierarchical architecture should have 3 leaf nodes
    hierarchical_architecture = InformationArchitecture(edges=hierarchical_edges)
    assert hierarchical_architecture.leaf_nodes == {nodes[3], nodes[4], nodes[6]}

    # Centralised architecture should have 2 leaf nodes
    centralised_architecture = InformationArchitecture(edges=centralised_edges)
    assert centralised_architecture.leaf_nodes == {nodes[3], nodes[6]}

    # Decentralised architecture should have 2 leaf nodes
    decentralised_architecture = InformationArchitecture(edges=decentralised_edges)
    assert decentralised_architecture.leaf_nodes == {nodes[2], nodes[1]}

    # Linear architecture should have 1 leaf node
    linear_architecture = InformationArchitecture(edges=linear_edges)
    assert linear_architecture.leaf_nodes == {nodes[0]}

    # Disconnected architecture should have 2 leaf nodes
    disconnected_architecture = InformationArchitecture(edges=disconnected_edges,
                                                        force_connected=False)
    assert disconnected_architecture.leaf_nodes == {nodes[1], nodes[3]}

    # Circular architecture should have no leaf nodes
    circular_architecture = InformationArchitecture(edges=circular_edges)
    assert circular_architecture.top_level_nodes == set()


def test_all_nodes(fixtures):
    nodes = fixtures["nodes"]
    simple_edges = fixtures["simple_edges"]
    hierarchical_edges = fixtures["hierarchical_edges"]
    centralised_edges = fixtures["centralised_edges"]
    linear_edges = fixtures["linear_edges"]
    decentralised_edges = fixtures["decentralised_edges"]
    disconnected_edges = fixtures["disconnected_edges"]
    circular_edges = fixtures["circular_edges"]

    # Simple architecture should have 3 nodes
    simple_architecture = InformationArchitecture(edges=simple_edges)
    assert simple_architecture.all_nodes == {nodes[0], nodes[1], nodes[2]}

    # Hierarchical architecture should have 7 nodes
    hierarchical_architecture = InformationArchitecture(edges=hierarchical_edges)
    assert hierarchical_architecture.all_nodes == {nodes[0], nodes[1], nodes[2], nodes[3],
                                                   nodes[4], nodes[5], nodes[6]}

    # Centralised architecture should have 7 nodes
    centralised_architecture = InformationArchitecture(edges=centralised_edges)
    assert centralised_architecture.all_nodes == {nodes[0], nodes[1], nodes[2], nodes[3],
                                                  nodes[4], nodes[5], nodes[6]}

    # Decentralised architecture should have 5 nodes
    decentralised_architecture = InformationArchitecture(edges=decentralised_edges)
    assert decentralised_architecture.all_nodes == {nodes[0], nodes[1], nodes[2], nodes[3],
                                                    nodes[4]}

    # Linear architecture should have 5 nodes
    linear_architecture = InformationArchitecture(edges=linear_edges)
    assert linear_architecture.all_nodes == {nodes[0], nodes[1], nodes[2], nodes[3], nodes[4]}

    # Disconnected architecture should have 4 nodes
    disconnected_architecture = InformationArchitecture(edges=disconnected_edges,
                                                        force_connected=False)
    assert disconnected_architecture.all_nodes == {nodes[0], nodes[1], nodes[2], nodes[3]}

    # Circular architecture should have 4 nodes
    circular_architecture = InformationArchitecture(edges=circular_edges)
    assert circular_architecture.all_nodes == {nodes[0], nodes[1], nodes[2], nodes[3], nodes[4]}

