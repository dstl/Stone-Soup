import numpy as np
import pytest

try:
    from stonesoup.types.graph import RoadNetwork
except ImportError:
    # Catch optional dependencies import error
    pytest.skip(
        "Skipping due to missing optional dependencies. Usage of the road network classes requires"
        "that the optional package dependencies 'geopandas' and 'networkx' are installed.",
        allow_module_level=True
    )


# The next two fixtures define a simple network with 4 nodes and 4 edges that looks like:
#        (.25)
#       3-->--4
#       |     |
#  (.5) ^     ^ (1.)     where the numbers in brackets are the edge weights and the arrows
#       |     |          indicate the direction of the edge
#       1-->--2
#        (1.)
@pytest.fixture()
def nodes():
    # Nodes are defined as a tuple of (node, {'pos': (x, y)})
    return [(1, {'pos': (0, 0)}),
            (2, {'pos': (1, 0)}),
            (3, {'pos': (0, 1)}),
            (4, {'pos': (1, 1)})]


@pytest.fixture()
def edges():
    # Edges are defined as a tuple of (node1, node2, {'weight': weight})
    return [(1, 2, {'weight': 1.}),
            (1, 3, {'weight': .5}),
            (2, 4, {'weight': 1.}),
            (3, 4, {'weight': .25})]


@pytest.mark.parametrize(
    "mode",
    [
        "individual",   # Add/remove nodes and edges individually
        "bulk",         # Add/remove nodes and edges in bulk
        "dict"          # Construct net from a dictionary
    ]
)
def test_road_network_building(nodes, edges, mode):
    """Test building a road network"""

    network = RoadNetwork()
    if mode == "individual":
        # Test we can build using individual nodes and edges
        for node in nodes:
            network.add_node(node[0], **node[1])
        for edge in edges:
            network.add_edge(edge[0], edge[1], **edge[2])
    elif mode == "bulk":
        # Test we can build using lists of nodes and edges
        network.add_nodes_from(nodes)
        network.add_edges_from(edges)
    else:
        dct = {'nodes': {node[0]: node[1] for node in nodes},
               'edges': {(edge[0], edge[1]): edge[2] for edge in edges}}
        network = RoadNetwork.from_dict(dct)

    # Assert the network nodes and edges are as expected
    assert network.number_of_nodes() == len(nodes)
    assert network.number_of_edges() == len(edges)
    for node in nodes:
        assert network.nodes[node[0]]['pos'] == node[1]['pos']
    for edge in edges:
        assert network.edges[edge[0:2]]['weight'] == edge[2]['weight']

    # Query the edge_list property and check it is as expected
    edge_list = network.edge_list
    assert len(edge_list) == len(edges)
    assert np.alltrue(edge_list == np.array([edge[0:2] for edge in edges]))

    # Test we can remove nodes and edges
    # Note that removing a node will also remove all edges connected to that node
    # so removing node 0 also removes 2 edges
    if mode == "individual":
        network.remove_node(1)  # Remove node 1
        network.remove_edge(3, 4)  # Remove edge (3, 4)
    else:
        network.remove_nodes_from([1])  # Remove node 1
        network.remove_edges_from([(3, 4)])  # Remove edge (3, 4)

    assert network.number_of_nodes() == len(nodes) - 1  # 3 nodes left
    assert network.number_of_edges() == len(edges) - 3  # 1 edge left

    # Test adding nodes with common position
    network.add_nodes_from([5, 6], pos=(0, 0))
    assert network.number_of_nodes() == len(nodes) + 1  # 3 + 2 = 5 nodes
    assert network.nodes[5]['pos'] == (0, 0)
    assert network.nodes[6]['pos'] == (0, 0)

    # Test we can clear the edges and nodes
    network.clear_edges()
    assert network.number_of_edges() == 0 and network.number_of_nodes() == len(nodes) + 1
    network.clear()
    assert network.number_of_nodes() == 0 and network.number_of_edges() == 0


def test_road_netword_to_gdf(nodes, edges):
    """Test converting a road network to GeoDataFrame"""
    network = RoadNetwork()
    network.add_nodes_from(nodes)
    network.add_edges_from(edges)

    gdf = network.to_gdf()
    assert gdf.shape[0] == len(edges)
    assert gdf.shape[1] == 4
    assert gdf.crs is None
    for i, e in enumerate(network.edges):
        assert gdf.iloc[i].geometry.coords[0] == network.nodes[e[0]]['pos']
        assert gdf.iloc[i].geometry.coords[1] == network.nodes[e[1]]['pos']
        assert gdf.iloc[i].weight == network.edges[e]['weight']
        assert gdf.iloc[i].from_node == e[0]
        assert gdf.iloc[i].to_node == e[1]


def test_road_network_shortest_path(nodes, edges):
    """Test finding the shortest path in a road network"""
    network = RoadNetwork()
    network.add_nodes_from(nodes)
    network.add_edges_from(edges)

    # Test shortest path between nodes 1 and 4
    node_paths = network.shortest_path(1, 4)
    assert node_paths[(1, 4)] == [1, 3, 4]
    edge_paths = network.shortest_path(1, 4, path_type='edge')
    assert edge_paths[(1, 4)] == [1, 3]
    paths = network.shortest_path(1, 4, path_type='both')
    node_paths, edge_paths = paths['node'], paths['edge']
    assert node_paths[(1, 4)] == [1, 3, 4]
    assert edge_paths[(1, 4)] == [1, 3]

    # Test shortest path between all nodes and nodes 2 and 4
    node_paths = network.shortest_path(None, [2, 4])
    assert len(node_paths) == 4
    assert node_paths[(1, 2)] == [1, 2]
    assert node_paths[(1, 4)] == [1, 3, 4]
    assert node_paths[(2, 4)] == [2, 4]
    assert node_paths[(3, 4)] == [3, 4]
    edge_paths = network.shortest_path(None, [2, 4], path_type='edge')
    assert len(edge_paths) == 4
    assert edge_paths[(1, 2)] == [0]
    assert edge_paths[(1, 4)] == [1, 3]
    assert edge_paths[(2, 4)] == [2]
    assert edge_paths[(3, 4)] == [3]

    # Test shortest path between nodes 2 and 4 and all nodes
    node_paths = network.shortest_path([2, 4], None)
    assert len(node_paths) == 1
    assert node_paths[(2, 4)] == [2, 4]
    edge_paths = network.shortest_path([2, 4], None, path_type='edge')
    assert len(edge_paths) == 1
    assert edge_paths[(2, 4)] == [2]

    # Test shortest path between all nodes and all nodes
    node_paths = network.shortest_path()
    assert len(node_paths) == 5
    assert node_paths[(1, 2)] == [1, 2]
    assert node_paths[(1, 3)] == [1, 3]
    assert node_paths[(1, 4)] == [1, 3, 4]
    assert node_paths[(2, 4)] == [2, 4]
    assert node_paths[(3, 4)] == [3, 4]
    edge_paths = network.shortest_path(path_type='edge')
    assert len(edge_paths) == 5
    assert edge_paths[(1, 2)] == [0]
    assert edge_paths[(1, 3)] == [1]
    assert edge_paths[(1, 4)] == [1, 3]
    assert edge_paths[(2, 4)] == [2]
    assert edge_paths[(3, 4)] == [3]

    # Test shortest path between unreachable nodes
    paths = network.shortest_path(2, 3, path_type='both')
    node_paths, edge_paths = paths['node'], paths['edge']
    assert len(node_paths) == 0
    assert len(edge_paths) == 0
    paths = network.shortest_path([2, 3], [3, 4], path_type='both')
    node_paths, edge_paths = paths['node'], paths['edge']
    assert len(node_paths) == 2
    assert node_paths[(2, 4)] == [2, 4]
    assert node_paths[(3, 4)] == [3, 4]
    assert len(edge_paths) == 2
    assert edge_paths[(2, 4)] == [2]
    assert edge_paths[(3, 4)] == [3]


def test_road_network_errors(nodes, edges):
    """Test error handling in road network"""
    network = RoadNetwork()

    # Test adding nodes and edges with invalid data
    # ---------------------------------------------
    # 1) adding node with invalid id
    with pytest.raises(TypeError, match="Road network nodes must be positive integers"):
        network.add_node('a', pos=(0, 0))
    with pytest.raises(TypeError, match="Road network nodes must be positive integers"):
        network.add_node(0, pos=(0, 0))
    # 2) adding node with missing position
    with pytest.raises(ValueError, match="Road network nodes must have a 'pos' attribute"):
        network.add_node(1)
    # 3) adding bulk nodes with invalid data
    with pytest.raises(ValueError, match="Invalid node format"):
        nodes.append({-1: {'pos': (0, 0)}})  # Wrong node format
        network.add_nodes_from(nodes)
    # 4) adding edge with missing weight
    with pytest.raises(ValueError, match="Road network edges must have a 'weight' attribute"):
        network.add_edge(0, 1)

    # Test update method
    # ------------------
    with pytest.raises(NotImplementedError, match="RoadNetwork does not support update"):
        network.update()  # Update method not implemented
