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

from stonesoup.functions.graph import normalise_re, calc_edge_len, get_xy_from_range_edge


# The next fixture defines a simple network with 4 nodes and 5 edges that looks like:
#        (.25)
#       3-->--4
#       |     ^
#  (.5) ^     | (1.)     where the numbers in brackets are the edge weights and the arrows
#       |     v          indicate the direction of the edge
#       1-->--2
#        (1.)
@pytest.fixture()
def graph():
    # Nodes are defined as a tuple of (node, {'pos': (x, y)})
    nodes = [(1, {'pos': (0, 0)}),
             (2, {'pos': (1, 0)}),
             (3, {'pos': (0, 1)}),
             (4, {'pos': (1, 1)})]
    # Edges are defined as a tuple of (node1, node2, {'weight': weight})
    edges = [(1, 2, {'weight': 1.}),
             (1, 3, {'weight': .5}),
             (2, 4, {'weight': 1.}),
             (3, 4, {'weight': .25}),
             (4, 2, {'weight': 1.})]
    # Create the network
    network = RoadNetwork()
    network.add_nodes_from(nodes)
    network.add_edges_from(edges)
    return network


@pytest.mark.parametrize(
    'r_i, e_i, path, expected',
    [
        # Particle is on the start node
        (0., 0, [0, 2], (0., 0)),
        # Particle is on the end node
        (1., 0, [0, 2], (1., 0)),
        # Particle is outside the path (before the start node)
        (-0.2, 0, [0, 2], (0., 0)),
        # Particle range is larger than the edge length
        (2.5, 1, [1, 4, 3], (0.5, 3)),
        # Particle range is negative (should transition to previous edge)
        (-0.2, 3, [1, 4, 3], (0.8, 4)),
        # Particle range is negative (should transition to previous edges)
        (-5., 3, [1, 4, 3], (0., 1)),
        # Particle is outside the path (after the end node)
        (5., 1, [1, 4, 3], (1.0, 3)),
        # Particle is outside the path (after the end node)
        (5., 3, [1, 4, 3], (1.0, 3)),
        # Particle range is larger than the edge length and longer than the path
        (5., 4, [1, 4, 3], (1.0, 3)),
    ]
)
def test_normalise_re(r_i, e_i, path, expected, graph):
    """Test normalise_re function"""
    result = normalise_re(r_i, e_i, path, graph)
    assert result == expected


def test_calc_edge_len(graph):
    """Test calc_edge_len function"""

    # Test the edge lengths for existing edges (should all be 1)
    for i, _ in enumerate(graph.edge_list):
        edge_len = calc_edge_len(i, graph)
        assert edge_len == 1

    # Add a diagonal edge and test the edge length (should be sqrt(2))
    graph.add_edge(1, 4, weight=2.)
    edge_idx = np.flatnonzero(np.all(graph.edge_list == (1, 4), axis=1))[0]
    edge_len = calc_edge_len(edge_idx, graph)
    assert edge_len == np.sqrt(2)


@pytest.mark.parametrize(
    'r, e, expected',
    [
        # Single range, single edge
        (0., 0, (0., 0.)),
        (1., 0, (1., 0.)),
        (.5, 0, (.5, 0.)),
        (0., 1, (0., 0.)),
        (1., 1, (0., 1.)),
        (.5, 1, (0., .5)),
        (0., 2, (1., 0.)),
        (1., 2, (1., 1.)),
        (0.5, 2, (1., .5)),
        (0., 3, (0., 1.)),
        (1., 3, (1., 1.)),
        (.5, 3, (.5, 1.)),
        # List of ranges, single edge
        ([0., .5, 1.], 0, [[0., 0.], [.5, 0.], [1., 0.]]),
        # Single range, list of edges
        (0., [0, 1, 2, 3], [[0., 0.], [0., 0.], [1., 0.], [0., 1.]]),
        # List of ranges, list of edges
        ([0., .5, 1.], [0, 1, 2], [[0., 0.], [0., .5], [1., 1.]]),
    ]
)
def test_xy_from_range_edge(r, e, expected, graph):
    xy = get_xy_from_range_edge(r, e, graph)
    assert np.allclose(xy.T, expected)
