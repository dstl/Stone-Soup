import itertools
from typing import Sequence, Tuple, Dict

import numpy as np
import geopandas
import networkx as nx
from shapely import LineString


class RoadNetwork(nx.DiGraph):
    """Road network type

    A road network is a directed graph, where nodes represent intersections and edges represent
    roads. Nodes must have a 'pos' attribute, which is a tuple of (x, y) coordinates. Edges must
    have a `'weight'` attribute, which is a float representing the weight of the edges
    (e.g. length).

    The underlying data structure is a :class:`networkx.DiGraph`, with some limitations on the
    allowed node and edge attributes, and some additional methods.

    .. note::

        The current implementation is not optimised for continuous updates of the graph structure.
        More importantly, although possible, it is not advisable to interleave structure update
        calls (e.g. :meth:`add_edge`) with query calls (e.g. :meth:`shortest_path`). Instead, the
        intended use is to construct the network once and then perform repeated queries.

    Parameters
    ----------
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._edge_list = []
        self._short_paths_cache = {}

    @property
    def edge_list(self):
        if self._edge_list is None:
            self._edge_list = np.array(list(self.edges.keys()))
        return self._edge_list

    def add_node(self, n, **attr):
        """Add a single node n and update node attributes.

        Parameters
        ----------
        n : int
            A node identifier
        attr
            Node attributes to update. Must contain a 'pos' attribute, that is a tuple of (x, y)
            coordinates.
        """
        if not isinstance(n, int) or n <= 0:
            raise TypeError("Road network nodes must be positive integers")
        if "pos" not in attr:
            raise ValueError("Road network nodes must have a 'pos' attribute")
        super().add_node(n, **attr)

    def add_nodes_from(self, nodes, **attr):
        """Add multiple nodes.

        Parameters
        ----------
        nodes : Union[Sequence[int], Sequence[Tuple[int, Dict]]]
            A container of nodes to be added. Can be a list of node identifiers, or a list of
            tuples of (node identifier, node attributes dict).
        attr
            Update attributes for all nodes in nodes. Node attributes specified in nodes as a
            tuple take precedence over attributes specified via keyword arguments.
        """
        for n in nodes:
            if isinstance(n, int):
                self.add_node(n, **attr)
            elif isinstance(n, tuple):
                self.add_node(n[0], **{**attr, **n[1]})
            else:
                raise ValueError("Invalid node format. Elements of nodes must be integers, or "
                                 "tuples of (int, dict)")

    def remove_node(self, n):
        """Remove node n.

        Parameters
        ----------
        n : int
            A node identifier
        """
        super().remove_node(n)

    def remove_nodes_from(self, nodes):
        """Remove multiple nodes.

        Parameters
        ----------
        nodes : Sequence[int]
            A sequence of node identifiers
        """
        for n in nodes:
            self.remove_node(n)

    def add_edge(self, u, v, **attr):
        """Add an edge between nodes u and v.

        Parameters
        ----------
        u : int
            Start node identifier
        v : int
            End node identifier
        attr
            Edge attributes to update. Must contain a 'weight' attribute, that is a float.
        """
        if "weight" not in attr:
            raise ValueError("Road network edges must have a 'weight' attribute that is a float")
        super().add_edge(u, v, **attr)
        self._edge_list = None
        self._short_paths_cache = {}

    def add_edges_from(self, ebunch, **kwargs):
        """Add multiple edges.

        Parameters
        ----------
        ebunch : Sequence[Tuple[int, int, Dict]]
            An iterable of edges. Edges must be specified as tuples (u, v, d) where d
            is a dict of edge attributes (must contain a 'weight' attribute that is a float).
        kwargs
            Update attributes for all edges in ebunch. Edge attributes specified in ebunch as a
            tuple take precedence over attributes specified via keyword arguments.
        """
        for e in ebunch:
            u, v = e[0:2]
            attr = e[2]
            attr.update(kwargs)
            self.add_edge(u, v, **attr)

    def remove_edge(self, u, v):
        """ Remove an edge between nodes u and v.

        Parameters
        ----------
        u : int
            Start node identifier
        v : int
            End node identifier
        """
        super().remove_edge(u, v)
        self._edge_list = None
        self._short_paths_cache = {}

    def remove_edges_from(self, ebunch):
        """Remove multiple edges.

        Parameters
        ----------
        ebunch : Sequence[Tuple[int, int]]
            An iterable of edges. Edges must be specified as tuples (u, v) where u and v are
            node identifiers.
        """
        for e in ebunch:
            u, v = e
            self.remove_edge(u, v)

    def update(self, edges=None, nodes=None):
        """ Not implemented

        Raises
        ------
        NotImplementedError
            This method is not implemented for :class:`~.RoadNetwork`.
        """
        raise NotImplementedError("RoadNetwork does not support update")

    def clear(self):
        """Remove all nodes and edges from the graph."""
        super().clear()
        self._edge_list = None
        self._short_paths_cache = {}

    def clear_edges(self):
        """Remove all edges from the graph without altering nodes."""
        super().clear_edges()
        self._edge_list = None
        self._short_paths_cache = {}

    def to_gdf(self, **kwargs):
        """Convert the road network to a GeoDataFrame.

        The GeoDataFrame has the following columns:

            - 'geometry': LineString
                A LineString representing the edge geometry.
            - 'weight': float
                The edge weight.
            - 'from_node': int
                The node identifier of the start node.
            - 'to_node': int
                The node identifier of the end node.
        """
        d = {'geometry': [], 'weight': [], 'from_node': [], 'to_node': []}
        for e in self.edges:
            d['geometry'].append(
                LineString([self.nodes[e[0]]['pos'], self.nodes[e[1]]['pos']])
            )
            d['weight'].append(self.edges[e]['weight'])
            d['from_node'].append(e[0])
            d['to_node'].append(e[1])
        gdf = geopandas.GeoDataFrame(d, **kwargs)
        return gdf

    @classmethod
    def from_dict(cls, dct):
        """Create a RoadNetwork from a dictionary.

        Parameters
        ----------
        dct : dict
            A dictionary with keys 'nodes' and 'edges'. The value of 'nodes' is a dictionary
            mapping node identifiers to a dictionary of node attributes. The value of 'edges' is
            a dictionary mapping edge identifiers to edge attributes.

        Returns
        -------
        RoadNetwork
            A road network object

        Examples
        --------
        >>> dct = {'nodes': {0: {'pos': (0, 0)}, 1: {'pos': (1, 0)},
        ...                  2: {'pos': (0, 1)}, 3: {'pos': (1, 1)}},
        ...        'edges': {(0, 1): {'weight': 1}, (0, 2): {'weight': 1},
        ...                  (1, 3): {'weight': 1}, (2, 3): {'weight': 1}}}
        >>> net = RoadNetwork.from_dict(dct)
        """
        # Create empty graph object
        net = RoadNetwork()

        # Add nodes to graph
        for node, attr in dct['nodes'].items():
            net.add_node(node, **attr)

        # Add edges to graph
        for edge, attr in dct['edges'].items():
            net.add_edge(*edge, **attr)

        return net

    def plot_network(self, ax, node_size=0.1, width=0.5, with_labels=False):
        # Get node positions
        pos = nx.get_node_attributes(self, 'pos')

        nx.draw_networkx(self, pos, arrows=False, ax=ax, node_size=node_size, width=width,
                         with_labels=with_labels)

    def highlight_nodes(self, ax, nodes, node_size=0.1, node_color='m', node_shape='s', label=None):
        # Get node positions
        pos = nx.get_node_attributes(self, 'pos')

        return nx.draw_networkx_nodes(self, pos, nodelist=nodes, ax=ax, node_size=node_size,
                                      node_color=node_color, node_shape=node_shape, label=label)

    def highlight_edges(self, ax, edges_idx, width=2.0, edge_color='m', style='solid', arrows=False,
                        label=None):
        edges = self.edge_list[edges_idx]
        pos = nx.get_node_attributes(self, 'pos')

        return nx.draw_networkx_edges(self, pos, edgelist=edges, ax=ax, width=width,
                                      edge_color=edge_color,
                                      style=style, arrows=arrows, label=label)

    def shortest_path(self, source=None, target=None, weight='weight', method='dijkstra',
                      path_type='node'):
        """Compute the shortest path(s) between source and target.

        This method is a wrapper around the NetworkX `nx.shortest_path()` function, that adds some
        additional functionality:

            - While the NetworkX function only allows to compute the shortest path(s) between a
              single source and a single target, this method allows to compute the shortest
              path(s) between a source node (or a list of source nodes) and a target node (or a
              list of target nodes).
            - The method allows to specify the type of path to return. By default, the method
              returns a list of node identifiers (same as NetworkX). However, it is also
              possible to return a list of edges, or even both types in a single call.
            - When a path is computed, it is stored in a cached dictionary. When possible, the
              method will attempt to retrieve the shortest path(s) from this cache, to improve
              performance.


        Parameters
        ----------
        source : int or list[int] or None
            A node identifier, or a list of node identifiers. If `None`, the shortest path(s) from
            all nodes to the specified target node(s) are computed.
        target : int or list[int] or None
            A node identifier, or a list of node identifiers. If `None`, the shortest path(s) to
            all nodes from the specified source node(s) are computed.
        weight : str
            The edge attribute to use as edge weights. Must be a float. Default is `'weight'`.
        method : str
            The method to use to compute the shortest path(s). Must be one of `'dijkstra'` or
            `'bellman-ford'`. Default is `'dijkstra'`.
        path_type : str
            The type of path to return. Must be one of `'node'`, `'edge'` or `'both'`. If `'node'`,
            each shortest path is a list of node identifiers. If `'edge'`, each shortest path is a
            list of edges. If `'both'`, then both types of paths are returned (see bellow). Default
            is `'node'`.

        Returns
        -------
        dict
            The shortest path(s) between source and target. The format depends on the value of
            `path_type`:

                - If `path_type` is `'node'`, return a dict whose keys are tuples (source, target)
                  and values are lists of node identifiers.
                - If `path_type` is `'edge'`, return a dict whose keys are tuples (source, target)
                  and values are lists of edge identifiers.
                - If `path_type` is `'both'`, return a dict with keys `'node'` and `'edge'`, where
                  the values follow the same format as above.

        """

        # Source and target must be lists, or None
        source = [source] if (source is not None and not isinstance(source, Sequence)) else source
        target = [target] if (target is not None and not isinstance(target, Sequence)) else target

        # First attempt to get the shortest path(s) from cache
        try:
            return self._get_shortest_path_from_cache(source, target, weight, method, path_type)
        except KeyError:
            pass

        # If retrieving the shortest path(s) from cache failed, proceed to compute them
        short_paths = self._compute_short_paths(source, target, weight, method)

        # Update cache
        self._update_short_paths_cache(short_paths, weight, method)

        # Return the requested path(s)
        return self._format_shortest_paths_output(short_paths, path_type)

    def _get_shortest_path_from_cache(self, source=None, target=None, weight='weight',
                                      method='dijkstra', path_type='both'):
        short_paths = {
            'node': dict(),
            'edge': dict()
        }

        if source is None:
            source = self.nodes
        if target is None:
            target = self.nodes

        # Attempt to get the requested path(s) from cache
        try:
            for s, t in itertools.product(source, target):
                if s == t:
                    continue
                if path_type == 'both' or path_type == 'node':
                    short_paths['node'][(s, t)] = \
                        self._short_paths_cache[method][weight]['node'][(s, t)]
                if path_type == 'both' or path_type == 'edge':
                    short_paths['edge'][(s, t)] = \
                        self._short_paths_cache[method][weight]['edge'][(s, t)]
        except KeyError as e:
            raise KeyError('Shortest path not found for the given source and target')

        # Return the requested path(s)
        return self._format_shortest_paths_output(short_paths, path_type)

    def _compute_short_paths(self, source, target, weight, method):
        paths = dict()
        if source is None and target is None:
            # Compute all shortest paths, from all nodes, to all nodes
            paths = nx.shortest_path(self, weight=weight, method=method)
        elif source is None or target is None:
            # If source or target is None, compute the shortest paths from/to the given nodes
            iter_ = source if source is not None else target
            for x in iter_:
                extra_kwargs = {'source': x} if source is not None else {'target': x}
                x_paths = nx.shortest_path(self, weight=weight, method=method, **extra_kwargs)
                for y, node_path in x_paths.items():
                    if x == y:
                        continue
                    s = x if source is not None else y
                    t = y if source is not None else x
                    try:
                        paths[s][t] = node_path
                    except KeyError:
                        paths[s] = {t: node_path}
        else:
            # Compute shortest paths from each source to each target
            for s, t in itertools.product(source, target):
                try:
                    path = nx.shortest_path(self, source=s, target=t, weight=weight, method=method)
                except nx.NetworkXNoPath:
                    continue

                try:
                    paths[s][t] = path
                except KeyError:
                    paths[s] = {t: path}

        # Convert computed paths to node and edge paths
        short_paths = {
            'node': dict(),
            'edge': dict()
        }
        edges_index = {edge: i for i, edge in enumerate(self.edges)}
        for s, dct in paths.items():
            for t, node_path in dct.items():
                if s == t:
                    continue
                path_edges = zip(node_path, node_path[1:])
                edge_path = [edges_index[edge] for edge in path_edges]
                short_paths['node'][(s, t)], short_paths['edge'][(s, t)] = (node_path, edge_path)
        return short_paths

    def _update_short_paths_cache(self, paths, weight='weight', method='dijkstra'):
        if method not in self._short_paths_cache:
            self._short_paths_cache[method] = dict()
        if weight not in self._short_paths_cache[method]:
            self._short_paths_cache[method][weight] = {'node': dict(), 'edge': dict()}
        self._short_paths_cache[method][weight]['node'].update(paths['node'])
        self._short_paths_cache[method][weight]['edge'].update(paths['edge'])

    @staticmethod
    def _format_shortest_paths_output(paths, path_type):
        # Return output
        if path_type == 'node':
            return paths['node']
        elif path_type == 'edge':
            return paths['edge']
        return paths
