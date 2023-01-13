from ..base import Property
from .base import Type
from ..sensor.sensor import Sensor

from typing import Set, List, Collection, Tuple
import networkx as nx
import plotly.graph_objects as go


class Node(Type):
    """Base node class"""
    position: Tuple[float] = Property(
        default=None,
        doc="Cartesian coordinates for node")
    label: str = Property(
        default=None,
        doc="Label to be displayed on graph")
    colour: str = Property(
        default=None,
        doc = 'Colour to be displayed on graph')
    shape: str = Property(
        default=None,
        doc='Shape used to display nodes')


class SensorNode(Node):
    """A node corresponding to a Sensor. Fresh data is created here,
    and possibly processed as well"""
    sensor: Sensor = Property(doc="Sensor corresponding to this node")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.colour:
            self.colour = '#1f77b4'
        if not self.shape:
            self.shape = 'square'


class ProcessingNode(Type):
    """A node that does not measure new data, but does process data it receives"""
    # Latency property could go here
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.colour:
            self.colour = '#1f77b4'
        if not self.shape:
            self.shape = 'square'


class RepeaterNode(Type):
    """A node which simply passes data along to others, without manipulating the data itself. """
    # Latency property could go here
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.colour:
            self.colour = '#ff7f0e'
        if not self.shape:
            self.shape = 'circle'

class Architecture(Type):
    edge_list: Collection = Property(
        default=None,
        doc="A Collection of edges between nodes. For A to be connected to B we would have (A, B)"
            "be a member of this list. Default is None")
    node_set: Set[Node] = Property(
        default=None,
        doc="A Set of all nodes involved, each of which may be a sensor, processing node, "
            "or repeater node. If provided, used to check all Nodes given are included "
            "in edges of the graph. Default is None")
    force_connected: bool = Property(
        default=True,
        doc="If True, the undirected version of the graph must be connected, ie. all nodes should "
            "be connected via some path. Set this to False to allow an unconnected architecture. "
            "Default is True"
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(self.edge_list, Collection) and not isinstance(self.edge_list, List):
            self.edge_list = list(self.edge_list)
        if self.edge_list and len(self.edge_list) > 0:
            self.di_graph = nx.to_networkx_graph(self.edge_list, create_using=nx.DiGraph)
            if self.force_connected and not self.is_connected:
                raise ValueError("The graph is not connected. Use force_connected=False, "
                                 "if you wish to override this requirement")
        else:
            if self.node_set:
                raise TypeError("Edge list must be provided, if a node set is. ")
            self.di_graph = nx.DiGraph()
        if self.node_set:
            if not set(self.di_graph.nodes) == self.node_set:
                raise ValueError("Provided node set does not match nodes on graph")
        else:
            self.node_set = set(self.di_graph.nodes)

    def plot(self, use_positions=True, label_nodes=False):
        """Creates a plot of the directed graph"""
        edge_x = []
        edge_y = []
        for edge in self.edge_list:
            if use_positions:
                x0, y0 = edge[0].position
                x1, y1 = edge[1].position
            else:
                x0, y0 = edge[0].position
                x1, y1 = edge[1].position # Add if statement to display as hierarchical if hierarchical
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')

        node_x = []
        node_y = []
        for node in self.node_set:
            node_x.append(node.position[0])
            node_y.append(node.position[1])

        mode = 'markers+text' if label_nodes else 'markers'
        marker_shape = self.
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode= mode,
            hoverinfo='text',
            marker=dict(
                showscale=True,
                # colorscale options
                # 'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
                # 'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
                # 'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
                colorscale='YlGnBu',
                reversescale=True,
                color=[],
                size=50,
                colorbar=dict(
                    thickness=15,
                    title='Node Connections',
                    xanchor='left',
                    titleside='right'
                ),
                line_width=2))

        node_trace.marker.color =
        node_trace.text = node_text
        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title='<br>Network graph made with Python',
                            titlefont_size=16,
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20, l=5, r=5, t=40),
                            annotations=[dict(
                                text="Python code: <a href='https://plotly.com/ipython-notebooks/network-graphs/'> https://plotly.com/ipython-notebooks/network-graphs/</a>",
                                showarrow=False,
                                xref="paper", yref="paper",
                                x=0.005, y=-0.002)],
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                        )
        fig.show()

    @property
    def density(self):
        """Returns the density of the graph, ie. the proportion of possible edges between nodes
        that exist in the graph"""
        num_nodes = len(self.node_set)
        num_edges = len(self.edge_list)
        architecture_density = num_edges/((num_nodes*(num_nodes-1))/2)
        return architecture_density

    @property
    def is_hierarchical(self):
        """Returns `True` if the :class:`Architecture` is hierarchical, otherwise `False`"""
        if len(list(nx.simple_cycles(self.di_graph))) > 0 or not self.is_connected:
            return False
        else:
            return True

    @property
    def is_connected(self):
        return nx.is_connected(self.to_undirected)

    @property
    def to_undirected(self):
        return self.di_graph.to_undirected()

    def __len__(self):
        return len(self.di_graph)


class InformationArchitecture(Architecture):
    """The architecture for how information is shared through the network. Node A is "
    "connected to Node B if and only if the information A creates by processing and/or "
    "sensing is received and opened by B without modification by another node. """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for node in self.node_set:
            if isinstance(node, RepeaterNode):
                raise TypeError("Information architecture should not contain any repeater nodes")


class NetworkArchitecture(Architecture):
    """The architecture for how data is propagated through the network. Node A is connected "
            "to Node B if and only if A sends its data through B. """


class CombinedArchitecture(Type):
    """Contains an information and a network architecture that pertain to the same scenario. """
    information_architecture: InformationArchitecture = Property(
        doc="The information architecture for how information is shared. ")
    network_architecture: NetworkArchitecture = Property(
        doc="The architecture for how data is propagated through the network. Node A is connected "
            "to Node B if and only if A sends its data through B. "
    )