import networkx as nx


def plot_network(G, ax, node_size=0.1, width=0.5, with_labels=False):
    # Get node positions
    pos = nx.get_node_attributes(G, 'pos')

    nx.draw_networkx(G, pos, arrows=False, ax=ax, node_size=node_size, width=width,
                     with_labels=with_labels)


def highlight_nodes(G, ax, nodes, node_size=0.1, node_color='m', node_shape='s', label=None):
    # Get node positions
    pos = nx.get_node_attributes(G, 'pos')

    return nx.draw_networkx_nodes(G, pos, nodelist=nodes, ax=ax, node_size=node_size,
                                  node_color=node_color, node_shape=node_shape, label=label)


def highlight_edges(G, ax, edges_idx, width=2.0, edge_color='m', style='solid', arrows=False,
                    label=None):
    edges = G.edge_list[edges_idx]
    pos = nx.get_node_attributes(G, 'pos')

    return nx.draw_networkx_edges(G, pos, edgelist=edges, ax=ax, width=width,
                                  edge_color=edge_color,
                                  style=style, arrows=arrows, label=label)