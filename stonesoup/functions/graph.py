import numpy as np

from stonesoup.functions import cart2pol, pol2cart


def normalise_re(r_i, e_i, path, G):
    """ Normalise the given range based on the provided edge and path

    Parameters
    ----------
    r_i: float
        The range to be normalised (i.e. the distance travelled along the edge)
    e_i: int
        The current edge index
    path: list of int
        The path to the destination, as a list of edge indices
    G: RoadNetwork
        The road network

    Returns
    -------
    float
        The normalised range
    int
        The new edge index

    """
    edge_len = calc_edge_len(e_i, G)
    idx = np.where(path == e_i)[0]

    if not len(idx):
        # If idx is empty, it means that the edge does not exist on the given
        # path to a destination. Therefore, this is an invalid particle, for
        # which nothing can be done, except to cap the range to the edge limits.
        if r_i > edge_len:
            r_i = edge_len
        elif r_i < 0:
            r_i = 0
    else:
        idx = idx[0]
        while r_i > edge_len or r_i < 0:
            if r_i > edge_len:
                if len(path) > idx+1:
                    # If particle has NOT reached the end of the path
                    r_i = r_i - edge_len
                    idx = idx + 1
                    e_i = path[idx]
                    edge_len = calc_edge_len(e_i, G)
                    if len(path) == idx + 1:
                        # If particle has reached the end of the path
                        if r_i > edge_len:
                            # Cap r_i to edge_length
                            r_i = edge_len
                        break
                else:
                    # Cap r_i to edge_length
                    r_i = edge_len
                    break
            elif r_i < 0:
                if idx > 0:
                    # If particle is within the path limits
                    idx = idx - 1
                    e_i = path[idx]
                    edge_len = calc_edge_len(e_i, G)
                    r_i = edge_len + r_i
                else:
                    # Else if the particle position is beyond the path
                    # limits, the set its range to 0.
                    r_i = 0
                    break

    return r_i, e_i


def calc_edge_len(e, G):
    """ Calculate the length of the given edge

    Parameters
    ----------
    e: int
        The edge index
    G: RoadNetwork
        The graph

    Returns
    -------
    float
        The length of the edge
    """
    edge = G.edge_list[int(e)]
    p1 = G.nodes[edge[0]]['pos']
    p2 = G.nodes[edge[1]]['pos']
    return np.hypot(p2[0] - p1[0], p2[1] - p1[1])


def get_xy_from_range_edge(r, e, G):

    r = np.atleast_1d(r)
    e = np.atleast_1d(e).astype(int)

    endnodes = G.edge_list[e]

    # Get endnode coordinates
    pos = G.nodes(data='pos')
    p1 = np.array([pos[en] for en in endnodes[:, 0]]).T
    p2 = np.array([pos[en] for en in endnodes[:, 1]]).T

    # Normalise coordinates of p2, assuming p1 is the origin
    p2norm = p2-p1

    # Compute angle between p2 and p1
    _, theta = cart2pol(p2norm[0, :], p2norm[1, :])

    # Compute XY normalised, assuming p1 is the origin
    x_norm, y_norm = pol2cart(r, theta)
    xy_norm = np.array([x_norm, y_norm])

    # Compute transformed XY
    xy = p1 + xy_norm

    return xy