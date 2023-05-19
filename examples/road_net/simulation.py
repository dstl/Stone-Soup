import networkx as nx
import numpy as np
from datetime import datetime,timedelta

from stonesoup.functions import pol2cart, cart2pol
from stonesoup.types.array import StateVector
from stonesoup.types.graph import RoadNetwork
from stonesoup.types.groundtruth import GroundTruthPath
from stonesoup.types.groundtruth import GroundTruthState


def simulate(G: RoadNetwork, source: int, destination: int, speed: float, track_id: int = None,
             timestamp_init: datetime = datetime.now(), interval: timedelta = timedelta(seconds=1)):
    """ Simulate a moving target along the network

    Parameters
    ----------
    G: :class:`nx.DiGraph`
        The road network
    source: :class:`int`
        The source node index
    destination: :class:`int`
        The destination node index
    speed: :class:`float`
        The speed of the target (assumed to be constant)
    timestamp_init: :class:`datetime.datetime`
        The initial timestamp
    interval: :class:`datetime.timedelta`
        The interval between ground-truth reports

    Returns
    -------
    :class:`~.GroundTruthPath`
        The ground truth path
    :class:`list`
        The list of node idxs in the route from source to destination
    :class:`list`
        The list of edge idxs in the route from source to destination

    """
    # Compute shortest path to destination
    gnd_route_tmp = G.shortest_path(source, destination, path_type='both')
    gnd_route_n = gnd_route_tmp['node'][(source, destination)]
    gnd_route_e = gnd_route_tmp['edge'][(source, destination)]
    path_len = len(gnd_route_n)

    # Get the node positions
    pos = nx.get_node_attributes(G, 'pos')

    # Initialize the Ground-Truth path
    sv = StateVector(np.array(pos[gnd_route_n[0]]))
    timestamp = timestamp_init
    state = GroundTruthState(sv, timestamp=timestamp)
    gnd_path = GroundTruthPath([state], id=track_id)

    r = 0  # Stores the distance travelled (range) along a given edge
    overflow = False  # Indicates when the range has overflown to a new edge

    # Iterate over the nodes in the route
    for k in range(1, path_len):
        # Index and position of last visited node
        node_km1 = gnd_route_n[k - 1]
        pos_km1 = np.array(pos[node_km1])
        # Index and position of next node along the edge
        node_k = gnd_route_n[k]
        pos_k = np.array(pos[node_k])
        # Compute distance (max range) and angle between the two nodes
        dpos = pos_k - pos_km1
        r_max, a = cart2pol(dpos[0], dpos[1])

        # Iterate until the next node has been reached
        reached = False
        while not reached:
            # Only add to the range if not overflown
            if not overflow:
                r += speed*interval.total_seconds()

            # If r falls within the max range of the edge
            # then we need to report the new gnd position
            # and continue until we have reached the next node
            if r <= r_max:
                overflow = False    # Reset the overflow flag
                # Compute and store the new gnd position
                x, y = pol2cart(r,a)
                x_i = pos_km1 + np.array([x, y])
                sv = StateVector(np.array(x_i))
                timestamp += interval
                state = GroundTruthState(sv, timestamp=timestamp)
                gnd_path.append(state)
                # If r == r_max it means we have reached the next node
                if r == r_max:
                    reached = True  # Signal that next node is reached
                    r = 0           # Reset r
            # Else if r is greater than the edge length, then
            # skip to the next edge, without reporting the gnd
            # position, unless we have reached the destination
            elif r > r_max:
                r -= r_max          # Update r to reflect the cross-over
                overflow = True     # Set the overflow flag
                reached = True      # Signal that we have reached the next node
                # If k == path_len-1 it means we have reached the destination
                # meaning that we should report the new position
                if k == path_len-1:
                    x, y = pol2cart(r_max, a)
                    x_i = pos_km1 + np.array([x, y])
                    sv = StateVector(np.array(x_i))
                    timestamp += interval
                    state = GroundTruthState(sv, timestamp=timestamp)
                    gnd_path.append(state)

    return gnd_path, gnd_route_n, gnd_route_e