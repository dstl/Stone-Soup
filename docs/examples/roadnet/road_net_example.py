"""
=======================================
Joint Tracking & Destination Estimation
=======================================
"""
# %%
# This example demonstrates how to perform joint tracking and destination estimation of a target travelling along a
# road network, using the SMC Sampler introduced in [#]_.
#
# The target is assumed to be moving with near constant velocity, and the measurements are noisy positional observations
# of the target's location. To estimate the target's destination, we use a particle filter with a transition model that
# incorporates the road network structure. The particle filter is used to estimate the target's state and destination
# based on the measurements received, under the assumption that the target is moving along the optimal path to its
# destination.

# %%
# Setup
# -----
# First, some general packages used throughout the example are imported and
# random number generation is seeded for repeatability.
#
from datetime import datetime, timedelta

import numpy as np
import networkx as nx
from scipy.stats import multivariate_normal as mvn


np.random.seed(818)


# %%
# Load road network
# -----------------
# For this example, we will use a road network stored in a :download:`local pickle file <../../../examples/roadnet/road_net.pkl>`.
# The network is serialised as a dictionary, which is then deserialised into a :class:`~stonesoup.types.graph.RoadNetwork` object.
# Information about the expected dictionary format can be found in the :py:meth:`~stonesoup.types.graph.RoadNetwork.from_dict` method.
#
# The :class:`~stonesoup.types.graph.RoadNetwork` class is a subclass of the :class:`networkx.DiGraph` class, and provides additional
# functionality for working with road networks. A road network is a directed graph, where nodes represent intersections and
# edges represent roads. Nodes must have a ``'pos'`` attribute, which is a tuple of (x, y) coordinates. Edges must have a
# ``'weight'`` attribute, which is a float representing the weight of the edges (e.g. length).
import pickle
from stonesoup.types.graph import RoadNetwork

# Load the road network dictionary from the pickle file
with open('./road_net.pkl', 'rb') as f:
    net_dict = pickle.load(f)

# Convert the dictionary to a RoadNetwork object
road_net = RoadNetwork.from_dict(net_dict)

# Get the number of nodes and edges in the graph
num_nodes = road_net.number_of_nodes()
num_edges = road_net.number_of_edges()

# %%
# Define parameters
# -----------------
# We define the parameters for the simulation, including the source and destination nodes,
# the speed of the target, and the number of particles to use in the particle filter.

# Parameters
num_destinations = 100              # Number of possible destinations
num_particles = 1000                # Number of particles in the particle filter
source = 357                        # Source node index
destination = 116                   # Destination node index
speed = 0.1                         # Speed of the target
zoom = 0.2                          # Zoom level for the plot
noise_covar = np.eye(2)*0.001      # Measurement noise covariance

# %%
# Simulation
# ----------
#
# Sample confuser destinations
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# We sample a set of confuser destinations from the road network. The destinations are sampled uniformly from the
# nodes in the graph, excluding the source and destination nodes.
confuser_destinations = np.random.choice(list(set(road_net.nodes()) - {source, destination}),
                                         num_destinations-1, replace=False)
all_destinations = np.append(destination, confuser_destinations)

# %%
# Simulate ground truth
# ~~~~~~~~~~~~~~~~~~~~~
# We simulate the ground truth path of the target moving along the road network. We first define the following
# helper function to generate the ground truth path, which includes the node and edge indices of the path.
from stonesoup.types.array import StateVector
from stonesoup.types.groundtruth import GroundTruthState, GroundTruthPath
from stonesoup.functions.graph import get_xy_from_range_edge, calc_edge_len, normalise_re

def simulate(G: RoadNetwork, source: int, destination: int, speed: float, track_id: int = None,
             timestamp_init: datetime = None, interval: timedelta = timedelta(seconds=1)):
    """ Simulate a moving target along the network

    Parameters
    ----------
    G: :class:`~.RoadNetwork`
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
    if timestamp_init is None:
        timestamp_init = datetime.now()

    # Compute shortest path to destination
    gnd_route_tmp = G.shortest_path(source, destination, path_type='both')
    gnd_route_n = gnd_route_tmp['node'][(source, destination)]   # Node indices
    gnd_route_e = gnd_route_tmp['edge'][(source, destination)]   # Edge indices

    # Get the node positions
    node_pos = nx.get_node_attributes(G, 'pos')

    # Initialize the Ground-Truth path
    init_node = gnd_route_n[0]
    sv = StateVector(np.array(node_pos[init_node]))
    init_state = GroundTruthState(sv, timestamp=timestamp_init)
    gnd_path = GroundTruthPath([init_state], id=track_id)

    # Compute the edge lengths
    edge_lengths = [calc_edge_len(edge_ind, G) for edge_ind in gnd_route_e]
    # Compute the total distance of the path
    total_distance = np.sum(edge_lengths)
    # Compute the total time of the path
    total_time = total_distance / speed
    # Compute the total number of intervals
    num_intervals = int(np.ceil(total_time / interval.total_seconds()))

    # Initialize the current range and edge index
    cur_r = 0
    cur_edge = gnd_route_e[0]
    for i in range(1, num_intervals+1):
        # Compute the current time and position
        cur_time = timestamp_init + i * interval
        # Compute the new range
        cur_r += speed * interval.total_seconds()
        # Normalize the range and edge index
        cur_r, cur_edge = normalise_re(cur_r, cur_edge, gnd_route_e, G)
        # Compute the new position
        pos_current = get_xy_from_range_edge(cur_r, cur_edge, G)
        # Create the state vector
        sv = StateVector(pos_current)
        # Create the ground truth state
        state = GroundTruthState(sv, timestamp=cur_time)
        # Append the state to the ground truth path
        gnd_path.append(state)

    return gnd_path, gnd_route_n, gnd_route_e

#%%
# Now, we call the above function to generate the ground truth path of the target moving from the source to the destination.

# Simulate ground-truth
timestamp_init = datetime.now()
gnd_path, gnd_route_n, gnd_route_e = simulate(road_net, source, destination, speed,
                                              timestamp_init=timestamp_init,
                                              interval=timedelta(seconds=1))

# %%
# Simulate detections
# ~~~~~~~~~~~~~~~~~~~
# We simulate the detections of the target along the ground truth path. The detections are generated by adding Gaussian noise
# to the ground truth state vector, and are of the form:
#
# .. math::
#     y_k = \left[\mathrm{x}_k, \mathrm{y}_k \right]^T + \mathcal{N}(0, R)
#
# where :math:`y_k` is the measurement, :math:`\mathrm{x}_k` and :math:`\mathrm{y}_k` are the x and y coordinates of the
# target's position, and :math:`R` is the measurement noise covariance matrix.
from stonesoup.types.detection import Detection

# Simulate detections
scans = []
for gnd_state in gnd_path:
    gnd_sv = gnd_state.state_vector
    det_sv = gnd_sv + np.atleast_2d(mvn.rvs(cov=noise_covar)).T
    timestamp = gnd_state.timestamp
    metadata = {"gnd_id": gnd_path.id}
    detection = Detection(state_vector=det_sv, timestamp=timestamp, metadata=metadata)
    scans.append((timestamp, {detection}))

# %%
# (Optional) Pre-compute shortest paths
# -------------------------------------
# We pre-compute the shortest paths from the source to a set of possible destinations. This step is done for two reasons:
#
#   1. To calculate the shortest paths for visualisation purposes, and
#   2. To speed up the estimation process in the particle filter.
#
# The :class:`~stonesoup.types.graph.RoadNetwork` class provides a method to compute the shortest path, and implements
# logic to cache the results for future use. Therefore, by calling the method once, we can trigger the caching
# mechanism in the road network object, so that the results are readily available for future use.
#
# It is also worth noting that, by default, the shortest path is computed using the ``'weight'`` attribute of the edges,
# rather than their length (i.e. the distance between the nodes). This is because the ``'weight'`` attribute can be used to
# represent other factors, such as travel time or cost.
#
# For more information, see the :py:meth:`~stonesoup.types.graph.RoadNetwork.shortest_path` method.

# Pre-compute short_paths
short_paths = road_net.shortest_path(source, all_destinations, path_type='both')

# %%
# Define Tracking components
# --------------------------
# Next, we define the tracking components, including the transition model, measurement model, predictor, and updater.
#
# Transition model
# ~~~~~~~~~~~~~~~~~
# The transition model is based on the :class:`~stonesoup.models.transition.graph.OptimalPathToDestinationTransitionModel` class,
# which uses the road network structure to model the target's movement. A :class:`~stonesoup.models.transition.linear.ConstantVelocity`
# model is used to model the target's motion as near constant velocity. Additionally, the transition model is parameterized with the
# set of possible destinations to which the target can move. This is an optional step, done to speed up the optimization process.
from stonesoup.models.transition.linear import ConstantVelocity
from stonesoup.models.transition.graph import OptimalPathToDestinationTransitionModel
# Transition model
cv_model = ConstantVelocity(0.0001)
transition_model = OptimalPathToDestinationTransitionModel(cv_model, road_net,
                                                           possible_destinations=all_destinations)

# %%
# Prior State
# ~~~~~~~~~~~
# Since a constant velocity model is used, the state vector :math:`x_k` is defined as:
#
# .. math::
#     x_k = \left[r_k, \dot{r}_k, e_k, d_k, s_k\right]^T
#
# where :math:`r_k` is the range travelled along the edge :math:`e_k`, :math:`\dot{r}_k` is the speed of the target,
# and :math:`d_k` and :math:`s_k` are the destination and source node indices, respectively.
#
# The state vector is initialized with the following values:
#
#   - :math:`r_k = 0`
#   - :math:`\dot{r}_k = \mathcal{N}(0, speed)`
#   - :math:`e_k = e_0`
#   - :math:`d_k = \mathcal{U}(destinations)`
#   - :math:`s_k = n_0`
from stonesoup.types.array import StateVectors
from stonesoup.types.update import ParticleStateUpdate
from stonesoup.types.hypothesis import SingleHypothesis

# Prior
timestamp_init = scans[0][0]
prior_sv = StateVectors([
    np.zeros((num_particles,)),                             # r
    mvn.rvs(0, speed, (num_particles,)),                    # speed
    np.full((num_particles,), gnd_route_e[0]),              # edge
    np.random.choice(all_destinations, (num_particles,)),   # destination
    np.full((num_particles,), gnd_route_n[0])               # source
])
prior_state = ParticleStateUpdate(state_vector=prior_sv,
                                  log_weight=np.full((num_particles,), np.log(1.0/num_particles)),
                                  hypothesis=SingleHypothesis(None, next(iter(scans[0][1]))),
                                  timestamp=timestamp_init)

# %%
# Measurement model
# ~~~~~~~~~~~~~~~~~
# The measurement model is based on the :class:`~stonesoup.models.measurement.graph.OptimalPathToDestinationMeasurementModel` class,
# that projects the target's position on the road network to a 2D position.
#
# The measurement model also implements a `use_indicator` parameter, which is used to parameterize the likelihood function.
# The likelihood function is defined in either of two ways, depending on the value of :attr:`use_indicator`:
#
#     - If :attr:`use_indicator` is `False`, then the likelihood function is defined as:
#
#     .. math::
#         p(y_k|x_k) = \mathcal{N}(y_k; h(x_k), R)
#
#     - If :attr:`use_indicator` is `True`, then the likelihood function is defined as:
#
#     .. math::
#
#         p(y_k|x_k) = \begin{cases}\mathcal{N}(y_k; h(x_k), R),
#                                             & \text{if } e_k \in \text{shortest_path}(s_k, d_k) \\
#                     0 & \text{otherwise}\end{cases}
#
#     where :math:`\text{shortest_path}(s_k, d_k)` is the shortest path between the source node
#     :math:`s_k` and destination node :math:`d_k` on the road network.
#
# This indicator function is used to inform the destination estimation process, by setting the likelihood
# to zero for particles that do not lie on the shortest path between the source and destination nodes. In this example,
# we set :attr:`use_indicator` to `True`.

from stonesoup.models.measurement.graph import OptimalPathToDestinationMeasurementModel

# Measurement model
mapping = [0, 1]
measurement_model = OptimalPathToDestinationMeasurementModel(
    ndim_state=5, mapping=mapping, noise_covar=noise_covar, graph=road_net, use_indicator=True)

# %%
# Predictor and Updater
# ~~~~~~~~~~~~~~~~~~~~~~~~
# We use the :class:`~stonesoup.predictor.particle.ParticlePredictor` and :class:`~stonesoup.updater.particle.ParticleUpdater`
# classes to implement the particle filter.
from stonesoup.predictor.particle import ParticlePredictor
from stonesoup.resampler.particle import SystematicResampler
from stonesoup.updater.particle import ParticleUpdater
# Predictor
predictor = ParticlePredictor(transition_model)
# Updater
resampler = SystematicResampler()
updater = ParticleUpdater(measurement_model, resampler=resampler)


# %%
# Estimation
# ----------
# We now run the particle filter to estimate the target's state and destination.
from stonesoup.types.track import Track

# Initiate track
track = Track([prior_state], id=gnd_path.id)

# Iterate over all scans (skipping the first one)
for timestamp, detections in scans[1:]:
    # Predict the next state
    prediction = predictor.predict(track.state, timestamp=timestamp)
    # Create a hypothesis object
    detection = next(iter(detections))
    hypothesis = SingleHypothesis(prediction, detection)
    # Update based on the hypothesis
    posterior = updater.update(hypothesis)
    # Append the posterior state to the track
    track.append(posterior)


# %%
# Plotting
# ---------
# Now that we have the estimated state distribution, we can proceed to plot the results.
#
# Helper functions
# ~~~~~~~~~~~~~~~~~
# First, we define a few helper functions for plotting.
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle

def plot_network(G, ax, node_size=0.1, width=0.5):
    """
    Plot the road network

    Parameters
    ----------
    G: :class:`~.RoadNetwork`
        The road network
    ax: :class:`matplotlib.axes.Axes`
        The axes to plot on
    node_size: :class:`float`
        The size of the nodes
    width: :class:`float`
        The width of the edges
    with_labels: :class:`bool`
        Whether to show the node labels
    """
    # Get node positions
    pos = nx.get_node_attributes(G, 'pos')

    nx.draw_networkx(G, pos, arrows=False, ax=ax, node_size=node_size, width=width,
                     with_labels=False)


def highlight_nodes(G, ax, nodes, node_size=0.1, node_color='m', node_shape='s', label=None):
    """
    Highlight nodes in the graph

    Parameters
    ----------
    G: :class:`~.RoadNetwork`
        The road network
    ax: :class:`matplotlib.axes.Axes`
        The axes to plot on
    nodes: :class:`Iterable`
        The list of node indices to highlight
    node_size: :class:`float`
        The size of the nodes
    node_color: :class:`str`
        The color of the nodes
    node_shape: :class:`str`
        The shape of the nodes
    label: :class:`str`
        The label for the nodes
    """
    # Get node positions
    pos = nx.get_node_attributes(G, 'pos')

    return nx.draw_networkx_nodes(G, pos, nodelist=nodes, ax=ax, node_size=node_size,
                                  node_color=node_color, node_shape=node_shape, label=label)


def highlight_edges(G, ax, edges_idx, width=2.0, edge_color='m', style='solid', arrows=False,
                    label=None):
    """
    Highlight edges in the graph

    Parameters
    ----------
    G: :class:`~.RoadNetwork`
        The road network
    ax: :class:`matplotlib.axes.Axes`
        The axes to plot on
    edges_idx: :class:`Iterable`
        The list of edge indices to highlight
    width: :class:`float`
        The width of the edges
    edge_color: :class:`str`
        The color of the edges
    style: :class:`str`
        The style of the edges
    arrows: :class:`bool`
        Whether to show arrows on the edges
    label: :class:`str`
        The label for the edges
    """
    edges = G.edge_list[edges_idx]
    pos = nx.get_node_attributes(G, 'pos')

    return nx.draw_networkx_edges(G, pos, edgelist=edges, ax=ax, width=width, style=style,
                                  edge_color=edge_color, arrows=arrows, label=label)


def calc_cov_ellipse(cov, pos, nstd=2, **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the
    ellipse patch artist.
    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.
    Returns
    -------
        A matplotlib ellipse artist
    """

    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    # Make sure width and height are not too small or NaN
    width = 0.2 if np.isnan(width) or width < 0.2 else width
    height = 0.2 if np.isnan(height) or height < 0.2 else height

    return Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)


def init():
    global est_xy
    est_xy = np.array([[], []])


def update(i):
    global dyn_arts, est_xy
    # Clear previous artists
    for art in dyn_arts:
        art.remove()
    dyn_arts = []


    # Compute statistics
    state = track.states[i]
    sv = state.state_vector
    detection = state.hypothesis.measurement
    v_dest, vd_counts = np.unique(sv[3,:], return_counts=True)
    xy = get_xy_from_range_edge(sv[0, :], sv[2, :], road_net)
    mu_pos = np.average(xy, axis=1, weights=np.exp(state.log_weight))
    est_xy = np.append(est_xy, np.atleast_2d(mu_pos).T, axis=1)

    # Compute estimated destination position
    est_dest_pos = np.array([list(pos[node]) for node in sv[3, :]]).T
    mu_dest = np.average(est_dest_pos, axis=1, weights=state.weight)
    cov_dest = np.cov(est_dest_pos, ddof=0, aweights=state.weight)

    # Plot particles, trajectory, and detections
    for ax in [ax1, ax2]:
        dyn_arts.append(ax.plot(xy[0, :], xy[1, :], '.r')[0])
        dyn_arts.append(ax.plot(est_xy[0, :], est_xy[1, :], '-b')[0])
        dyn_arts.append(ax.plot(*detection.state_vector.ravel(), 'xc')[0])

    # Plot the destination ellipse
    ellipse = calc_cov_ellipse(cov_dest, mu_dest, nstd=3, fill=None, edgecolor='r',
                               label="Destination uncertainty")
    ax1.add_artist(ellipse)
    dyn_arts.append(ellipse)

    # Plot and set zoom
    zoom_box = Rectangle(mu_pos-zoom, 2*zoom, 2*zoom, fill=None, ec='k')
    ax1.add_patch(zoom_box)
    dyn_arts.append(zoom_box)
    ax2.set_xlim((mu_pos[0] - zoom, mu_pos[0] + zoom))
    ax2.set_ylim((mu_pos[1] - zoom, mu_pos[1] + zoom))

    # Plot the destination distribution
    ax3.cla()
    barlist = ax3.bar([str(int(d)) for d in v_dest], vd_counts / np.sum(vd_counts), color='m')
    try:
        # Highlight the destination in the bar chart, if it exists
        idx = v_dest.tolist().index(destination)
        barlist[idx].set_color('r')
    except ValueError:
        pass
    ax3.set_title('Destination Distribution', fontsize='small')
    plt.xticks(rotation=90, fontsize=5)

    # Update ax1 legend
    ax1.legend(loc='upper right', bbox_to_anchor=(1., 0.5),fontsize='x-small')

# %%
# Animation
# ~~~~~~~~~
# We create a figure with 3 subplots:
#
#   - The first subplot shows the global view of the road network, with the true path and confuser paths highlighted.
#   - The second subplot shows a zoomed-in view of the target's trajectory, with the particles and measurements plotted.
#   - The third subplot shows the distribution of the estimated destinations.
#
# From the animation, we can see that the particles converge towards the true path of the target, and the estimated destination
# distribution becomes more concentrated around the true destination as time progresses.
from matplotlib import animation

# Create a figure and axes
fig = plt.figure(figsize=(9, 5))
# Create a gridspec layout for the subplots
gs = fig.add_gridspec(2, 3)
ax1 = fig.add_subplot(gs[:, :2])
ax2 = fig.add_subplot(gs[0, 2])
ax3 = fig.add_subplot(gs[1, 2])
# Set the aspect ratio of the axes
fig.subplots_adjust(
    top=0.95,
    bottom=0.05,
    left=0.01,
    right=0.99,
    hspace=0.2,
    wspace=0.2
)

# Plot the road network and set up the axes, legend, and title
for ax in [ax1, ax2]:
    plot_network(road_net, ax)
    for key, value in short_paths['edge'].items():
        highlight_edges(road_net, ax, value, edge_color='y')
    highlight_edges(road_net, ax, gnd_route_e, edge_color='g')
    highlight_nodes(road_net, ax, all_destinations, node_color='m', node_size=10)
    highlight_nodes(road_net, ax, [destination], node_color='r', node_size=10)
ax1.plot([], [], '-g', label='True path')
ax1.plot([], [], 'sr', label='True destination')
ax1.plot([], [], '-y', label='Confuser paths')
ax1.plot([], [], 'sm', label='Confuser destinations')
ax1.set_title('Global view', fontsize='small')
ax2.plot([], [], 'r.', label='Particles')
ax2.plot([], [], 'b-', label='Trajectory')
ax2.plot([], [], 'cx', label='Measurements')
ax2.set_title('Zoomed view', fontsize='small')
ax2.legend(loc='upper right', fontsize='x-small')

# Plotting variables
dyn_arts = []                           # Container for dynamic artists
pos = nx.get_node_attributes(road_net, 'pos')  # Get node positions
est_xy = np.array([[], []])             # Estimated trajectory cache

ani = animation.FuncAnimation(fig, update, frames=len(track.states), init_func=init,
                              blit=False, repeat=False)
plt.show()

# %%
# References
# ----------
# .. [#] L. Vladimirov and S. Maskell, "A SMC Sampler for Joint Tracking and Destination
#        Estimation from Noisy Data," 2020 IEEE 23rd International Conference on Information
#        Fusion (FUSION), Rustenburg, South Africa, 2020, pp. 1-8,
#        doi: 10.23919/FUSION45008.2020.9190463.
