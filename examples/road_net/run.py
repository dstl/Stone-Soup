import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from matplotlib.patches import Circle, Ellipse
from scipy.io import loadmat
from scipy.stats import multivariate_normal as mvn
import cProfile as profile


from simulation import simulate
from stonesoup.functions.graph import get_xy_from_range_edge
from stonesoup.models.transition.linear import ConstantVelocity
from stonesoup.models.transition.graph import ShortestPathToDestinationTransitionModel
from stonesoup.models.measurement.graph import ShortestPathToDestinationMeasurementModel
from stonesoup.predictor.particle import ParticlePredictor
from stonesoup.resampler.particle import SystematicResampler
from stonesoup.types.array import StateVector, StateVectors
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.state import ParticleState
from stonesoup.types.detection import Detection
from stonesoup.types.particle import Particle
from stonesoup.types.graph import RoadNetwork

# Load Graph
from stonesoup.types.track import Track
from stonesoup.types.update import ParticleStateUpdate
from stonesoup.updater.particle import ParticleUpdater


def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
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

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    ax.add_artist(ellip)
    return ellip


def load_graph_dict(path):
    wp = loadmat(path)
    s = wp['S'][0, 0]
    S = dict()
    num_edges = s['Edges']['EndNodes'][0, 0].shape[0]
    num_nodes = s['Nodes']['Longitude'][0, 0].shape[0]

    S['edges'] = dict()
    for i in range(num_edges):
        endnodes = tuple(s['Edges']['EndNodes'][0, 0][i, :])
        attr = dict()
        attr['weight'] = s['Edges']['Weight'][0, 0][i, 0]
        S['edges'][endnodes] = attr

    S['nodes'] = dict()
    for i in range(num_nodes):
        attr = dict()
        attr['pos'] = (s['Nodes']['Longitude'][0, 0][i, 0], s['Nodes']['Latitude'][0, 0][i, 0])
        S['nodes'][i+1] = attr
    return S

seed = 818 # np.random.randint(0, 1000) # 706, 959, 547
print(seed)
np.random.seed(seed)

path = r'.\data\minn_2.mat'

net_dict = load_graph_dict(path)
G = RoadNetwork.from_dict(net_dict)

num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()

# Global vars
USE_NEW_PF = True
num_destinations = 100
num_particles = 1000
source = 520
destination = 116
speed = 0.1
zoom = 0.2
LOAD = False

# a = G.shortest_path(source)

if LOAD:
    gnd_path, gnd_route_n, gnd_route_e = \
        pickle.load(open(f'./data/single_track_{source}_{destination}.pickle', 'rb'))
else:
    # Simulate ground-truth
    gnd_path, gnd_route_n, gnd_route_e = simulate(G, source, destination, speed)
    pickle.dump([gnd_path, gnd_route_n, gnd_route_e],
                open(f'./data/single_track_{source}_{destination}.pickle', 'wb'))

# Pre-compute short_paths
feed = [destination]
feed_tmp = set([i+1 for i in range(num_nodes)])-set(feed)
destinations = feed + list(np.random.choice(list(feed_tmp),(num_destinations-len(feed),),False))
short_paths_n = dict()
short_paths_e = dict()
short_paths = G.shortest_path(source, destinations, path_type='both')


# Transition model
cv_model = ConstantVelocity(0.01)
transition_model = ShortestPathToDestinationTransitionModel(
    cv_model, G, possible_destinations=destinations)

# Measurement model
mapping = [0,1]
R = np.eye(2)*0.0002
measurement_model = ShortestPathToDestinationMeasurementModel(
    ndim_state=5, mapping=mapping, noise_covar=R, graph=G)

# Simulate detections
scans = []
for gnd_state in gnd_path:
    gnd_sv = gnd_state.state_vector
    det_sv = gnd_sv + measurement_model.rvs()
    timestamp = gnd_state.timestamp
    metadata = {"gnd_id": gnd_path.id}
    detection = Detection(state_vector=det_sv, timestamp=timestamp, metadata=metadata)
    scans.append((timestamp, set([detection])))

# initiator = DestinationBasedInitiator(measurement_model, num_particles, speed, G)
# tracks = initiator.initiate(scans[0][1])
#
# track =

# Prior
timestamp_init = scans[0][0]
prior_sv = StateVectors([
    np.zeros((num_particles,)),                         # r
    mvn.rvs(0, speed, (num_particles,)),                # speed
    np.full((num_particles,), gnd_route_e[0]),          # edge
    np.random.choice(destinations, (num_particles,)),   # destination
    np.full((num_particles,), gnd_route_n[0])           # source
])

# for i, sv in enumerate(zip(prior_r, prior_speed, prior_e, prior_destinations, prior_source)):
#     prior_particle_sv[:, i] = np.array(sv)
prior_state = ParticleStateUpdate(state_vector=prior_sv,
                                  log_weight=np.full((num_particles,), np.log(1.0/num_particles)),
                                  hypothesis=SingleHypothesis(None, next(d for d in scans[0][1])),
                                  timestamp=timestamp_init)

predictor = ParticlePredictor(transition_model)
# Updater
resampler = SystematicResampler()
updater = ParticleUpdater(measurement_model, resampler=resampler)
# Initiate track
track = Track([prior_state], id=gnd_path.id)


pos = nx.get_node_attributes(G, 'pos')

fig = plt.figure(figsize=(17, 10))
gs = fig.add_gridspec(2, 3)
ax1 = fig.add_subplot(gs[:, :2])
ax2 = fig.add_subplot(gs[0, 2])
ax3 = fig.add_subplot(gs[1, 2])
G.plot_network(ax1)
G.plot_network(ax2)
for key, value in short_paths['edge'].items():
    G.highlight_edges(ax1, value, edge_color='y')
    G.highlight_edges(ax2, value, edge_color='y')
G.highlight_edges(ax1, gnd_route_e, edge_color='g')
G.highlight_edges(ax2, gnd_route_e, edge_color='g')
G.highlight_nodes(ax1, destinations, node_color='m', node_size=10)
G.highlight_nodes(ax2, destinations, node_color='m', node_size=10)
G.highlight_nodes(ax1, [destination], node_color='r', node_size=10)
G.highlight_nodes(ax2, [destination], node_color='r', node_size=10)
ax1.plot([], [], '-g', label='True path')
ax1.plot([], [], 'sr', label='True destination')
ax1.plot([], [], '-y', label='Confuser paths')
ax1.plot([], [], 'sm', label='Confuser destinations')
ax1.legend(loc='upper right')
ax1.set_title('Global view')
ax2.plot([], [], 'r.', label='Particles')
ax2.plot([], [], 'b-', label='Trajectory')
ax2.plot([], [], 'cx', label='Measurements')
ax2.set_title('Zoomed view')
ax2.legend(loc='upper right')
arts1 = []
arts2 = []

# pr = profile.Profile()
# pr.disable()
est = np.array([[], []])
est_xy = np.array([[], []])
for timestamp, detections in scans:
    print(timestamp)
    detection = list(detections)[0]
    # pr.enable()
    # Run PF
    prediction = predictor.predict(track.state, timestamp=timestamp)
    hypothesis = SingleHypothesis(prediction, detection)
    posterior = updater.update(hypothesis)
    track.append(posterior)

    # pr.disable()

    # Compute statistics
    data = posterior.state_vector

    # Compute counts of destinations and current positions
    v_dest, vd_counts = np.unique(data[3,:], return_counts=True)
    id = np.argmax(vd_counts)
    v_edges, ve_counts = np.unique(data[2,:], return_counts=True)
    ie = np.argmax(ve_counts)
    est = np.append(est, [[track.state.mean[0, 0]], [v_edges[ie]]], axis=1)
    print('Estimated edge: {} - Estimated destination: {}'.format(v_edges[ie], v_dest[id]))
    xy = get_xy_from_range_edge(data[0, :], data[2, :], G)

    est_dest_pos = np.array([list(pos[node]) for node in data[3, :]]).T
    mu = np.average(est_dest_pos, axis=1, weights=posterior.weight)
    cov = np.cov(est_dest_pos, ddof=0, aweights=posterior.weight)

    # Plot
    # plot_network(G, ax)
    for art in arts1:
        art.remove()
    for art in arts2:
        art.remove()
    arts1 = []
    arts2 = []

    ind1 = np.flatnonzero(v_dest == destination)
    arts1.append(ax1.plot(xy[0, :], xy[1, :], '.r')[0])
    arts2.append(ax2.plot(xy[0, :], xy[1, :], '.r')[0])
    est_xy = np.append(est_xy,
                       np.atleast_2d(np.average(xy, axis=1, weights=np.exp(posterior.log_weight))).T,
                       axis=1)
    # xy1 = get_xy_from_range_edge(est[0, :], est[1, :], G)
    arts1.append(ax1.plot(est_xy[0, :], est_xy[1, :], '-b')[0])
    arts2.append(ax2.plot(est_xy[0, :], est_xy[1, :], '-b')[0])
    detection_data = np.array([detection.state_vector for detection in detections])
    arts1.append(ax1.plot(detection_data[:, 0], detection_data[:, 1], 'xc', label="Detections")[0])
    arts2.append(ax2.plot(detection_data[:, 0], detection_data[:, 1], 'xc', label="Detections")[0])
    if np.trace(cov) > 1e-10:
        arts1.append(plot_cov_ellipse(cov, mu, ax=ax1, nstd=3, fill=None, edgecolor='r'))
    else:
        circ = Circle(mu, 0.2, fill=None, edgecolor='r')
        ax1.add_artist(circ)
        arts1.append(circ)
    ax3.cla()
    barlist = ax3.bar([str(int(d)) for d in v_dest], vd_counts / np.sum(vd_counts))
    # barlist = ax3.hist(data[3,:])
    try:
        idx = v_dest.tolist().index(destination)
        barlist[idx].set_color('m')
    except:
        pass
    ax3.set_title('Destination Distribution')
    plt.xticks(rotation=90, fontsize=5)

    mu = np.mean(xy, axis=1)
    arts1.append(ax1.plot([mu[0] - zoom, mu[0] + zoom, mu[0] + zoom, mu[0] - zoom, mu[0] - zoom],
                          [mu[1] - zoom, mu[1] - zoom, mu[1] + zoom, mu[1] + zoom, mu[1] - zoom],
                          '-k')[0])
    ax2.set_xlim((mu[0] - zoom, mu[0] + zoom))
    ax2.set_ylim((mu[1] - zoom, mu[1] + zoom))

    plt.pause(0.01)

    a=2
