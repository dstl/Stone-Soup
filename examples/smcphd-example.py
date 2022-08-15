from matplotlib import pyplot as plt
from stonesoup.resampler.particle import SystematicResampler
from stonesoup.types.array import StateVector, StateVectors
from stonesoup.types.numeric import Probability
from stonesoup.types.particle import Particle
from stonesoup.types.state import GaussianState, ParticleState
from stonesoup.custom.smcphd import SMCPHDFilter

from datetime import datetime
from datetime import timedelta
import numpy as np
from scipy.stats import uniform, multivariate_normal

from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
    ConstantVelocity
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.types.detection import TrueDetection
from stonesoup.types.detection import Clutter
from stonesoup.models.measurement.linear import LinearGaussian

np.random.seed(1991)

# Parameters
# ==========
start_time = datetime.now()     # Simulation start time
prob_detect = Probability(.9)   # 90% chance of detection.
prob_death = Probability(0.01)  # Probability of death
prob_birth = Probability(0.1)   # Probability of birth
birth_rate = 0.05               # Birth-rate (Mean number of new targets per scan)
clutter_rate = .01                # Clutter-rate (Mean number of clutter measurements per scan)
surveillance_region = [[-10, 30], [0, 30]]  # The surveillance region x=[-10, 30], y=[0, 30]
surveillance_area = (surveillance_region[0][1] - surveillance_region[0][0]) \
                    * (surveillance_region[1][1] - surveillance_region[1][0])
clutter_intensity = clutter_rate / surveillance_area  # Clutter intensity per unit volume/area
birth_density = GaussianState(StateVector(np.array([10., 0.0, 10., 0.0])),
                              np.diag([10. ** 2, 1. ** 2, 10. ** 2, 1. ** 2]))  # Birth density
birth_scheme = 'mixture'  # Birth scheme. Possible values are 'expansion' and 'mixture'
num_particles = 2 ** 12  # Number of particles used by the PHD filter
num_iter = 100  # Number of simulation steps
PLOT = True  # Set [True | False] to turn plotting [ON | OFF]

# Models
# ======
# Transition model
transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.05),
                                                          ConstantVelocity(0.05)])
# Measurement model
measurement_model = LinearGaussian(ndim_state=4,
                                   mapping=(0, 2),
                                   noise_covar=np.array([[0.02, 0],
                                                         [0, 0.02]]))

# Simulate Groundtruth
# ====================
gnd_transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.),
                                                              ConstantVelocity(0.)])
truths = set()
truth = GroundTruthPath([GroundTruthState([0, 0.2, 0, 0.2], timestamp=start_time)])
for k in range(1, num_iter + 1):
    truth.append(GroundTruthState(
        gnd_transition_model.function(truth[k - 1], noise=True, time_interval=timedelta(seconds=1)),
        timestamp=start_time + timedelta(seconds=k)))
truths.add(truth)

truth = GroundTruthPath([GroundTruthState([0, 0.2, 20, -0.2], timestamp=start_time)])
for k in range(1, num_iter + 1):
    truth.append(GroundTruthState(
        gnd_transition_model.function(truth[k - 1], noise=True, time_interval=timedelta(seconds=1)),
        timestamp=start_time + timedelta(seconds=k)))
truths.add(truth)

timestamps = []
for k in range(1, num_iter + 1):
    timestamps.append(start_time + timedelta(seconds=k))

# Plot ground truth.
if PLOT:
    from stonesoup.plotter import Plotter

    plotter = Plotter()
    plotter.ax.set_ylim(0, 25)
    plotter.plot_ground_truths(truths, [0, 2])

# Simulate measurements
# =====================
scans = []

for k in range(num_iter):
    measurement_set = set()

    # True detections
    for truth in truths:
        # Generate actual detection from the state with a 10% chance that no detection is received.
        if np.random.rand() <= prob_detect:
            measurement = measurement_model.function(truth[k], noise=True)
            measurement_set.add(TrueDetection(state_vector=measurement,
                                              groundtruth_path=truth,
                                              timestamp=truth[k].timestamp,
                                              measurement_model=measurement_model))

        # Generate clutter at this time-step
        truth_x = truth[k].state_vector[0]
        truth_y = truth[k].state_vector[2]

    # Clutter detections
    for _ in range(np.random.poisson(clutter_rate)):
        x = uniform.rvs(-10, 30)
        y = uniform.rvs(0, 25)
        measurement_set.add(Clutter(np.array([[x], [y]]), timestamp=truth[k].timestamp,
                                    measurement_model=measurement_model))
    scans.append((timestamps[k], measurement_set))

# Initialise PHD Filter
# =====================
resampler = SystematicResampler()
phd_filter = SMCPHDFilter(birth_density=birth_density, transition_model=transition_model,
                          measurement_model=None, prob_detect=prob_detect,
                          prob_death=prob_death, prob_birth=prob_birth,
                          birth_rate=birth_rate, clutter_intensity=clutter_intensity + 0.001,
                          num_samples=num_particles, resampler=resampler,
                          birth_scheme=birth_scheme)

# Estimate
# ========

# Sample prior state from birth density
state_vector = StateVectors(multivariate_normal.rvs(birth_density.state_vector.ravel(),
                                                    birth_density.covar,
                                                    size=num_particles).T)
weight = np.ones((num_particles,)) * Probability(1 / num_particles)
state = ParticleState(state_vector=state_vector, weight=weight, timestamp=start_time)

# Plot the prior
if PLOT:
    fig1 = plt.figure(figsize=(13, 7))
    ax1 = plt.gca()
    ax1.plot(state.state_vector[0, :], state.state_vector[2, :], 'r.')

# Main tracking loop
for k, (timestamp, detections) in enumerate(scans):

    new_state = phd_filter.iterate(state, detections, timestamp)
    state = new_state

    print('Num targets: ', np.sum(state.weight))

    # Plot resulting density
    if PLOT:
        ax1.cla()
        for i, truth in enumerate(truths):
            data = np.array([s.state_vector for s in truth[:k + 1]])
            ax1.plot(data[:, 0], data[:, 2], '--', label=f'Groundtruth Track {i+1}')
        if len(detections):
            det_data = np.array([det.state_vector for det in detections])
            ax1.plot(det_data[:, 0], det_data[:, 1], '*g', label='Detections')
        ax1.plot(new_state.state_vector[0, :], new_state.state_vector[2, :],
                 'r.', label='Particles')
        plt.axis([*surveillance_region[0], *surveillance_region[1]])
        plt.legend(loc='center right')
        plt.pause(0.01)
