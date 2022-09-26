from datetime import datetime, timedelta

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from stonesoup.custom.functions import get_camera_footprint
from stonesoup.custom.sensor.action.pan_tilt import ChangePanTiltAction
from stonesoup.custom.sensor.pan_tilt import PanTiltUAVCamera
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
    ConstantVelocity
from stonesoup.platform import FixedPlatform
from stonesoup.types.angle import Bearing, Elevation, Angle
from stonesoup.types.array import StateVector
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.types.state import State



# Parameters
# ==========
start_time = datetime.now()         # Simulation start time
num_iter = 100  # Number of simulation steps
rotation_offset = StateVector([Angle(0), Angle(-np.pi/2), Angle(0)])  # Camera rotation offset
pan_tilt = StateVector([Angle(0), Angle(-np.pi/32)])  # Camera pan and tilt
camera = PanTiltUAVCamera(ndim_state=6, mapping=[0, 2, 4], noise_covar=np.diag([0.001, 0.001, 0.001]),
                          fov_angle=np.radians(10), rotation_offset=rotation_offset, pan_tilt=pan_tilt)
platform = FixedPlatform(position_mapping=(0, 2, 4), orientation=StateVector([0, 0, 0]),
                         states=[State([10., 0., 10., 0., 100., 0], timestamp=start_time)],
                         sensors=[camera])



# Models
# ======
# Transition model
transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.01),
                                                          ConstantVelocity(0.01),
                                                          ConstantVelocity(0.0)])

# Simulate Groundtruth
# ====================
gnd_transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.),
                                                              ConstantVelocity(0.),
                                                              ConstantVelocity(0.)])
truths = set()
truth = GroundTruthPath([GroundTruthState([0, 0.2, 0, 0.2, 0, 0], timestamp=start_time)])
for k in range(1, num_iter + 1):
    truth.append(GroundTruthState(
        gnd_transition_model.function(truth[k - 1], noise=True, time_interval=timedelta(seconds=1)),
        timestamp=start_time + timedelta(seconds=k)))
truths.add(truth)

truth = GroundTruthPath([GroundTruthState([0, 0.2, 20, -0.2, 0, 0], timestamp=start_time)])
for k in range(1, num_iter + 1):
    truth.append(GroundTruthState(
        gnd_transition_model.function(truth[k - 1], noise=True, time_interval=timedelta(seconds=1)),
        timestamp=start_time + timedelta(seconds=k)))
truths.add(truth)

timestamps = []
for k in range(1, num_iter + 1):
    timestamps.append(start_time + timedelta(seconds=k))

# Simulate measurements
# =====================
scans = []

# Schedule an action to change the pan and tilt of the camera after 30 seconds
generator = next(g for g in camera.actions(start_time + timedelta(seconds=30)))
action = generator.action_from_value(StateVector([Angle(0), Angle(0)]))

camera.add_actions([action])
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(1, 1, 1)
for k in range(num_iter):
    timestamp = timestamps[k]
    camera.act(timestamp)
    truth_states = [truth[k] for truth in truths]
    measurement_set = camera.measure(truth_states, timestamp=timestamp)
    scan = (timestamp, measurement_set)
    scans.append(scan)
    ax.cla()
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_xlim(-10, 30)
    ax.set_ylim(-10, 30)
    ax.set_aspect('equal')

    # Fov ranges (min, center, max)
    xmin, xmax, ymin, ymax = get_camera_footprint(camera)

    ax.add_patch(Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, facecolor='none', edgecolor='r'))
    # x, y = pol2cart(100, camera.orientation[2] - camera.fov_angle / 2)
    # ax.plot([0, x], [0, y], 'r-', label="Camera FOV")
    # x, y = pol2cart(100, camera.orientation[2] + camera.fov_angle / 2)
    # ax.plot([0, x], [0, y], 'r-')
    for truth in truths:
        data = np.array([state.state_vector for state in truth[:k + 1]])
        ax.plot(data[:, 0], data[:, 2], '--', label="Ground truth")
    detections = scan[1]
    for detection in detections:
        # x, y = pol2cart(100, detection.state_vector[1] + camera.orientation[2])
        # ax.plot([0, x], [0, y], 'b-')
        ax.plot(detection.state_vector[0], detection.state_vector[1], 'bx')
    plt.pause(0.1)

# # Plot results
# # ============
# for k, scan in enumerate(scans):
#
# a = 2