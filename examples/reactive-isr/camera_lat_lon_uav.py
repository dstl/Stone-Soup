from datetime import datetime, timedelta

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from stonesoup.custom.sensor import PanTiltCamera, ChangePanTiltAction, PanTiltCameraLatLong, \
    PanTiltUAVCamera
from stonesoup.functions import pol2cart
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
rotation_offset = StateVector([Angle(0), Angle(-np.pi/32), Angle(0)])  # Camera rotation offset
camera = PanTiltUAVCamera(ndim_state=6, mapping=[0, 2, 4], noise_covar=np.diag([0.001, 0.001, 0.001]),
                          fov_angle=np.radians(10), rpm=np.array([10, 10]), rotation_offset=rotation_offset)
platform = FixedPlatform(position_mapping=(0, 2, 4), orientation=StateVector([0, -np.pi/2, 0]),
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

generator = next(g for g in camera.actions(start_time + timedelta(seconds=10), start_timestamp=start_time))
action = ChangePanTiltAction(rotation_end_time=start_time + timedelta(seconds=15),
                             generator=generator,
                             end_time=start_time + timedelta(seconds=15),
                             target_value=StateVector([Angle(0),
                                                       Angle(0),
                                                       Angle(0)]),
                             increasing_angle=[True, False])

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
    fov_range_tilt = (camera.rotation_offset[1]-camera.fov_angle/2, camera.rotation_offset[1], camera.rotation_offset[1]+camera.fov_angle/2)
    fov_range_pan = (camera.rotation_offset[2]-camera.fov_angle/2, camera.rotation_offset[2], camera.rotation_offset[2]+camera.fov_angle/2)

    altitude = camera.position[2]
    x_min = altitude * np.tan(fov_range_tilt[0]) + camera.position[0]
    x_max = altitude * np.tan(fov_range_tilt[2]) + camera.position[0]
    y_min = altitude * np.tan(fov_range_pan[0]) + camera.position[1]
    y_max = altitude * np.tan(fov_range_pan[2]) + camera.position[1]

    ax.add_patch(Rectangle((x_min, y_min), x_max-x_min, y_max-y_min, facecolor='none', edgecolor='r'))
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
    a=2

# # Plot results
# # ============
# for k, scan in enumerate(scans):
#
# a = 2