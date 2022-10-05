# Standard libraries
from datetime import datetime, timedelta

import numpy as np
from matplotlib import pyplot as plt

# Now the stoneSoup components
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
    ConstantVelocity
from stonesoup.platform.base import MovingPlatform, FixedPlatform
from stonesoup.plotter import Plotter
from stonesoup.sensor.radar.radar import RadarElevationBearingRangeStochasticDetectability
from stonesoup.types.array import StateVector
from stonesoup.types.state import State

start_time = datetime(2022, 1, 1, 0, 0, 0)


starting_distance = 15000
closing_speed = 100

n_time_steps = int(starting_distance/closing_speed)-10

min_range = 3000
max_range = 12000

radar = RadarElevationBearingRangeStochasticDetectability(
    ndim_state=6,
    position_mapping=[0, 2, 4],
    noise_covar=np.diag([0.01, 0.01, 1]),
    min_range=min_range,
    max_range=max_range
)

sensor_platform = FixedPlatform(
    states=State(state_vector=StateVector([0, 0,
                                           0, 0,
                                           0, 0]),
                 timestamp=start_time),
    sensors=[radar],
    position_mapping=np.array([0, 2, 4]))

transition_model = CombinedLinearGaussianTransitionModel(
    [ConstantVelocity(1.0),
     ConstantVelocity(1.0),
     ConstantVelocity(1.0)])

target = MovingPlatform(
    states=State(state_vector=StateVector([0, 0,
                                           starting_distance, -closing_speed,
                                           0, 0]),
                 timestamp=start_time),
    transition_model=transition_model,
    position_mapping=[0, 2, 4],
)

detectability_model = radar.detectability_model

times = [start_time + timedelta(seconds=x) for x in range(n_time_steps)]

ranges_to_target = []
probability_of_detect_model = []
probability_of_detect_test = []

n_measures_per_timestep = 10
all_detections = []

for time in times:
    if time != start_time:
        target.move(time)

    range_to_target, _, _ = sensor_platform.range_and_angles_to_other(target)
    ranges_to_target.append(range_to_target)
    probability_of_detect_model.append(detectability_model.probability_at_value(range_to_target))

    detections = list()
    for _ in range(n_measures_per_timestep):
        detections += radar.measure({target})

    probability_of_detect_test.append(len(detections) / n_measures_per_timestep)

    all_detections += radar.measure({target})


plt.plot(ranges_to_target, probability_of_detect_model, label="Theory")
plt.plot(ranges_to_target, probability_of_detect_test, "x", label="Test")
plt.plot([min_range, min_range], [0, 1],  label="99% Chance of detection")
plt.plot([max_range, max_range], [0, 1],  label="1% Chance of detection")
plt.grid(which='both')
plt.xlabel("Range (m)")
plt.ylabel("Probability of Detection")
plt.legend()
plt.title("Probability of Detection")

plotter = Plotter()
plotter.plot_measurements(all_detections, mapping=[0, 2])
plotter.plot_ground_truths(target, mapping=[0, 2])
plotter.plot_sensors(radar)
plt.title("Example Detections from an Approaching Target")

plt.figure()
detectability_model.plot()
plt.title("In-built Probability of Detection Plot")

plt.show()
