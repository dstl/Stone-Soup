#!/usr/bin/env python
# coding: utf-8

"""
Creating an Actionable Sensor Example
=====================================
This example shows how one can create an actionable sensor with a property that can be modified
via action types.
"""

# %%
# Creating Action and Action Generator
# ------------------------------------
# We will create a simple actionable sensor that can 'look' at one sector out of several at any
# one time.
# To do so, we require an :class:`~.Action` and :class:`~.ActionGenerator`.
# The action contains the logic of how a particular sensor property is modified. In this instance
# we simply want to change the what sector the sensor is looking at, at a given time.
# When the sensor is called to `act`, it calls `act` on the scheduled actions of each of its
# properties that are :class:`~.ActionableProperty` types (the current implementation allows for
# one scheduled action per actionable property in a sensor).
#
# The action's `act` method should return the new value of the modifiable property for the sensor.
# The modfiable property for our sensor will be its `sector`. Therefore, our `act` code can be as
# simple as returning the new sector name.
#
# Actions also have an `end_time` attribute, indicating when the action is to be finished by.
# In our case, we will simply require that, before the action's end-time is reached, nothing
# happens. It is only when the action's end-time is reached that the sensor will switch sector.
#
# An :class:`~.ActionGenerator` is required for querying the potential actions of the sensor.
# If we want to know what the sensor can do in the next 5 minutes, we should call
# `sensor.actions(current_time + 5 minutes)`, which will return a set of
# :class:`~.ActionGenerator` types, one for each :class:`~.ActionableProperty`. These generators
# can be looped over to determine all possible actions up to 5 minutes in the future.
# For our sensor, we'll say that it can reach any possible sector in any time-frame.
# There is also a `default_action` for an action generator. In case it is deemed that the sensor
# need not action its property in any particular way, it will resort to the default action of the
# property's corresponding action generator. For our sensor, that will be an action to remain
# looking at the same sector that it currently is (this is also an action returned by the
# generator's `__iter__` method in this instance, but doesn't necessarily need to be).
#
# For sensor managers, it is useful to determine whether a particular action is possible, or that
# a particular value of an :class:`~.ActionableProperty` is possible, in a time-frame. The logic
# for checking this is contained in the action generator's `__contains__` method. I.E. if the
# action or value is "in" the action generator, it is possible to achieve it.
#
# We will also create a useful method `action_from_value`, so that we can get an action from the
# generator simply by passing a sector name.

sectors = ["A", "B", "C", "D"]

from stonesoup.sensor.action import Action, ActionGenerator
from typing import Sequence
from stonesoup.base import Property


class ChangeSectorAction(Action):
    """Simply changes the sector that the sensor is looking at when the action `end_time` is
    reached."""
    def act(self, current_time, timestamp, init_value):
        """Modify sector to target value if `timestamp` is after action end-time. Otherwise return
        same sector. Assume `current_time` is before action end-time"""
        if timestamp >= self.end_time:
            return self.target_value
        else:
            return init_value


class SectorActionsGenerator(ActionGenerator):
    """Return an action for each sector that the sensor can 'look' at by `end_time`."""
    owner: object = Property(doc="Sensor with `timestamp`, `sector` and `sectors` attributes")

    @property
    def sectors(self):
        return self.owner.sectors

    @property
    def default_action(self):
        """The default action is to remain 'looking' at the same sector."""
        return ChangeSectorAction(generator=self,
                                  end_time=self.end_time,
                                  target_value=self.current_value)

    def __contains__(self, item):
        """Can view any sector in any time-frame. So as long as `item` is one of the sectors,
        it is contained in generator."""
        if isinstance(item, ChangeSectorAction):
            item = item.target_value
        return item in self.sectors

    def __iter__(self):
        for sector in self.sectors:
            return ChangeSectorAction(generator=self,
                                      end_time=self.end_time,
                                      target_value=sector)

    def action_from_value(self, value):
        """Return a `ChangeSensorAction` for a specific sector."""
        if value not in self.sectors:
            raise ValueError("Can only generate action from valid sector")
        return ChangeSectorAction(generator=self,
                                  end_time=self.end_time,
                                  target_value=value)


# %%
# Creating Actionable Sensor
# --------------------------
# The sensor will use the :class:`~.ActionableProperty` descriptor for its `sector` property.
# This descriptor has a `generator_cls` attribute determining the action generator associated with
# it. I.E. the logic for the property's modification is encompassed in what generator class is
# passed to it.

from stonesoup.sensor.sensor import Sensor
from stonesoup.sensor.actionable import ActionableProperty
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.types.detection import TrueDetection


class SectorLookingSensor(Sensor):
    sector: int = ActionableProperty(doc="Sector that sensor is looking in",
                                     generator_cls=SectorActionsGenerator)
    sectors: Sequence = Property(doc="List of sectors the sensor can view.")

    @property
    def measurement_model(self):
        return LinearGaussian(ndim_state=4,
                              mapping=(0, 2),
                              noise_covar=np.eye(2))

    def measure(self, ground_truths, noise=None, **kwargs):
        """Note: `ground_truths` must be composed of GroundTruthPath types with metadata indicating
        the sector they belong to."""

        detections = set()
        for truth in ground_truths:
            true_sector = truth.metadata.get("sector")
            if true_sector == self.sector:
                measurement_vector = self.measurement_model.function(truth, noise=False, **kwargs)
                detection = TrueDetection(measurement_vector,
                                          measurement_model=self.measurement_model,
                                          timestamp=truth.timestamp,
                                          groundtruth_path=truth,
                                          metadata={"sector": true_sector})
                detections.add(detection)

        return detections

# %%
# Creating simulator
# ------------------
# The ground truth paths can exist in one of 4 sectors. We'll assert that they cannot move between
# sectors. We will add metadata to each path, detailing what sector it belongs to.

from stonesoup.models.transition.linear import ConstantVelocity, \
    CombinedLinearGaussianTransitionModel
from stonesoup.types.state import GaussianState
import datetime
import numpy as np
from stonesoup.simulator.simple import MultiTargetGroundTruthSimulator
from stonesoup.feeder.multi import MultiDataFeeder
from stonesoup.buffered_generator import BufferedGenerator


class GroundTruthSectorFeeder(MultiTargetGroundTruthSimulator):
    sector: Sequence = Property(doc="Sector of produced ground truths")

    @BufferedGenerator.generator_method
    def groundtruth_paths_gen(self):
        for time, groundtruth_paths in super().groundtruth_paths_gen():
            for truth in groundtruth_paths:
                truth.state.metadata = {"sector": self.sector}
            yield time, groundtruth_paths

transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.1),
                                                          ConstantVelocity(0.1)])
now = datetime.datetime.now()
initial_state = GaussianState([0, 0, 0, 0], covar=np.diag([25, 1, 25, 1]), timestamp=now)
groundtruth_A = GroundTruthSectorFeeder(transition_model, initial_state,
                                        number_steps=25, sector="A")
groundtruth_B = GroundTruthSectorFeeder(transition_model, initial_state,
                                        number_steps=25, sector="B")
groundtruth_C = GroundTruthSectorFeeder(transition_model, initial_state,
                                        number_steps=25, sector="C")
groundtruth_D = GroundTruthSectorFeeder(transition_model, initial_state,
                                        number_steps=25, sector="D")
groundtruth_sim = MultiDataFeeder(readers=[groundtruth_A, groundtruth_B,
                                           groundtruth_C, groundtruth_D])

# %%
# To demonstrate the sensor in action, we will use a :class:`~.PlatformDetectionSimulator`.

from stonesoup.platform.base import FixedPlatform
from stonesoup.types.state import State
from stonesoup.simulator.platform import PlatformDetectionSimulator

sensor = SectorLookingSensor(sector="A", sectors=sectors)  # sensor starts by looking at sector "A"

platform = FixedPlatform(position_mapping=(0, 2),
                         states=[State([0, 0, 0, 0], timestamp=now)],
                         sensors=[sensor])

detector = PlatformDetectionSimulator(groundtruth_sim, {platform})

# %%
# The sensor starts the simulation by looking at sector "A". After 20 iterations (4 feeders => 5
# increments of time), we will create an action generator by calling `actions` on the sensor. This
# queries what the sensor can do up to a given timestamp. In this instance, the sensor has one
# modifiable property, so one generator will be returned of type given by the property's
# `generator_cls` (i.e. our `SectorActionsGenerator`). We defined a method `action_from_value`,
# and will use this to create a `ChangeSectorAction` to make the sensor 'look' at sector "B" in 5
# seconds time.
# We will action the sensor to subsequently look at sector "C" after 10 increments of time.

groundtruths = set()
all_detections = set()

for time_index, (time, detections) in enumerate(detector, 1):

    if time_index == 20:
        generator = sensor.actions(
            time + datetime.timedelta(seconds=5)).pop()  # only one action generator
        action = generator.action_from_value("B")  # look at sector B in 5 seconds
        sensor.add_actions({action})
    elif time_index == 40:
        generator = sensor.actions(time + datetime.timedelta(seconds=10)).pop()
        action = generator.action_from_value("C")  # look at sector C in 10 seconds
        sensor.add_actions({action})
    groundtruths.update(groundtruth_sim.current[1])
    all_detections.update(detections)

print(f"{len(groundtruths) = } {len(all_detections) = }")

# %%
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, constrained_layout=True)
fig.suptitle("Sectors", fontsize=16)


def plot_axis(sector, ax):
    ax.set_title(sector, fontsize=16)

    sector_truths = {truth for truth in groundtruths if truth.metadata.get("sector") == sector}
    for truth in sector_truths:
        X = list()
        Y = list()
        for state in truth:
            X.append(state.state_vector[0])
            Y.append(state.state_vector[2])

        ax.plot(X, Y, c="blue")
    ax.figure.set_size_inches(20, 20)

    sector_detections = {detection for detection in all_detections if
                         detection.metadata.get("sector") == sector}
    for detection in sector_detections:
        x, y = detection.state_vector
        ax.scatter(x, y, c="red", marker="*")


for sector, ax in zip(sectors, axes[0]):
    plot_axis(sector, ax)
for sector, ax in zip(sectors[2:], axes[1]):
    plot_axis(sector, ax)
