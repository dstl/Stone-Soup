import itertools
from datetime import datetime

import numpy as np

from stonesoup.custom.sensor.pan_tilt import PanTiltUAVCamera
from stonesoup.types.angle import Angle
from stonesoup.types.array import StateVector

# Specify the rotation offset of the camera
# In this case we rotate the camera around the Y axis by 90 degrees, meaning that the camera is
# pointing downwards
# NOTE: Panning moves the footprint of the camera along the Y axis, and tilting moves the
# footprint along the X axis
rotation_offset = StateVector([Angle(0), Angle(-np.pi / 2), Angle(0)])  # Camera rotation offset

# Specify the initial pan and tilt of the camera
pan = Angle(0)
tilt = Angle(0)

# The camera is positioned at x=10, y=10, z=100
position = StateVector([10., 10., 100.])

# We can also set the resolution of each actionable property. The resolution is used when
# discretising the action space. In this case, we set the resolution of both the pan and tilt to
# 10 degrees, meaning that the action space will contain values in the range [-pi/2, pi/2] with
# a step size of 10 degrees for each property.
# NOTE: Currently, the current state of each property is appended to the action space, meaning
# that the action space will contain 19 values for each property (not 18). In the current example,
# this means that the action for 0 degrees will be duplicated. This is a known "feature" and will
# be fixed in a future release.
resolutions = {'pan': np.radians(10), 'tilt': np.radians(10)}

# Create a camera object
sensor = PanTiltUAVCamera(ndim_state=6, mapping=[0, 2, 4],
                          noise_covar=np.diag([0.05, 0.05, 0.05]),
                          fov_angle=[np.radians(15), np.radians(10)],
                          rotation_offset=rotation_offset,
                          pan=pan, tilt=pan,
                          resolutions=resolutions,
                          position=position)

# Set a query time
timestamp = datetime.now()

# Calling sensor.actions() will return a set of action generators. Each action generator is an
# object that contains all the actions that can be performed by the sensor at a given time. In this
# case, the sensor can perform two actions: pan and tilt. Hence, the result of sensor.actions() is
# a set of two action generators: one for panning and one for tilting.
action_generators = sensor.actions(timestamp)

# Let's look at the action generators
# The first action generator is for panning. We can extract the action generator by searching for
# the action generator that controls the 'pan'. So, the following line of code simply filters the
# action generators that control the 'pan' of the camera (the for-if statement) and then selects
# the first action generator (since there is only one), via the next() statement.
pan_action_generator = next(ag for ag in action_generators if ag.attribute == 'pan')
# The second action generator is for tilting. We can extract the action generator by searching for
# the action generator that controls the 'tilt'.
tilt_action_generator = next(ag for ag in action_generators if ag.attribute == 'tilt')

# We can now look at the actions that can be performed by the action generators. The action
# generators provide a Python "iterator" interface. This means that we can iterate over the action
# generators to get the actions that can be performed (e.g. with a "for" loop). Instead, we can
# also use the list() function to get a list of all the actions that can be performed.
possible_pan_actions = list(pan_action_generator)
possible_tilt_actions = list(tilt_action_generator)

# Each action has a "target_value" property that specifies the value that the property will be
# set to if the action is performed. The following line of code prints the target values of the
# 10th action for pan and tilt.
print(possible_pan_actions[9].target_value)
print(possible_tilt_actions[9].target_value)

# To get all the possible combinations of actions, we can use the itertools.product() function.
possible_action_combinations = list(itertools.product(possible_pan_actions, possible_tilt_actions))

# Let us now select the 10th action combination and task the sensor to perform the action.
chosen_action_combination = possible_action_combinations[9]
sensor.add_actions(chosen_action_combination)
sensor.act(timestamp)

# The statement below is just an extra statement to allow us to breakpoint the code and inspect
# the possible actions.
end = True