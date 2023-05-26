import itertools
from datetime import datetime

import numpy as np

from stonesoup.custom.sensor.movable import MovableUAVCamera
from stonesoup.types.angle import Angle
from stonesoup.types.array import StateVector

# The camera is initially positioned at x=10, y=10, z=100
position = StateVector([10., 10., 100.])

# We can also set the resolution of each actionable property. The resolution is used when
# discretising the action space. In this case, we set the resolution of both the X and Y locations
# to 10 units.
resolutions = {'location_x': 10., 'location_y': 10.}

# Furthermore, we can specify the limits of the action space. In this case, we set the limits of
# both the X and Y locations to [-100, 100]. This means that the action space will contain values
# in the range [-100, 100] with a step size of 10 units for each property (based on the resolution
# specified above).
limits = {'location_x': [-100, 100], 'location_y': [-100, 100]}

# Create a camera object
sensor = MovableUAVCamera(ndim_state=6, mapping=[0, 2, 4],
                          noise_covar=np.diag([0.05, 0.05, 0.05]),
                          location_x=position[0], location_y=position[1],
                          resolutions=resolutions,
                          position=position,
                          fov_radius=100,
                          limits=limits)

# Set a query time
timestamp = datetime.now()

# Calling sensor.actions() will return a set of action generators. Each action generator is an
# object that contains all the actions that can be performed by the sensor at a given time. In this
# case, the sensor has two actionable properties: X and Y location. Hence, the result of
# sensor.actions() is a set of two action generators: one for moving on the X-axis and one for
# moving on the Y-axis.
action_generators = sensor.actions(timestamp)

# Let's look at the action generators
# The first action generator is for the X location. We can extract the action generator by
# searching for the action generator that controls the 'location_x' property. So, the following
# line of code simply filters the action generators that control 'location_x' (the for-if
# statement) and then selects the first action generator (since there is only one), via the next()
# statement.
x_action_generator = next(ag for ag in action_generators if ag.attribute == 'location_x')
# The second action generator is for the Y location. We can extract the action generator by
# searching for the action generator that controls 'location_y'.
y_action_generator = next(ag for ag in action_generators if ag.attribute == 'location_y')

# We can now look at the actions that can be performed by the action generators. The action
# generators provide a Python "iterator" interface. This means that we can iterate over the action
# generators to get the actions that can be performed (e.g. with a "for" loop). Instead, we can
# also use the list() function to get a list of all the actions that can be performed.
possible_x_actions = list(x_action_generator)
possible_y_actions = list(y_action_generator)

# Each action has a "target_value" property that specifies the value that the property will be
# set to if the action is performed. The following line of code prints the target values of the
# 10th action for pan and tilt.
print(possible_x_actions[9].target_value)
print(possible_y_actions[9].target_value)

# To get all the possible combinations of actions, we can use the itertools.product() function.
possible_action_combinations = list(itertools.product(possible_x_actions, possible_y_actions))

# Let us now select the 10th action combination and task the sensor to perform the action.
chosen_action_combination = possible_action_combinations[9]
sensor.add_actions(chosen_action_combination)
sensor.act(timestamp)

# We can also create a custom action combination. For example, we can move the camera to the
# location (0, 10, 100) by generating an action that sets the X location to 0 and an action that
# sets the Y location to 10. We can then combine these two actions into a single action combination
# and task the sensor to perform the action.
custom_action_x = x_action_generator.action_from_value(0)   # Action that sets the X location to 0
custom_action_y = y_action_generator.action_from_value(10)  # Action that sets the Y location to 10
custom_action_combination = (custom_action_x, custom_action_y)
sensor.add_actions(custom_action_combination)
sensor.act(timestamp)

# The statement below is just an extra statement to allow us to breakpoint the code and inspect
# the possible actions.
end = True
