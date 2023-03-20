
# Non Myopic Sensor Management

By Tom Aldridge, Dstl, 2022-23

This work aims to adapt existing BruteForceSensorManager code in order to perform non-myopic action generation. Non-myopic action generation looks N steps into the future (user specifyable) and uses all available future data to pick the optimal action set leading up to the final time step. 

## Usage/Examples

```python
# Imports
from stonesoup.sensormanager.nonmyopic import BruteForceNonMyopicSensorManager 
from stonesoup.sensormanager.reward import UncertaintyRewardFunction

# Instantiate sensor manager on sensor 'sensorA'
sensormanager = BruteNonMyopicForceSensorManager({sensorA}, reward_function)

''' Set up hypothesiser, data associator etc as usual '''

# Set horizons for the sensor manager
sensormanager.set_horizons(evaluation_horizon = 2, scheduling_horizon = 2)

for timestep in timesteps[1:]:
    
    # Check if no actions scheduled
    if len(sensorA.scheduled_actions == 0):
        
        # Generate future timesteps to pass into our sensor manager
        future_timesteps = sensormanager.get_future_timesteps(timestep, timesteps)
        
        # Get list of actions from sensor manager
        chosen_actions = sensormanager.choose_actions(tracks, future_timesteps)

        # Queue actions to sensor
        for chosen_action in chosen_actions:
            for sensor, actions in chosen_actions.items():
                sensor.add_actions(actions)

    # update measurements as usual
    # update tracks as usual
```


## Explanation

See ```stonesoup/sensormanager/nonmyopic.py``` for code referenced below.

### Classes and Methods
Class ```ConfigWrapper```:
 - Inherits ```NodeMixin``` from ```anytree``` package. 
 - Wraps a ```StoneSoup Config``` object in a tree data structure. Allows accessing the parents and children of each config object, so that the final selected config can be reverse iterated through returning the list of actions to take.

Class ```BruteForceNonMyopicSensorManager```:
 - ```set_horizons```: Sets the evaluation and schedule horizon of the sensor manager. The evaluation horizon is how many steps we look into the future, whilst the schedule horizon is how many steps we schedule into the future. Evaluation horizon must be greater than or equal to the scheduling horizon.
 - ```get_future_timesteps```: Returns a list of timesteps at which the sensor manager will be evaluating. If the number of requested timesteps is greater than the number of timesteps remaining, then only the remaining timesteps are returned.
 - ```choose_actions```: Returns the (list of) action(s) to queue to the sensor. Gets the initial set of action choices, makes this the root of the tree, then calculates next actions to take using the following submethods:
    - ```get_action_choices```: Returns a dictionary containing all combinations of actions for each sensor.
    - ```evaluate_reward_function```: Iterates through all configs and returns the config with maximal reward value, as well as its corresponding reward value.
    - ```calculate_path```: Recursively calculate optimal actions. First, the sensor is updated with the first action chosen. Action choices are then recalculated from this new predicted state. This is repeated until we are at the required depth, where the reward function is evaluated on the final configs. Returns the config with optimal reward value.
    - ```generate_actions```: Given a final node/config in the tree, return the actions taken leading up to and including this config as a list.

## Current Issues

#### Reward Runaway

Typical reward values for a config range between 0 and 1. During stages of this algorithm, reward values can increase dramatically to values in the 100s. This can often happen if there have been multiple steps previously where there is no optimal reward (e.g. no tracks can be seen by the sensor). **Suggested solution**: Adapting reward function to weight sooner rewards with higher values (Bellman Equations). 


