from datetime import timedelta
from copy import copy, deepcopy
import numpy as np

from ...base import Property
from ...types.state import State
from ...sensormanager.action import Action
from ...simulator.transition import ConstantJerkSimulator
from .transition_action import TransitionModelActionGenerator


class ConstantJerkAction(Action):
    """The action of moving a platform to a destination using a
    :class:`~.ConstantJerk` transition , via the :class:`~.ConstantJerkSimulator`."""

    target_value: State = Property(doc="Destination state of the platform.")

    def act(self, current_time, end_time, init_value, *args, **kwargs):
        """Propagates the platform position using the :class:`~.ConstantJerkSimulator`.

        Parameters
        ----------
        current_time: :class:`datetime.datetime
            Current time
        end_time: :class:`datetime.datetime
            A timestamp signifying the end of the maneuver
        init_value: Any
            Current platform state

        Returns
        -------
        Any
            New platform state

        Notes
        -----
        This method updates the value of the platform :attr:`states`.
        """

        states = copy(init_value)

        # Time step is one increment in the simulation.
        # This function calculates the state of the platform for one time step
        duration = end_time - current_time

        transition_model = ConstantJerkSimulator(
            position_mapping=self.generator.position_mapping,
            velocity_mapping=self.generator.velocity_mapping,
            init_state=states[-1],
            final_state=self.target_value)

        temp_state = State(
            state_vector=transition_model.function(
                state=states[-1],
                time_interval=duration,
                **kwargs),
            timestamp=end_time
        )

        # Update the platform state
        states.append(temp_state)

        return states


class JerkActionGenerator(TransitionModelActionGenerator):
    """Generates possible actions for moving a platform with a
    :class:`~.ConstantJerk` transition model."""

    @property
    def default_action(self):
        """
        Default action is for platform to stay in the same location.
        """

        state = deepcopy(self.state)
        # Duration set to be how long it takes to decelerate to 0 velocity
        duration = self.constraints[0] / self.constraints[1]
        state.state_vector[self.velocity_mapping, ] = 0
        state.timestamp += timedelta(seconds=duration)
        return self.jerk_action_from_state(state)

    def __iter__(self):
        duration = self.end_time - self.state.timestamp

        for point in self.movement_grid(duration):
            state = State(state_vector=np.zeros(self.ndim),
                          timestamp=self.end_time)
            for i in range(len(self.position_mapping)):
                state.state_vector[self.position_mapping[i]] = point[i]
            yield self.jerk_action_from_state(state)

    @property
    def initial_value(self):
        """Initial value is the current location of the platform."""
        return self.current_value[-1].state_vector

    @property
    def min(self):
        centre, amax, _, _ = self.define_movement_ellipse()
        min_position = centre - amax

        min_state = np.zeros(self.ndim)
        for i, j in enumerate(self.position_mapping):
            min_state[j] = min_position[i]

        return min_state

    @property
    def max(self):
        centre, amax, _, _ = self.define_movement_ellipse()
        max_position = centre + amax

        max_state = np.zeros(self.ndim)
        for i, j in enumerate(self.position_mapping):
            max_state[j] = max_position[i]

        return max_state

    def action_from_value(self, x):
        """
        Generates a :class:`~.ConstantJerkAction` which would enable the platform to reach state x.
        Parameters
        ----------
        x: array
        Returns
        -------
        :class:`~.ConstantJerkAction`
        """

        return self.jerk_action_from_state(State(x, self.end_time))

    def jerk_action_from_state(self, state):
        """
        Generates a :class:`~.ConstantJerkAction` which would enable the platform to
        reach the given state.
        Parameters
        ----------
        state: :class:`~.State`
        Returns
        -------
        :class:`~.ConstantJerkAction`
        """
        end_time = state.timestamp
        init_state = deepcopy(self.state)
        duration = end_time - init_state.timestamp
        point = state.state_vector[self.position_mapping, ]

        if not self.is_reachable(point, duration):
            return None

        return ConstantJerkAction(end_time=end_time,
                                  target_value=state,
                                  generator=self)
