# -*- coding: utf-8 -*-

import numpy as np
from typing import Set, Union, Callable, Tuple

from ..base import Property
from .base import MeasurementModel
from ...types.detection import Clutter
from ...types.groundtruth import GroundTruthState
from ...types.array import StateVector, StateVectors
from ...types.numeric import Probability
from ...types.state import State

class ClutterModel(MeasurementModel):
    """A simulator that generates sensor clutter (false alarms) according to a specified 
    distribution in 2D space relative to the sensor's position.

    Note
    ----
    This implementation of this class assumes a 3D Cartesian space.
    """

    clutter_rate: float = Property(
        default=1.0,
        doc="The average number of clutter points per time step. The actual "
            "number is Poisson distributed")
    distribution: Callable = Property(
        default=np.random.default_rng().uniform,
        doc="A function which represents the distribution of the clutter over the "
            "measurement space. The function should return a single value (ie, do "
            "not use multivariate distributions).")
    dist_params: Tuple = Property(
        default=((-200, 200), (-200, 200)),
        doc="The required parameters for the clutter distribution function. The "
        "length of the list must be equal to the number of dimensions (`self.ndim_meas`) "
        "and should be defined for use in Cartesian space."
        "The default defines the space for a uniform distribution in 2D. The call "
        "`np.array([self.distribution(*arg) for arg in self.dist_params])` "
        "must return a numpy array of length equal to the number of dimensions.")


    def function(self, ground_truths: Set[GroundTruthState], **kwargs) -> Set[Clutter]:
        """
        Use the defined distribution and parameters to create simulated clutter
        for the current time step. Return this clutter to the calling sensor so
        that it can be added to the measurements.

        Parameters
        ----------
        ground_truths : a set of :class:`~.GroundTruthState`
            The truth states which exist at this time step.
        
        Returns
        -------
        : set of :class:`~.Clutter`
            The simulated clutter.
        """
        # Extract the timestamp from the ground_truths. Groundtruth is
        # necessary to get the proper timestamp. If there is no
        # groundtruth return a set of no Clutter.
        if not ground_truths:
            return set()

        # Generate the clutter for this time step
        timestamp = next(iter(ground_truths)).timestamp
        clutter = set()
        for _ in range(np.random.poisson(self.clutter_rate)):
            # Call the distribution function to generate a random vector in the space
            random_vector = np.array([self.distribution(*arg) for arg in self.dist_params])
            
            # Make a State object with the random vector
            state = State([0.0] * self.measurement_model.ndim_state, timestamp=timestamp)
            state.state_vector[self.measurement_model.mapping, 0] += random_vector 

            # Use the sensor's measurement model to incorporate the
            # translation offset and sensor rotation. This will also
            # convert the vector to the proper measurement space
            # (polar or spherical coordinates)
            clutter_vector = self.measurement_model.function(state)

            # Create a clutter object.
            clutter.add(Clutter(state_vector=clutter_vector, 
                                timestamp=timestamp,
                                measurement_model=self.measurement_model))

        return clutter
    
    
    @property
    def ndim_meas(self) -> int:
        """ndim_meas getter method

        Returns
        -------
        :class:`int`
            The number of measurement dimensions. Equal to the number of dimensions in the 
            Sensor's measurement model.
        """
        return self.measurement_model.ndim_state

    def rvs(self, num_samples: int = 1, **kwargs) -> Union[StateVector, StateVectors]:
        """
        Must be implemented to properly inherit the parent Model.
        """
        return None 
    
    def pdf(self, state1: State, state2: State, **kwargs) -> Union[Probability, np.ndarray]:
        """
        Must be implemented to properly inherit the parent Model.
        """
        return None
