# -*- coding: utf-8 -*-

import numpy as np
from typing import Set, Union, Tuple
from enum import Enum

from ..base import Property
from .base import MeasurementModel
from ...types.detection import Clutter
from ...types.groundtruth import GroundTruthState
from ...types.array import CovarianceMatrix, StateVector, StateVectors
from ...types.numeric import Probability
from ...types.state import State

class Distribution(Enum):
    """
    Explicitly define the distributions that are supported for the clutter model.
    """
    uniform = "uniform"
    normal = "normal"

class ClutterBearingRange(MeasurementModel):
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
    distribution: Distribution = Property(
        default=Distribution.uniform,
        doc="The type of distribution that the clutter follows. Supported "
            "types are described by the Enum Distribution class.")
    state_space: Tuple[Tuple[float, float], Tuple[float, float]] = Property(
        default=None,
        doc="A 2-D list in the form of math:`[[x_min, dx], [y_min, dy]]`. These give the "
            "dimension bounds in the Cartesian space. For example, if the sensor is "
            "positioned at -100 in the x-axis and can 'see' for 200m, then "
            "math:`[x_min, dx] = [-100, 200]`. This is only needed when the distribution "
            "is uniform.")
    mean: Tuple[float, float, float] = Property(
        default=None,
        doc="The mean of the distribution for normally-distributed clutter. The "
        "mean is defined in Cartesian space and must have length 2."
        "This parameter is only needed when using the normal distribution.")
    covariance: CovarianceMatrix = Property(
        default=None,
        doc="Covariance matrix of the distribution for normally-distributed clutter. "
        "It must be a 2x2 matrix. This parameter is only needed when using the normal "
        "distribution.")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Check that the chosen distribution matches the parameters given
        if self.distribution == Distribution.uniform and self.state_space is None:
            raise ValueError("To use the uniform distribution, the state_space parameter must be defined")
        elif self.distribution == Distribution.normal and self.mean is None and self.covariance is None:
            raise ValueError("To use the normal distribution, the mean and covariance parameters must be defined")

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
            if self.distribution == Distribution.uniform:
                random_vector = self.generate_uniform_vector()
            else:
                random_vector = self.generate_normal_vector()
            
            # Make a State object with the random vector
            state = State([0.0] * self.measurement_model.ndim_state, timestamp=timestamp)
            state.state_vector[self.measurement_model.mapping, 0] += random_vector 

            # Use the sensor's measurement model to incorporate the
            # translation offset and sensor rotation. This will also
            # convert the vector to the proper measurement space
            # (polar coordinates). 
            clutter_vector = self.measurement_model.function(state)

            # Create a clutter object.
            clutter.add(Clutter(state_vector=clutter_vector, 
                                timestamp=timestamp,
                                measurement_model=self.measurement_model))

        return clutter
    
    def generate_uniform_vector(self):
        """Generate a random vector in the uniform distribution according
        to the model's defined state space.

        Returns
        -------
        : :class:`numpy.array`
            A uniform random vector with length equal to self.ndim_meas.
        """
        arr = np.array([np.random.default_rng().uniform(*space) 
                for space in self.state_space])
        return arr
        

    def generate_normal_vector(self):
        """Generate a random vector in the normal distribution according
        to the model's defined mean and covariance.

        Returns
        -------
        : :class:`numpy.array`
            A normal random vector. Has the same length as the mean.
        """
        # Generate a multivariate random normal vector according to our parameters
        random_vector = np.random.default_rng().multivariate_normal(self.mean, self.covariance)
        
        return random_vector
    
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


class ClutterElevationBearingRange(ClutterBearingRange):
    """A simulation that generates sensor clutter (false alarms) according to a specified 
    distribution in 3D space relative to the sensor's position.

    Note
    ----
    This implementation of this class assumes a 3D Cartesian space.
    """

    state_space: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = Property(
        default=None,
        doc="A 2-D list in the form of math:`[[x_min, dx], [y_min, dy], [z_min, dz]]`. "
            "These give the dimension bounds in the Cartesian space. For example, if the sensor "
            "is positioned at -100 in the x-axis and can 'see' for 200m, then "
            "math:`[x_min, dx] = [-100, 200]`. This is only needed when the distribution is uniform.")
    mean: Tuple[float, float, float] = Property(
        default=None,
        doc="The mean of the distribution for normally-distributed clutter. The "
        "mean is defined in Cartesian space and must have length 3."
        "This parameter is only needed when using the normal distribution.")
    covariance: CovarianceMatrix = Property(
        default=None,
        doc="Covariance matrix of the distribution for normally-distributed clutter. "
        "It must be a 3x3 matrix. This parameter is only needed when using the normal "
        "distribution.")
