# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import uniform
from typing import Set, Union

from ..base import Property
from .base import MeasurementModel
from ...types.detection import Clutter
from ...types.angle import Bearing, Elevation
from ...types.groundtruth import GroundTruthState
from ...types.array import CovarianceMatrix
from ...functions import cart2pol, cart2sphere
from ...types.array import StateVector, StateVectors
from ...types.numeric import Probability
from ...types.state import State


class ClutterBearingRange(MeasurementModel):
    """A simulation that generates sensor clutter (false alarms) according to a specified 
    distribution in 2D space relative to the sensor's position.

    Note
    ----
    This implementation of this class assumes a 3D Cartesian space.
    """

    clutter_rate: float = Property(
        default=1.0,
        doc="The average number of clutter points per time step. The actual "
            "number is Poisson distributed")
    distribution: np.string_ = Property(
        default='uniform',
        doc="The type of distribution that the clutter follows. Supported "
            "distributions are: uniform, normal (Gaussian)")
    state_space: list = Property(
        default=None,
        doc="A 2-D list in the form of math:`[[x_min, dx], [y_min, dy]]`. These give the "
            "dimension bounds in the Cartesian space. For example, if the sensor is "
            "positioned at -100 in the x-axis and can 'see' for 200m, then "
            "math:`[x_min, dx] = [-100, 200]`. This is only needed when the distribution "
            "is uniform.")
    mean: list = Property(
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
        if self.distribution == 'uniform' and self.state_space is None:
            raise ValueError("To use the uniform distribution, the state_space parameter must be defined")
        elif self.distribution == 'normal' and self.mean is None and self.covariance is None:
            raise ValueError("To use the normal distribution, the mean and covariance parameters must be defined")
        elif self.distribution not in ['uniform', 'normal']:
            raise ValueError("The clutter model currently supports only uniform or normal (Gaussian) distributions.")

    def function(self, ground_truths: Set[GroundTruthState], position=None, **kwargs) -> Set[Clutter]:
        """
        Use the defined distribution and parameters to create simulated clutter
        for the current time step. Return this clutter to the calling sensor so
        that it can be added to the measurements.

        Parameters
        ----------
        ground_truths : a set of :class:`~.GroundTruthState`
            The truth states which exist at this time step.
        position : a 2x1 :class:`~.StateVector` of Cartesian coordinates
            The position of the sensor which uses this model.
        
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
            if self.distribution == 'uniform':
                random_vector = self.generate_uniform_vector()
            else:
                random_vector = self.generate_normal_vector()
            
            # Subtract the mounting sensor's position from the clutter,
            # then convert to polar coordinates.
            if position is not None:
                clutter_vector = random_vector - position[0:2].flatten()

            rho, phi = cart2pol(*clutter_vector)
            clutter_vector = np.array([Bearing(phi), rho])
            
            # Create a clutter object. The measurement_model is
            # inherited from the Detector on which the sensor 
            # and platform are mounted once it is used.
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
            A uniform random vector with length 2.
        """
        x = uniform.rvs(self.state_space[0][0], self.state_space[0][1])
        y = uniform.rvs(self.state_space[1][0], self.state_space[1][1])

        return np.array([x, y])
        

    def generate_normal_vector(self):
        """Generate a random vector in the normal distribution according
        to the model's defined mean and covariance.

        Returns
        -------
        : `list`
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
            The number of measurement dimensions
        """

        return 2

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

    state_space: list = Property(
        default=None,
        doc="A 2-D list in the form of math:`[[x_min, dx], [y_min, dy], [z_min, dz]]`. "
            "These give the dimension bounds in the Cartesian space. For example, if the sensor "
            "is positioned at -100 in the x-axis and can 'see' for 200m, then "
            "math:`[x_min, dx] = [-100, 200]`. This is only needed when the distribution is uniform.")
    mean: list = Property(
        default=None,
        doc="The mean of the distribution for normally-distributed clutter. The "
        "mean is defined in Cartesian space and must have length 3."
        "This parameter is only needed when using the normal distribution.")
    covariance: CovarianceMatrix = Property(
        default=None,
        doc="Covariance matrix of the distribution for normally-distributed clutter. "
        "It must be a 3x3 matrix. This parameter is only needed when using the normal "
        "distribution.")

    def function(self, ground_truths: Set[GroundTruthState], position=None, **kwargs) -> Set[Clutter]:
        """
        Use the defined distribution and parameters to create simulated clutter
        for the current time step. Return this clutter to the calling sensor so
        that it can be added to the measurements.

        Parameters
        ----------
        ground_truths : a set of :class:`~.GroundTruthState`
            The truth states which exist at this time step.
        position : a 3x1 :class:`~.StateVector` of Cartesian coordinates
            The position of the sensor which uses this model.
        
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
        timestamp = list(ground_truths)[0].timestamp
        clutter = set()
        for _ in range(np.random.poisson(self.clutter_rate)):
            # Call the appropriate helper function according to 
            # the chosen distribution
            if self.distribution == 'uniform':
                random_vector = self.generate_uniform_vector()
            else:
                random_vector = self.generate_normal_vector()
            
            # Subtract the mounting sensor's position from the clutter,
            # then convert to spherical coordinates.
            if position is not None:
                clutter_vector = random_vector - position[0:3].flatten()

            rho, phi, theta = cart2sphere(*clutter_vector)
            clutter_vector = np.array([Elevation(theta), Bearing(phi), rho])

            # Create a clutter object. The measurement_model is
            # inherited from the Detector on which the sensor 
            # and platform are mounted once it is used.
            clutter.add(Clutter(state_vector=clutter_vector, 
                                timestamp=timestamp,
                                measurement_model=self.measurement_model))

        return clutter
    
    def generate_uniform_vector(self):
        """Generate a random vector in the uniform distribution according
        to the sensor's defined state space.

        Returns
        -------
        : :class:`numpy.array`
            A uniform random vector with length 3.
        """
        x = uniform.rvs(self.state_space[0][0], self.state_space[0][1])
        y = uniform.rvs(self.state_space[1][0], self.state_space[1][1])
        z = uniform.rvs(self.state_space[2][0], self.state_space[2][1])
        return np.array([x, y, z])
    
    @property
    def ndim_meas(self) -> int:
        """ndim_meas getter method

        Returns
        -------
        :class:`int`
            The number of measurement dimensions
        """

        return 3
