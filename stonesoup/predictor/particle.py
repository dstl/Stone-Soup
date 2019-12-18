# -*- coding: utf-8 -*-
from functools import lru_cache
import numpy as np

from random import random
from .base import Predictor
from ..types.particle import Particle
from ..types.prediction import ParticleStatePrediction
from ..models.transition.linear import ConstantVelocity, ConstantAcceleration, ConstantTurn, CombinedLinearGaussianTransitionModel


class ParticlePredictor(Predictor):
    """ParticlePredictor class

    An implementation of a Particle Filter predictor.
    """

    @lru_cache()
    def predict(self, prior, control_input=None, timestamp=None, **kwargs):
        """Particle Filter prediction step

        Parameters
        ----------
        prior : :class:`~.ParticleState`
            A prior state object
        control_input : :class:`~.State`, optional
            The control input. It will only have an effect if
            :attr:`control_model` is not `None` (the default is `None`)
        timestamp: :class:`datetime.datetime`, optional
            A timestamp signifying when the prediction is performed
            (the default is `None`)

        Returns
        -------
        : :class:`~.ParticleStatePrediction`
            The predicted state
        """
        noise_coeff = [0.1]
        turn = [0.1]

        """When converting between models, a different number of state dimensions are needed ie ConstantVelocity 
        requires position and velocity, constant acceleration requires position, velocity and acceleration,
        this is causing a problem with the matrix multiplication"""

        transition_matrix = np.array([[0.9, 0.1],
                                      [0.1, 0.9]])

        sum_cv = np.cumsum(transition_matrix[0])
        sum_ca = np.cumsum(transition_matrix[1])

        # Compute time_interval
        try:
            time_interval = timestamp - prior.timestamp
        except TypeError:
            # TypeError: (timestamp or prior.timestamp) is None
            time_interval = None

        new_particles = []
        for particle in prior.particles:
            # if the dynamic model value is 1 then set the dynamic model for all positions to be ConstantVelocity
            if particle.dynamic_model == 0:
                # Change the value of the dynamic value randomly according to the above defined transition matrix
                dynamic_model = np.searchsorted(sum_cv, random())
                # Setting the new dynamic model to be only ConstantVelocity models
                model_list = (ConstantVelocity(noise_coeff),
                              ConstantVelocity(noise_coeff),
                              ConstantVelocity(noise_coeff))
                self.transition_model = CombinedLinearGaussianTransitionModel(model_list)
                # Creating a new state vector that does not contain the acceleration state
                pos_vel_particle = np.array([particle.prior_state[i]
                                             for i in range(len(particle.prior_state))
                                             if i not in [2, 5, 8]])
                # Runs the ConstantVelocity model on this new position and velocity particle
                new_state_vector = self.transition_model.function(
                    pos_vel_particle,
                    time_interval=time_interval,
                    **kwargs)

                updated_prior = np.insert(new_state_vector, 2, particle.prior_state[2])
                updated_prior = np.insert(updated_prior, 5, particle.prior_state[5])
                updated_prior = np.insert(updated_prior, 8, particle.prior_state[8])
                updated_prior = np.reshape(updated_prior, (-1, 1))

                # Appends to the new particles, adding this new state vector without the acceleration components and
                # stores the state vector with acceleration as the prior_state attribute
                new_particles.append(
                    Particle(new_state_vector,
                             weight=particle.weight,
                             parent=particle.parent,
                             dynamic_model=dynamic_model,
                             prior_state=updated_prior))

            elif particle.dynamic_model == 1:
                dynamic_model = np.searchsorted(sum_ca, random())
                model_list = (ConstantAcceleration(noise_coeff),
                              ConstantAcceleration(noise_coeff),
                              ConstantAcceleration(noise_coeff))
                self.transition_model = CombinedLinearGaussianTransitionModel(model_list)

                # Runs the ConstantAcceleration model on this new position and velocity particle
                new_state_vector = self.transition_model.function(
                    particle.prior_state,
                    time_interval=time_interval,
                    **kwargs)

                # Appends to the new particles, adding this new state vector without the acceleration components and
                # stores the state vector with acceleration as the prior_state attribute
                new_particles.append(
                    Particle(new_state_vector,
                             weight=particle.weight,
                             parent=particle.parent,
                             dynamic_model=dynamic_model,
                             prior_state=new_state_vector))

        return ParticleStatePrediction(new_particles, timestamp=timestamp)
